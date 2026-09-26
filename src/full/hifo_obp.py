"""Reproduce HiFo-Prompt on the LLM4AD Online Bin Packing task.

Companion to ``reprod/eoh_obp.py`` — same CLI / LLM backends / logging surface,
but it drives the **HiFo-Prompt package's native engine** instead of LLM4AD's
EoH. HiFo = EoH backbone + two prompt-augmenting modules:
    - InsightPool (hindsight): a bandit-managed pool of natural-language design
      tips, credited by offspring performance;
    - EvolutionaryNavigator (foresight): a stagnation/diversity-driven regime +
      design directive;
plus the ``m3`` generalization operator (operators default to e1,e2,m1,m2,m3).

To keep it 1:1 comparable with ``eoh_obp.py`` we plug the SAME pieces into HiFo:
    - Evaluator: an adapter (``_OBPAdapter``) exposing HiFo's problem interface
      (``.prompts`` + ``.evaluate(code)``) over LLM4AD's ``OBPEvaluation`` — same
      Weibull instances / n_items / capacity / --seed as eoh_obp.py.
    - LLM: HiFo's internal ``InterfaceLLM`` is monkeypatched to a bridge over the
      SAME ``CachedLLM`` backends (openrouter/ollama/mistral/vllm), with a
      per-call cache salt so identical prompts (e.g. i1 init) still diversify.
    - Parallelism: HiFo parallelises offspring with ``joblib.Parallel``; we force
      the *threading* backend so the CachedLLM bridge is shared (not pickled),
      matching how EoH already shares one CachedLLM across sampler threads.
    - Logging: HiFo's per-generation population dumps are post-processed into
      eoh_obp-style ``trajectory.json`` + ``heuristics.json`` (+ ``args.yaml`` /
      ``terminal.txt`` / run-tag). HiFo's own ``hifo_prompt_log.json`` (insights /
      navigator guidance) is kept alongside.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import random
import sys
import threading
import time
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "packages" / "HiFo-Prompt" / "hifo" / "src"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, make_wandb_logger, OpenRouterClient, OllamaClient,
                   MistralClient, vLLMClient)
from utils.obp_utils import _obp_lower_bound
from utils.logger import make_log_dir, mirror_stdout_to

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.base import SecureEvaluator

# HiFo-Prompt package (native engine)
from hifo.hifo import EVOL
from hifo.utils.getParas import Paras
import hifo.methods.hifo.hifo_evolution as _hifo_evolution
import hifo.methods.hifo.hifo_hp as _hifo_hp


# Bash-script defaults (mirror reprod/eoh_obp.py) — used for the folder-name tag.
_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 20,
    "max_generations": 20,
    "n_instances": 25,
    "n_items": 5000,
    "capacity": 100,
    "selection_num": 5,
    "timeout_seconds": 30,
    "temperature": 0.9,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
    # --- Layer B: HiFo method internals (paper defaults; see hifo_hp.py) ----- #
    # Insight Pool (Hindsight)
    "pool_capacity": 30,
    "novelty_threshold": 0.7,
    "selection_count": 3,
    "usage_penalty_weight": 0.1,
    "recency_bonus": 0.2,
    "recency_window": 2,
    "ema_alpha": 0.3,
    "decay_rate": 0.01,
    "probation_usage": 3,
    # Credit-assignment tiers (Eq. 3 intercepts)
    "credit_best": 0.8,
    "credit_inc": 0.2,
    "credit_pen": -0.3,
    # Evolutionary Navigator (Foresight)
    "progress_eps": 1e-4,
    "stagnation_threshold": 3,
    "progress_threshold": 2,
    "diversity_threshold": 0.3,
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "pop_size": "ps",
    "max_generations": "mg",
    "n_instances": "ni",
    "n_items": "nit",
    "capacity": "cap",
    "selection_num": "sn",
    "timeout_seconds": "to",
    "temperature": "temp",
    "num_threads": "nt",
    "num_cores": "nc",
    "llm_model": "model",
    "llm_backend": "llm",
    # Layer B — HiFo internals
    "pool_capacity": "pcap",
    "novelty_threshold": "nov",
    "selection_count": "sc",
    "usage_penalty_weight": "wu",
    "recency_bonus": "rb",
    "recency_window": "rw",
    "ema_alpha": "ema",
    "decay_rate": "dr",
    "probation_usage": "pu",
    "credit_best": "cb",
    "credit_inc": "cinc",
    "credit_pen": "cpen",
    "progress_eps": "peps",
    "stagnation_threshold": "stag",
    "progress_threshold": "prog",
    "diversity_threshold": "divt",
}


def _run_tag(args: argparse.Namespace) -> str:
    """Compact folder-name tag encoding args that differ from the defaults."""
    overrides = []
    for key, default_val in _BASH_DEFAULTS.items():
        actual_val = getattr(args, key, None)
        if actual_val == default_val:
            continue
        short = _ABBREV.get(key, key)
        if key == "llm_model" and isinstance(actual_val, str) and "/" in actual_val:
            actual_val = actual_val.split("/")[-1]
        if isinstance(actual_val, bool):
            overrides.append(short if actual_val else f"no{short}")
        else:
            overrides.append(f"{short}{actual_val}")
    return "_".join(overrides) if overrides else "default"


def _mean_lb(datasets: dict) -> float:
    lbs = [_obp_lower_bound(datasets[name]["items"], datasets[name]["capacity"])
           for name in datasets]
    return float(np.mean(lbs))


# --------------------------------------------------------------------------- #
# LLM bridge: route HiFo's InterfaceLLM.get_response() through our CachedLLM.
# --------------------------------------------------------------------------- #
class _HiFoLLMBridge:
    """Exposes HiFo's ``get_response(prompt_str) -> str`` over a ``CachedLLM``.

    A per-call incrementing cache salt (like ``utils.llm._draw_sample``) ensures
    identical prompts — notably HiFo's ``i1`` initialisation, where every sample
    shares one prompt — still yield diverse (and separately-cached) completions."""

    def __init__(self, cached: CachedLLM, temperature: float, max_tokens: int,
                 verbose: bool = True):
        self._cached = cached
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._n = 0
        self._inflight = 0
        self._ok = 0
        self._fail = 0
        self._verbose = verbose
        self._lock = threading.Lock()

    def get_response(self, prompt_content) -> str:
        # Every HiFo LLM call funnels through here, so this is the one place to make
        # the otherwise-silent generation loop observable: log start/finish, wall
        # time, response size, running success/fail tally, and in-flight concurrency
        # for each individual query (calls run concurrently across --num-threads, so
        # each line is tagged with its call number #N to stay traceable).
        with self._lock:
            self._n += 1
            n = self._n
            self._inflight += 1
            inflight = self._inflight
        salt = f"hifo-call-{n}"
        prompt_str = str(prompt_content)
        if self._verbose:
            print(f"[llm] #{n} start  (inflight={inflight}, prompt={len(prompt_str)} chars)",
                  flush=True)
        messages = [{"role": "user", "content": prompt_str}]
        t0 = time.time()
        try:
            resp = self._cached.chat(
                messages, temperature=self._temperature,
                max_tokens=self._max_tokens, cache_salt=salt,
            )
        except Exception as e:
            dt = time.time() - t0
            with self._lock:
                self._fail += 1
                self._inflight -= 1
                ok, fail = self._ok, self._fail
            print(f"[llm] #{n} FAILED in {dt:6.1f}s: {type(e).__name__}: {e}  "
                  f"(ok={ok} fail={fail})", flush=True)
            raise
        dt = time.time() - t0
        n_chars = len(resp) if isinstance(resp, str) else 0
        with self._lock:
            self._ok += 1
            self._inflight -= 1
            ok, fail = self._ok, self._fail
        if self._verbose:
            empty = "  [EMPTY RESPONSE]" if n_chars == 0 else ""
            print(f"[llm] #{n} ok     in {dt:6.1f}s ({n_chars} chars){empty}  "
                  f"(ok={ok} fail={fail})", flush=True)
        return resp


# --------------------------------------------------------------------------- #
# Problem adapter: HiFo's problem interface (.prompts + .evaluate) over
# LLM4AD's OBPEvaluation, so HiFo scores on the SAME task/instances as eoh_obp.
# --------------------------------------------------------------------------- #
class _OBPPrompts:
    """GetPrompts-compatible prompt spec for OBP, using OBPEvaluation's ``priority``
    convention (function ``priority(item, bins) -> scores``). ``func_outputs`` is
    the RETURN-VARIABLE name (distinct from the function name) — HiFo reconstructs
    code as ``def priority(...) ... return scores``."""

    def __init__(self):
        self.prompt_task = (
            "Given a set of identical bins with a fixed capacity and a stream of "
            "items arriving one at a time, assign each incoming item to one feasible "
            "bin so as to minimize the total number of bins used. For the current "
            "item, score every feasible bin (a bin whose remaining capacity is at "
            "least the item size) and place the item in the bin with the highest score."
        )
        self.prompt_func_name = "priority"
        self.prompt_func_inputs = ["item", "bins"]
        self.prompt_func_outputs = ["scores"]
        self.prompt_inout_inf = (
            "'item' is the size of the current item; 'bins' is a Numpy array of the "
            "remaining capacities of the feasible bins (each entry >= item). Return "
            "'scores', a Numpy array of the same length as 'bins' giving the priority "
            "of assigning the item to each bin (higher = preferred)."
        )
        self.prompt_other_inf = (
            "Note that 'item' is an int while 'bins' and 'scores' are Numpy arrays. "
            "The function should be non-trivial to achieve strong performance, and "
            "self-consistent. Include 'import numpy as np' at the top of the code."
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


class _OBPAdapter:
    """HiFo problem object over LLM4AD ``OBPEvaluation``.

    HiFo calls ``.evaluate(code_string)`` and minimises the returned objective, so
    we return the **mean number of bins (lower = better)**.

    Crucially, the candidate is run through LLM4AD's ``SecureEvaluator`` — the SAME
    path ``eoh_obp.py`` uses (EoH wraps its ``OBPEvaluation`` in a ``SecureEvaluator``
    internally). It executes each generated ``priority`` in a SEPARATE PROCESS that is
    ``terminate()``/``kill()``-ed at ``OBPEvaluation.timeout_seconds``. A direct
    ``OBPEvaluation.evaluate(fn)`` call (the previous approach) ran the candidate
    IN-PROCESS with no enforceable timeout, so a slow/runaway heuristic hung the HiFo
    joblib worker (``ThreadPoolExecutor`` shutdown blocks on the un-killable thread)
    and tripped the per-offspring budget -> "Parallel time out" -> empty population.
    Routing through ``SecureEvaluator`` makes a slow candidate simply score ``None``
    and the run continue, exactly like EoH. Returns ``None`` on failure/timeout."""

    def __init__(self, evaluation: OBPEvaluation):
        self._obp = evaluation
        # Process-based, killable evaluator (honours evaluation.timeout_seconds).
        self._secure = SecureEvaluator(evaluation, debug_mode=False)
        self.prompts = _OBPPrompts()
        self._lock = threading.Lock()
        self.n_evals = 0

    def evaluate(self, code_string):
        try:
            # SecureEvaluator runs `priority` in a subprocess and hard-kills it at
            # timeout_seconds; returns OBPEvaluation.evaluate's -mean(num_bins), or
            # None on timeout / crash / unparseable code.
            neg_mean_bins = self._secure.evaluate_program(code_string)
        except Exception:
            return None
        if neg_mean_bins is None or not math.isfinite(float(neg_mean_bins)):
            return None
        with self._lock:
            self.n_evals += 1
        return float(-neg_mean_bins)                  # mean bins (lower is better)


# --------------------------------------------------------------------------- #
# Logging: HiFo population dumps -> eoh_obp-style trajectory.json + heuristics.json
# --------------------------------------------------------------------------- #
_ARTIFACT_REFRESH_SECONDS = 30  # how often the background watcher rebuilds the logs


def _build_eoh_style_logs(log_dir: pathlib.Path, label: str, mean_lb: float,
                          n_instances: int, n_operators: int, pop_size: int,
                          n_init_batches: int = 2, wandb_logger=None,
                          verbose: bool = True) -> None:
    """Post-process HiFo's per-generation population dumps
    (``results/pops/population_generation_*.json``) into the same log surface
    ``eoh_obp.py`` writes: a per-generation best-so-far ``trajectory.json`` and a
    ``heuristics.json`` of the (deduplicated) candidates seen.

    Each dumped individual carries ``objective`` = mean bins (lower better, from
    ``_OBPAdapter.evaluate``). ``score`` = incumbent mean bins;
    ``gap`` = (score - mean_lb) / mean_lb.

    ``used_budget`` = cumulative candidate EVALUATIONS x n_instances, matching
    ``eoh_obp.py`` (which counts all sampled candidates). It is derived from HiFo's
    generation STRUCTURE, not from the dump sizes: the dumps hold only the survivors
    (after ``population_management`` drops None-objective candidates, dedups by
    objective, and keeps the ``pop_size`` best), so counting them would badly
    undercount the real budget. HiFo evaluates ``n_init_batches * pop_size``
    offspring for the initial population (gid 0) and ``n_operators * pop_size`` per
    later generation (one ``pop_size`` batch for each of the e1/e2/m1/m2/m3
    operators)."""
    pops_dir = log_dir / "results" / "pops"
    dumps = sorted(
        pops_dir.glob("population_generation_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    code_to_cand: dict = {}
    heuristics: list = []
    per_gen: dict = {}
    for path in dumps:
        gid = int(path.stem.rsplit("_", 1)[-1])
        try:
            pop = json.load(open(path))
        except Exception:
            continue
        per_gen[gid] = []
        for i, ind in enumerate(pop):
            code = ind.get("code") or ""
            obj = ind.get("objective")
            mean_bins = float(obj) if isinstance(obj, (int, float)) else float("inf")
            if code not in code_to_cand:
                cand_id = f"gen{gid:02d}_cand{i:02d}"
                code_to_cand[code] = cand_id
                heuristics.append({"cand_id": cand_id, "gen_id": gid,
                                   "score": mean_bins, "source": code})
            per_gen[gid].append((code, mean_bins))

    # heuristics.json: prefer the FULL offspring record (all n_operators*pop_size
    # candidates/gen, gen<g>_cand00..) captured in all_candidates.jsonl, so it matches
    # EoH's heuristics.json instead of only the survivors in HiFo's population dumps.
    # (trajectory.json still uses the survivor dumps above for the best-so-far incumbent.)
    from utils.hifo_candidate_log import build_heuristics_from_candidate_log
    _full = build_heuristics_from_candidate_log(log_dir, n_init_batches, n_operators, pop_size)
    if _full is not None:
        heuristics = _full

    # Map incumbent CODE -> the cand_id it carries in heuristics.json, so the trajectory's
    # incumbent resolves to the SAME source _final_eval re-evaluates. With the full offspring
    # record, heuristics.json's cand_ids come from all_candidates.jsonl — a DIFFERENT namespace
    # than the survivor-dump `code_to_cand`; using the dump cand_id would point the incumbent at
    # a different heuristic (exploded valid scores). Falls back to the dump map for old runs.
    if _full is not None:
        _inc_cand_by_code: dict = {}
        for _h in _full:
            _inc_cand_by_code.setdefault(_h["source"], _h["cand_id"])
    else:
        _inc_cand_by_code = code_to_cand

    trajectory: list = []
    best = float("inf")
    best_code = None
    for gid in sorted(per_gen):
        for code, mean_bins in per_gen[gid]:
            if math.isfinite(mean_bins) and mean_bins < best:
                best, best_code = mean_bins, code
        best_cand = _inc_cand_by_code.get(best_code) if best_code is not None else None
        score = best if best_code is not None else float("inf")
        gap = ((score - mean_lb) / mean_lb
               if (best_code is not None and mean_lb > 0) else float("inf"))
        # Candidates EVALUATED through this generation (attempts, incl. failed/dupes),
        # from HiFo's structure -- NOT the survivor count in the dumps.
        cand_evaluated = n_init_batches * pop_size + gid * n_operators * pop_size
        trajectory.append({
            "gen_id": gid,
            "cand_id": best_cand,
            "score": float(score),
            "gap": float(gap),
            "n_instances": n_instances,
            "used_budget": cand_evaluated * n_instances,
        })

    # Atomic-ish writes (write to a temp then replace) so a reader/analysis never
    # sees a half-written file while the watcher rebuilds mid-run.
    for name, payload in (("trajectory.json", {"label": label, "trajectory": trajectory}),
                          ("heuristics.json", {"label": label,
                                               "total_sampled": len(heuristics),
                                               "heuristics": heuristics})):
        tmp = log_dir / (name + ".tmp")
        json.dump(payload, open(tmp, "w"), indent=2)
        os.replace(tmp, log_dir / name)
    if verbose:
        print(f"  logged -> {log_dir / 'trajectory.json'}  ({len(trajectory)} generations)")
        print(f"  logged -> {log_dir / 'heuristics.json'}  ({len(heuristics)} candidates)")
    if wandb_logger is not None:
        for row in trajectory:
            wandb_logger.log_trajectory_row(row)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce HiFo-Prompt on Online Bin Packing.")
    p.add_argument("--pop-size", type=int, default=10, help="HiFo ec_pop_size (population size). "
                   "10 matches the wrappers and hifo_obp.json (fix-init loads 2*pop_size=20).")
    p.add_argument("--max-generations", type=int, default=20, help="HiFo ec_n_pop (number of generations).")
    p.add_argument("--n-instances", type=int, default=5, help="Number of Weibull OBP instances.")
    p.add_argument("--n-items", type=int, default=5000, help="Items per instance.")
    p.add_argument("--capacity", type=int, default=100, help="Bin capacity.")
    p.add_argument("--selection-num", type=int, default=5,
                   help="HiFo ec_m: number of parents for e1/e2 (>=2).")
    p.add_argument("--timeout-seconds", type=int, default=30,
                   help="Per-candidate eval wall-clock cap (HiFo eva_timeout).")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="LLM sampling temperature for the HiFo prompts.")

    # --- Layer B: HiFo method internals (default == paper config; injected
    #     into hifo.methods.hifo.hifo_hp before the engine runs) -------------- #
    g_ip = p.add_argument_group("HiFo Insight Pool (Hindsight)")
    g_ip.add_argument("--pool-capacity", type=int, default=30,
                      help="C_pool: max insights retained (paper 30).")
    g_ip.add_argument("--novelty-threshold", type=float, default=0.7,
                      help="theta_novelty: Jaccard admission cutoff (paper 0.7).")
    g_ip.add_argument("--selection-count", type=int, default=3,
                      help="s: top-s insights injected per generation (paper 3).")
    g_ip.add_argument("--usage-penalty-weight", type=float, default=0.1,
                      help="w_u: usage-penalty coefficient (paper 0.1).")
    g_ip.add_argument("--recency-bonus", type=float, default=0.2,
                      help="tau_r: recency bonus magnitude (paper 0.2).")
    g_ip.add_argument("--recency-window", type=int, default=2,
                      help="T_w: recency bonus window in generations (paper 2).")
    g_ip.add_argument("--ema-alpha", type=float, default=0.3,
                      help="alpha: EMA rate for effectiveness (paper 0.3).")
    g_ip.add_argument("--decay-rate", type=float, default=0.01,
                      help="R_decay: eviction-score time decay (paper 0.01).")
    g_ip.add_argument("--probation-usage", type=int, default=3,
                      help="T_usage: uses below which an insight is eviction-immune (paper 3).")

    g_cr = p.add_argument_group("HiFo credit-assignment tiers (Eq. 3 intercepts)")
    g_cr.add_argument("--credit-best", type=float, default=0.8,
                      help="beta_best: intercept when offspring beats the best (paper 0.8).")
    g_cr.add_argument("--credit-inc", type=float, default=0.2,
                      help="beta_inc: intercept when offspring beats the average (paper 0.2).")
    g_cr.add_argument("--credit-pen", type=float, default=-0.3,
                      help="beta_pen: intercept when offspring is below average (paper -0.3).")

    g_nav = p.add_argument_group("HiFo Evolutionary Navigator (Foresight)")
    g_nav.add_argument("--progress-eps", type=float, default=1e-4,
                       help="delta g: min best-fitness gain counted as progress (paper 1e-4).")
    g_nav.add_argument("--stagnation-threshold", type=int, default=3,
                       help="tau_stag: stagnant gens before forcing Explore (paper 3).")
    g_nav.add_argument("--progress-threshold", type=int, default=2,
                       help="tau_prog: progress gens before forcing Exploit (paper 2).")
    g_nav.add_argument("--diversity-threshold", type=float, default=0.3,
                       help="delta_p: phenotypic-diversity floor for Explore (paper 0.3).")

    p.add_argument("--label", type=str, default="hifo/obp")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm"],
                   help="LLM backend (same options as reprod/eoh_obp.py).")
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next",
                   help="Model id passed to the selected backend.")
    p.add_argument("--ollama-host", type=str, default=None,
                   help="Ollama host:port (only for --llm-backend=ollama).")
    p.add_argument("--num-threads", type=int, default=2,
                   help="HiFo exp_n_proc: parallel offspring workers (threading backend).")
    p.add_argument("--num-cores", type=int, default=2,
                   help="Kept for CLI parity with eoh_obp; HiFo uses one (threaded) "
                        "offspring pool sized by --num-threads.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-wandb", action="store_true", default=False,
                   help="Log the trajectory to Weights & Biases.")
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed HiFo gen-0 from the paired fixed population in src/init_pop/ "
                        "(2*pop_size heuristics, re-evaluated this run) then run normal "
                        "population_management, instead of sampling from the LLM.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    main_t0 = time.time()
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)

    now = _dt.datetime.now()
    dt_stamp = args.run_stamp or f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    log_dir = make_log_dir(args.log_root, args.label, dt_stamp, args.seed, tag=_run_tag(args))
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mirror_stdout_to(log_dir / "terminal.txt")
    print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

    n_op = 5  # HiFo default operators: e1, e2, m1, m2, m3
    print(f"pop_size={args.pop_size}  max_generations={args.max_generations}  "
          f"n_instances={args.n_instances}  n_items={args.n_items}  capacity={args.capacity}")
    print(f"HiFo operators/gen = {n_op} (e1,e2,m1,m2,m3); nominal candidates ~ "
          f"{args.pop_size * n_op * args.max_generations}")

    # ---- LLM backend (identical to reprod/eoh_obp.py) -> CachedLLM -------- #
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
    cached = CachedLLM(client, cache_dir=cache_dir,
                       prompt_log=log_dir / "llm_prompts.jsonl")
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    # Bridge HiFo's LLM interface to our CachedLLM (monkeypatch the factory used
    # inside hifo_evolution.Evolution.__init__).
    bridge = _HiFoLLMBridge(cached, temperature=args.temperature, max_tokens=args.llm_max_tokens)
    _hifo_evolution.InterfaceLLM = lambda *a, **k: bridge

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    args_dict["llm_model"] = cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    # ---- OBP evaluator (identical instances to eoh_obp.py) --------------- #
    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluation = OBPEvaluation(
        timeout_seconds=args.timeout_seconds,
        n_instances=args.n_instances,
        n_items=args.n_items,
        capacity=args.capacity,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)

    mean_lb = _mean_lb(evaluation._datasets)
    print(f"  mean lower-bound bins = {mean_lb:.2f}  "
          f"(over {args.n_instances} instances of {args.n_items} items, capacity={args.capacity})")

    adapter = _OBPAdapter(evaluation)
    # Log EVERY evaluated candidate (all n_operators*pop_size offspring/gen, plus the
    # seed/init evals) to all_candidates.jsonl, so heuristics.json can list them all like
    # EoH does (HiFo's population dumps hold only the pop_size survivors).
    from utils.hifo_candidate_log import wrap_adapter_logging
    wrap_adapter_logging(adapter, log_dir)

    # ---- HiFo method internals (Layer B) -> hifo_hp ---------------------- #
    # Override the package's paper-default constants from the CLI. Done BEFORE
    # EVOL.run() builds the InsightPool / Navigator; the call sites read
    # hifo_hp.<NAME> at call time, so these take effect for the whole run.
    _hifo_hp.POOL_CAPACITY = args.pool_capacity
    _hifo_hp.NOVELTY_THRESHOLD = args.novelty_threshold
    _hifo_hp.SELECTION_COUNT = args.selection_count
    _hifo_hp.USAGE_PENALTY_WEIGHT = args.usage_penalty_weight
    _hifo_hp.RECENCY_BONUS = args.recency_bonus
    _hifo_hp.RECENCY_WINDOW = args.recency_window
    _hifo_hp.EMA_ALPHA = args.ema_alpha
    _hifo_hp.DECAY_RATE = args.decay_rate
    _hifo_hp.PROBATION_USAGE_COUNT = args.probation_usage
    _hifo_hp.CREDIT_BEST = args.credit_best
    _hifo_hp.CREDIT_INC = args.credit_inc
    _hifo_hp.CREDIT_PEN = args.credit_pen
    _hifo_hp.PROGRESS_EPS = args.progress_eps
    _hifo_hp.STAGNATION_THRESHOLD = args.stagnation_threshold
    _hifo_hp.PROGRESS_THRESHOLD = args.progress_threshold
    _hifo_hp.DIVERSITY_THRESHOLD = args.diversity_threshold
    print(f"HiFo internals: C_pool={_hifo_hp.POOL_CAPACITY} nov={_hifo_hp.NOVELTY_THRESHOLD} "
          f"s={_hifo_hp.SELECTION_COUNT} w_u={_hifo_hp.USAGE_PENALTY_WEIGHT} "
          f"tau_r={_hifo_hp.RECENCY_BONUS}/T_w={_hifo_hp.RECENCY_WINDOW} alpha={_hifo_hp.EMA_ALPHA} "
          f"R_decay={_hifo_hp.DECAY_RATE} T_usage={_hifo_hp.PROBATION_USAGE_COUNT} | "
          f"credit=[{_hifo_hp.CREDIT_BEST},{_hifo_hp.CREDIT_INC},{_hifo_hp.CREDIT_PEN}] | "
          f"nav: eps={_hifo_hp.PROGRESS_EPS} tau_stag={_hifo_hp.STAGNATION_THRESHOLD} "
          f"tau_prog={_hifo_hp.PROGRESS_THRESHOLD} delta_p={_hifo_hp.DIVERSITY_THRESHOLD}")

    # HiFo runs ONE offspring's LLM-generation AND its evaluation inside a single
    # joblib task. The stock package used a SINGLE knob (eva_timeout) for BOTH the
    # inner eval cap and the outer per-offspring joblib budget (eva_timeout+15) that
    # must also cover the LLM call -- so enlarging the budget to fit the LLM also
    # inflated the eval cap, letting a slow candidate eval blow the whole batch.
    # We DECOUPLE them via hifo_hp (read at call time inside hifo_interface_EC):
    #   EVAL_TIMEOUT = HiFo eval-thread net (an OUTER safety net; the REAL per-candidate
    #                  bound is SecureEvaluator's subprocess kill at OBPEvaluation.
    #                  timeout_seconds == --timeout-seconds). Set a bit ABOVE that kill
    #                  so HiFo's future never fires first / blocks shutdown on a live thread.
    #   TASK_BUDGET  = whole-offspring budget = LLM (<= --llm-timeout) + eval net + margin
    _hifo_hp.EVAL_TIMEOUT = args.timeout_seconds + 15
    _hifo_hp.TASK_BUDGET = args.llm_timeout + _hifo_hp.EVAL_TIMEOUT + 15
    print(f"HiFo timeouts: candidate subprocess kill = {args.timeout_seconds}s | HiFo eval "
          f"net = {_hifo_hp.EVAL_TIMEOUT}s | per-offspring budget = {_hifo_hp.TASK_BUDGET}s "
          f"(llm_timeout {args.llm_timeout} + eval net {_hifo_hp.EVAL_TIMEOUT} + 15)")

    # ---- HiFo parameters ------------------------------------------------- #
    paras = Paras()
    paras.set_paras(
        method="hifo",
        problem=adapter,                # non-string -> HiFo's Probs uses it directly
        ec_pop_size=args.pop_size,
        ec_n_pop=args.max_generations,
        ec_m=max(2, args.selection_num),
        exp_n_proc=args.num_threads,
        exp_output_path=str(log_dir),
        exp_debug_mode=False,
        eva_timeout=args.timeout_seconds,   # eval cap (fallback; hifo_hp.EVAL_TIMEOUT/TASK_BUDGET drive it)
        eva_numba_decorator=False,      # OBPEvaluation runs plain Python; no numba
        llm_use_local=False,
        llm_api_endpoint=args.llm_backend,   # unused (InterfaceLLM is bridged)
        llm_api_key="bridged",
        llm_model=cached.client.model,
    )

    print(f"[{args.label}] running HiFo (native engine, threading backend, "
          f"n_proc={args.num_threads})...")
    t0 = time.time()

    # HiFo dumps results/pops/population_generation_*.json AFTER every generation,
    # but the eoh-style trajectory.json / heuristics.json were only built once, at
    # the very end. A HiFo run is slow (~n_op x pop_size LLM calls per generation),
    # so a long or interrupted run left NO trajectory.json / heuristics.json. Fix:
    # a background watcher rebuilds them from the per-gen dumps every few seconds
    # (idempotent: _build_eoh_style_logs reads all dumps and rewrites atomically),
    # and a final guaranteed build runs in `finally` even if the engine raises.
    _stop = threading.Event()

    def _log_watcher():
        while not _stop.wait(_ARTIFACT_REFRESH_SECONDS):
            try:
                _build_eoh_style_logs(log_dir, args.label, mean_lb, args.n_instances,
                                      n_operators=n_op, pop_size=args.pop_size,
                                      wandb_logger=None, verbose=False)
            except Exception:
                pass  # transient (e.g. a dump being written) -> retry next tick

    watcher = threading.Thread(target=_log_watcher, name="hifo-log-watcher", daemon=True)
    watcher.start()

    # Force joblib's THREADING backend so the shared CachedLLM bridge is not
    # pickled to worker processes (HiFo parallelises offspring with joblib).
    import joblib
    try:
        with joblib.parallel_backend("threading", n_jobs=args.num_threads):
            if args.fix_init_pop:
                # Inject 2*pop_size fixed heuristics as HiFo's gen-0 batch (via the
                # native seed evaluator, so they're re-scored on THIS run's instances),
                # then let HiFo's normal population_management prune to pop_size.
                from utils.fixed_init_pop import load_fixed_initial_population
                from hifo.methods.hifo.hifo_interface_EC import InterfaceEC as _IEC
                import re as _re
                _fp = ROOT / "src" / "init_pop" / "hifo_obp.json"
                _srcs = [h["source"] for h in
                         load_fixed_initial_population(_fp, 2 * args.pop_size)]
                def _fixed_population_generation(self, __srcs=_srcs):
                    import math as _math
                    from joblib import Parallel as _Par, delayed as _del
                    seeds = []
                    for _c in __srcs:
                        _m = _re.search(r"def\s+(\w+)", _c) or _re.search(r"class\s+(\w+)", _c)
                        seeds.append({"algorithm": (_m.group(1) if _m else "fixed"), "code": _c})
                    # Robust seed init: HiFo's population_generation_seed does
                    # np.round(np.array(fitness),5) and exit()s if ANY seed evaluates to
                    # None (a fixed heuristic that crashes/times-out on some instance).
                    # Evaluate here and DROP the None/non-finite seeds so one bad heuristic
                    # degrades gracefully instead of aborting the run; population_management
                    # then prunes the finite survivors to pop_size. (Analog of the EoH
                    # fix-init -1e6 coercion — the failure just must not kill the run.)
                    _fit = _Par(n_jobs=self.n_p)(
                        _del(self.interface_eval.evaluate)(_s["code"]) for _s in seeds)
                    _pop = []
                    for _s, _f in zip(seeds, _fit):
                        if _f is None or not isinstance(_f, (int, float)) or not _math.isfinite(_f):
                            continue
                        _pop.append({"algorithm": _s["algorithm"], "code": _s["code"],
                                     "objective": round(float(_f), 5), "other_inf": None})
                    print(f"  [fixed-init-pop] {len(_pop)}/{len(seeds)} seeds evaluated "
                          f"finite (dropped {len(seeds) - len(_pop)} that failed); no HiFo "
                          f"exit() on failure", flush=True)
                    if not _pop:
                        raise RuntimeError(
                            "fixed-init-pop: ALL seed heuristics failed evaluation — cannot "
                            "seed HiFo gen-0. Check src/init_pop/*.json (heuristics must "
                            "evaluate finite on this run's instances).")
                    return _pop
                _IEC.population_generation = _fixed_population_generation
                print(f"  [fixed-init-pop] injecting {len(_srcs)} heuristics (2*pop_size) "
                      f"into HiFo gen-0 then normal management, from {_fp}", flush=True)
            EVOL(paras).run()
    finally:
        _stop.set()
        watcher.join(timeout=5)
        # Guaranteed final rebuild from whatever generations completed (so even an
        # interrupted / crashed run leaves usable trajectory.json + heuristics.json).
        _build_eoh_style_logs(log_dir, args.label, mean_lb, args.n_instances,
                              n_operators=n_op, pop_size=args.pop_size,
                              wandb_logger=wandb_logger)
    dt = time.time() - t0
    print(f"[{args.label}] HiFo finished in {dt:.1f}s, {adapter.n_evals} evaluations")
    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
