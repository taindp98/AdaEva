"""Reproduce EoH on the LLM4AD Online Bin Packing task.

Uses `llm4ad.task.optimization.online_bin_packing.OBPEvaluation` (Weibull
instances, fixed bin capacity, item-by-item online packing with an
LLM-generated priority function).

Paper setting (reproduced here as the defaults):
  - pop_size           = 20
  - max_generations    = 20
  - selection_num (p)  = 5     (parents in E1/E2 prompts)
  - n_instances        = 5     Weibull instances
  - n_items            = 5000  items per instance
  - capacity           = 100

Fitness: OBPEvaluation.evaluate returns `-mean(num_bins)`. The run log
(`trajectory.json`) reports, per generation, the best-so-far incumbent:
  - score = mean(num_bins)  (positive; lower = better)
  - gap   = (mean(num_bins) - mean(lb)) / mean(lb)
            where lb = ceil(sum(items)/capacity) per instance.
This matches the paper's "fraction of excess bins to the lower bound", and
mirrors `racing_eoh_obp.py`'s `trajectory.json` so the two runs compare 1:1.
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
import time
import yaml
from threading import Lock
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))

import numpy as np
from dotenv import load_dotenv

from utils import CachedLLM, make_wandb_logger, OpenRouterClient, OllamaClient, MistralClient, vLLMClient
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD
from utils.obp_utils import _obp_lower_bound
from utils.logger import make_log_dir, mirror_stdout_to
from utils.fixed_init_pop import load_fixed_initial_population, install_fixed_initial_population

from llm4ad.method.eoh import EoH
from llm4ad.method.eoh.profiler import EoHProfiler
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation


# Bash script defaults (run_reprod_eoh_obp.sh). Used to detect overrides and
# build a descriptive folder-name tag so each run is self-documenting.
_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 20,
    "max_generations": 20,
    "n_instances": 25,
    "n_items": 5000,
    "capacity": 100,
    "selection_num": 5,
    "timeout_seconds": 30,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
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
    "num_threads": "nt",
    "num_cores": "nc",
    "llm_model": "model",
    "llm_backend": "llm",
}


def _run_tag(args: argparse.Namespace) -> str:
    """Return a compact folder-name tag encoding args that differ from the
    bash-script defaults. Returns 'default' when nothing was overridden."""
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


def save_run_log(log_dir: pathlib.Path, label: str, trajectory: list) -> None:
    """Write the run log: the per-generation best-so-far trajectory.

    Structured to line up 1:1 with `racing_eoh_obp.py`'s `trajectory.json` so
    the two runs can be plotted/compared directly. `trajectory` is a flat list
    with one entry per generation; each entry carries:

        - ``gen_id``      -- generation index (0 = initial population).
        - ``cand_id``     -- the best-so-far incumbent up to this generation.
        - ``score``       -- the incumbent's mean bin count (positive; lower
                             is better), matching the racing log's ``score``.
        - ``gap``         -- ``(score - mean_lb) / mean_lb``.
        - ``n_instances`` -- instances each candidate is evaluated on (EoH
                             always uses the full set, so this is constant).
        - ``used_budget`` -- cumulative (candidate, instance) evaluations
                             spent up to and including this generation.
    """
    path = log_dir / "trajectory.json"
    log = {"label": label, "trajectory": trajectory}
    json.dump(log, open(path, "w"), indent=2)
    print(f"  logged -> {path}  ({len(trajectory)} generations)")


def _mean_lb(datasets: dict) -> float:
    lbs = []
    for name in datasets:
        inst = datasets[name]
        lbs.append(_obp_lower_bound(inst["items"], inst["capacity"]))
    return float(np.mean(lbs))


# ----------------------------------------------------------------------------- #
# Recording profiler (OBP variant of reprod_eoh_tsp's RecordingProfiler).
# Mirrors the same record-order gen chunking so logs line up across tasks.
# ----------------------------------------------------------------------------- #
class RecordingProfiler(EoHProfiler):
    def __init__(
        self,
        label: str,
        log_dir: pathlib.Path,
        pop_size: int,
        n_instance: int,
        mean_lb: float,
        wandb_logger=None,
    ):
        super().__init__(log_dir=None, create_random_path=False)
        self.label = label
        self.log_dir = log_dir
        self.pop_size = pop_size
        self.n_instance = n_instance
        self.mean_lb = mean_lb
        self.records: List[dict] = []
        self._heuristics: List[dict] = []
        self.times: List[float] = []
        self.populations: Dict[int, list] = {}
        self._t0 = time.time()
        self._last_pop_gen = 0
        self._gen_lock = Lock()
        self.wandb_logger = wandb_logger
        self._wandb_logged_rows = 0

    def record_parameters(self, *args, **kwargs):
        pass

    def register_function(self, func, program=None, *args, **kwargs):
        s = (
            func.score
            if (func is not None and func.score is not None)
            else float("-inf")
        )
        try:
            src = str(func) if func is not None else ""
        except Exception:
            src = ""
        with self._gen_lock:
            self.records.append({"gen_id": None, "score": float(s)})
            self._heuristics.append({"gen_id": None, "score": float(s), "source": src})
        elapsed = time.time() - self._t0
        self.times.append(elapsed)
        dt = elapsed - (self.times[-2] if len(self.times) > 1 else 0.0)
        if math.isfinite(s):
            mean_bins = -float(s)
            gap = (
                (mean_bins - self.mean_lb) / self.mean_lb
                if self.mean_lb > 0
                else float("inf")
            )
            tag = f"score={s:.4f} mean_bins={mean_bins:.2f} gap={gap:.4f}"
        else:
            tag = "FAILED (score=-inf)"
        print(f"  [pending] {tag}  +{dt:.1f}s  (elapsed {elapsed:.0f}s)", flush=True)

    def register_population(self, pop):
        try:
            cur = int(pop.generation)
        except Exception:
            return
        with self._gen_lock:
            advanced = cur > self._last_pop_gen
            for r in self.records:
                if r["gen_id"] is None:
                    r["gen_id"] = cur
            for h in self._heuristics:
                if h["gen_id"] is None:
                    h["gen_id"] = cur
            self._last_pop_gen = cur
            need_snapshot = advanced and cur not in self.populations
        if need_snapshot:
            self._snapshot_population(pop, cur)
            self._checkpoint()

    def _snapshot_population(self, pop, gen_id: int):
        # Capture each survivor's IDENTITY (source) + score, not just the score, so the
        # environmental-selection survivor set entering a generation can be matched back to
        # heuristics.json cand_ids (ranking analysis: parent+offspring selection pool).
        survivors = []
        for f in pop.population:
            try:
                src = str(f)
            except Exception:
                src = ""
            survivors.append({
                "source": src,
                "score": (float(f.score) if f.score is not None else float("-inf")),
            })
        with self._gen_lock:
            self.populations[gen_id] = survivors

    def finalize_population(self, pop):
        try:
            cur = int(pop.generation)
        except Exception:
            return
        with self._gen_lock:
            if cur in self.populations:
                return
        self._snapshot_population(pop, cur)
        self._checkpoint()

    def _checkpoint(self):
        with self._gen_lock:
            snapshot_records = list(self.records)
            snapshot_heuristics = list(self._heuristics)
        finalized: list = []
        finalized_h: list = []
        for i, r in enumerate(snapshot_records):
            if r.get("gen_id") is not None:
                finalized.append(r)
                finalized_h.append(snapshot_heuristics[i])
        init_records: list = []
        init_h: list = []
        valid_count = 0
        evo_start = 0
        for i, r in enumerate(finalized):
            init_records.append(r)
            init_h.append(finalized_h[i])
            if math.isfinite(r["score"]):
                valid_count += 1
            if valid_count >= self.pop_size:
                evo_start = i + 1
                break
        if valid_count < self.pop_size:
            evo_start = len(finalized)
        evo_records = finalized[evo_start:]
        evo_h = finalized_h[evo_start:]
        groups: Dict[int, list] = {0: init_records}
        heur_groups: Dict[int, list] = {0: init_h}
        for k in range(0, len(evo_records), self.pop_size):
            chunk = evo_records[k : k + self.pop_size]
            if len(chunk) < self.pop_size:
                break
            gid = 1 + k // self.pop_size
            groups[gid] = chunk
            heur_groups[gid] = evo_h[k : k + self.pop_size]

        # Per-generation best-so-far trajectory. Mirrors the racing log:
        # `score` = incumbent mean bins (positive, lower better), `used_budget`
        # = cumulative (candidate, instance) evaluations spent so far.
        trajectory = []
        best_mean_bins = float("inf")
        best_cand = None
        cum_budget = 0
        for gid in sorted(groups.keys()):
            chunk = groups[gid]
            cum_budget += len(chunk) * self.n_instance
            for i, r in enumerate(chunk):
                s = r["score"]
                if math.isfinite(s):
                    mean_bins = -float(s)
                    if mean_bins < best_mean_bins:
                        best_mean_bins = mean_bins
                        best_cand = f"gen{gid:02d}_cand{i:02d}"
            if best_cand is None:
                score = gap = float("inf")
            else:
                score = best_mean_bins
                gap = (
                    (best_mean_bins - self.mean_lb) / self.mean_lb
                    if self.mean_lb > 0
                    else float("inf")
                )
            trajectory.append(
                {
                    "gen_id": gid,
                    "cand_id": best_cand,
                    "score": float(score),
                    "gap": float(gap),
                    "n_instances": self.n_instance,
                    "used_budget": cum_budget,
                }
            )
        save_run_log(self.log_dir, self.label, trajectory)
        for row in trajectory[self._wandb_logged_rows :]:
            self.wandb_logger.log_trajectory_row(row)
        self._wandb_logged_rows = len(trajectory)
        self._save_evaluations_per_gen(groups)
        self._save_heuristics(heur_groups)
        self._save_survivors(heur_groups)

    def _save_survivors(self, heur_groups: Dict[int, list]) -> None:
        """Dump the environmental-selection SURVIVOR set per generation to survivors.json.

        ``self.populations[g]`` holds the survivors kept AFTER generation g's survival()
        (each: source + score). We map every survivor's ``source`` back to its cand_id, so
        downstream analysis can form the true parent+offspring selection pool
        (survivors(g-1) + offspring(g)) by cand_id.

        The source->cand_id map MUST match heuristics.json's ``gen{gid}_cand{i}`` scheme, so we
        reproduce ``_checkpoint``'s chunking over the full per-sample record
        (``self._heuristics``): gen0 = the first pop_size VALID (finite-score) samples, then
        consecutive pop_size chunks. Unlike heuristics.json — which DROPS the trailing
        <pop_size chunk — we also index that partial trailing chunk so a freshly-sampled
        surviving elite still resolves instead of logging cand_id=None. A survivor whose source
        is genuinely unlogged stays cand_id=None.

        Inherited by HeteroRecordingProfiler (its overridden _checkpoint calls this too).
        (Survivor keys are EoH's pop.generation, possibly offset from chunk gid; match by SOURCE.)"""
        with self._gen_lock:
            full_h = list(self._heuristics)
            pops = {g: list(s) for g, s in self.populations.items()}
        # Key by a DOCSTRING-INSENSITIVE canonical source: survivors from pop.population have
        # their docstring stripped in str(f) while register_function logged str(func) WITH the
        # docstring, so a raw-string match misses exactly those candidates (cand_id=None).
        from utils.source_key import canon_source
        finalized = [h for h in full_h if h.get("gen_id") is not None]
        src_to_cid: Dict[str, str] = {}

        def _assign(gid: int, entries: list) -> None:
            for i, h in enumerate(entries):
                src_to_cid.setdefault(canon_source(h.get("source", "")), f"gen{gid:02d}_cand{i:02d}")

        valid_count = 0
        evo_start = len(finalized)
        for idx, h in enumerate(finalized):
            if math.isfinite(h.get("score", float("-inf"))):
                valid_count += 1
            if valid_count >= self.pop_size:
                evo_start = idx + 1
                break
        _assign(0, finalized[:evo_start])                     # gen0 = initial population
        evo_h = finalized[evo_start:]
        for k in range(0, len(evo_h), self.pop_size):          # gen1.. = pop_size chunks
            _assign(1 + k // self.pop_size, evo_h[k:k + self.pop_size])

        out: Dict[str, list] = {}
        for gid in sorted(pops.keys()):
            rows = []
            for s in pops[gid]:
                cid = src_to_cid.get(canon_source(s.get("source", "")))
                rows.append({"cand_id": cid, "score": float(s["score"])})
            out[str(gid)] = rows
        payload = {"label": self.label, "pop_size": self.pop_size, "survivors_by_gen": out}
        json.dump(payload, open(self.log_dir / "survivors.json", "w"), indent=2)

    def _save_evaluations_per_gen(self, groups: Dict[int, list]) -> None:
        entries = []
        for gid in sorted(groups.keys()):
            n_heur = len(groups[gid])
            entries.append(
                {
                    "gen_id": gid,
                    "num_heuristics": n_heur,
                    "n_instance": self.n_instance,
                    "total_instance_evals": n_heur * self.n_instance,
                }
            )
        payload = {
            "label": self.label,
            "n_instance": self.n_instance,
            "cumulative_instance_evals": sum(
                e["total_instance_evals"] for e in entries
            ),
            "generations": entries,
        }
        json.dump(
            payload, open(self.log_dir / "evaluations_per_gen.json", "w"), indent=2
        )

    def _save_heuristics(self, heur_groups: Dict[int, list]) -> None:
        out: List[dict] = []
        for gid in sorted(heur_groups.keys()):
            for i, h in enumerate(heur_groups[gid]):
                out.append(
                    {
                        "cand_id": f"gen{gid:02d}_cand{i:02d}",
                        "gen_id": gid,
                        "score": float(h["score"]),
                        "source": h["source"],
                    }
                )
        payload = {"label": self.label, "total_sampled": len(out), "heuristics": out}
        json.dump(payload, open(self.log_dir / "heuristics.json", "w"), indent=2)

    def finish(self, *args, **kwargs):
        pass


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce EoH on Online Bin Packing.")
    p.add_argument(
        "--pop-size", type=int, default=20, help="Population size (paper: 20)."
    )
    p.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Number of generations (paper: 20).",
    )
    p.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of Weibull instances (paper: 5).",
    )
    p.add_argument(
        "--n-items", type=int, default=5000, help="Items per instance (paper: 5000)."
    )
    p.add_argument(
        "--capacity", type=int, default=100, help="Bin capacity (paper: 100)."
    )
    p.add_argument(
        "--selection-num",
        type=int,
        default=5,
        help="Parents p for E1/E2 prompts (paper: 5).",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-candidate eval timeout passed to OBPEvaluation.",
    )
    p.add_argument("--label", type=str, default="eoh/obp")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument(
        "--llm-backend",
        type=str,
        default="openrouter",
        choices=["openrouter", "ollama", "mistral", "vllm"],
        help="LLM backend to use. 'openrouter' (default) queries the "
        "OpenRouter API; 'ollama' queries a local Ollama server; 'mistral' queries Mistral API.",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="qwen/qwen3-coder-next",
        help="Model id passed to the selected backend. For OpenRouter: "
        "'qwen/qwen3-coder-next', etc. For Ollama: local model name "
        "e.g. 'codellama'. Overrides OPENROUTER_MODEL / OLLAMA_MODEL "
        "env vars.",
    )
    p.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server host:port (default: localhost:11434). "
        "Only used when --llm-backend=ollama. "
        "Overrides OLLAMA_HOST env var.",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Parallel samplers for LLM API.",
    )
    p.add_argument(
        "--num-cores",
        type=int,
        default=2,
        help="Parallel evaluators (process pool).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fix-init-pop", action="store_true",
                   help="Evaluate src/init_pop/eoh_obp.json as generation 0 instead of sampling it from the LLM.")
    p.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Log trajectory to Weights & Biases in real-time. "
        "Requires WANDB_API_KEY to be set in the environment / .env.",
    )
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
    log_dir = make_log_dir(
        args.log_root, args.label, dt_stamp, args.seed, tag=_run_tag(args)
    )
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mirror_stdout_to(log_dir / "terminal.txt")
    print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

    total_samples = args.pop_size * args.max_generations
    budget = total_samples * args.n_instances
    print(
        f"pop_size={args.pop_size}  max_generations={args.max_generations}  "
        f"n_instances={args.n_instances}  n_items={args.n_items}  "
        f"capacity={args.capacity}"
    )
    print(f"total candidates = {total_samples}, function evaluations = {budget}")

    if args.llm_backend == "ollama":
        client = OllamaClient(
            host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout
        )
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = OllamaLLM4AD(cached)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = MistralLLM4AD(cached)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = vLLMLLM4AD(cached)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(
            timeout=args.llm_timeout, x_title=args.label, model=args.llm_model
        )
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = OpenRouterLLM4AD(cached)
    print(
        f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}"
    )

    args_dict = {
        k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()
    }
    args_dict["llm_model"] = cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    wandb_logger = make_wandb_logger(
        enabled=args.use_wandb,
        project="llm4ad",
        name=run_name,
        config=args_dict,
    )

    # OBP instance pool. Note: generate_weibull_dataset uses Python's random
    # module; re-seed right before construction so the instance set is
    # reproducible given args.seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluation = OBPEvaluation(
        timeout_seconds=args.timeout_seconds,
        n_instances=args.n_instances,
        n_items=args.n_items,
        capacity=args.capacity,
    )
    # Re-seed after dataset construction so EoH-side randomness is controlled
    # by args.seed, not the instance-generation state.
    random.seed(args.seed)
    np.random.seed(args.seed)

    mean_lb = _mean_lb(evaluation._datasets)
    print(
        f"  mean lower-bound bins = {mean_lb:.2f}  "
        f"(over {args.n_instances} instances of {args.n_items} items, "
        f"capacity={args.capacity})"
    )

    prof = RecordingProfiler(
        label=args.label,
        log_dir=log_dir,
        pop_size=args.pop_size,
        n_instance=args.n_instances,
        mean_lb=mean_lb,
        wandb_logger=wandb_logger,
    )
    method = EoH(
        llm=llm,
        profiler=prof,
        evaluation=evaluation,
        # +1 compensates for EoH's phantom survival generation (see TSP variant).
        max_generations=args.max_generations + 1,
        max_sample_nums=10**9,
        pop_size=args.pop_size,
        selection_num=args.selection_num,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        num_samplers=args.num_threads,
        num_evaluators=args.num_cores,
        multi_thread_or_process_eval="process",
        debug_mode=False,
    )
    if args.fix_init_pop:
        install_fixed_initial_population(method, ROOT / "src" / "init_pop" / "eoh_obp.json")

    print(f"[{args.label}] running EoH (budget = {budget} evaluations)...")
    t0 = time.time()
    method.run()
    dt = time.time() - t0
    print(f"[{args.label}] done in {dt:.1f}s, recorded {len(prof.records)} candidates")
    try:
        prof.finalize_population(method._population)
    except Exception as e:
        print(f"  [finalize_population] skipped: {type(e).__name__}: {e}")
    prof._checkpoint()
    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
