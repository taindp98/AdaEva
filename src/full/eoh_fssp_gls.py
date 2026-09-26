"""Reproduce EoH on the LLM4AD FSSP-GLS task (GLS-based, EoH paper-faithful).

Uses `FSSP_GLS_Evaluation` with per-instance GLS caps overridden to match
TSP-GLS: 60 s wall-clock and 1000 iterations (the FSSP evaluation module
defaults to time_max=10 s; we patch it to 60 s here). Two independent time
limits apply:
  - 60s/instance inside `gls()` (gls.py:215): breaks the GLS iteration loop
    after 60s and returns the best makespan found so far. This is the primary
    per-instance cap for well-behaved heuristics.
  - 65s × n_instances subprocess wall-clock (timeout_seconds): kills the entire
    per-candidate subprocess if it hangs — e.g. a generated `get_matrix_and_jobs`
    that infinite-loops before the 60s check can fire. The subprocess dying means
    EoH receives score=None for that candidate (treated as a failed evaluation).
The 5s/instance slack between the two ensures a well-behaved candidate always
finishes within the outer timeout.

Paper setting (reproduced here as the defaults):
  - pop_size           = 10
  - max_generations    = 20
  - selection_num (p)  = 5      (parents in E1/E2 prompts)
  - n_instances        = 64     FSSP instances
  - n_jobs             = 50     jobs per instance
  - n_machines         = 2-20   machines per instance (varies)

Fitness: evaluate_without_time returns `-mean(makespan)`. The run log
(`trajectory.json`) reports, per generation, the best-so-far incumbent:
  - score = mean(makespan)  (positive; lower = better)
No gap is reported (FSSP has no known optimal solution).
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

from utils import (CachedLLM, make_wandb_logger, OpenRouterClient, OllamaClient,
                   MistralClient, vLLMClient)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD
from utils.logger import make_log_dir, mirror_stdout_to
from utils.fixed_init_pop import load_fixed_initial_population, install_fixed_initial_population

from llm4ad.method.eoh import EoH
from llm4ad.method.eoh.profiler import EoHProfiler
from llm4ad.task.optimization.fssp_gls.evaluation import FSSP_GLS_Evaluation
from llm4ad.task.optimization.fssp_gls.get_instance import GetData
import llm4ad.task.optimization.fssp_gls.evaluation as _fssp_eval_mod


_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 10,
    "max_generations": 20,
    "n_instances": 64,
    "n_jobs": 50,
    "selection_num": 5,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "pop_size": "ps", "max_generations": "mg", "n_instances": "ni",
    "n_jobs": "nj", "selection_num": "sn",
    "num_threads": "nt", "num_cores": "nc", "llm_model": "model",
    "llm_backend": "llm",
}


def _run_tag(args: argparse.Namespace) -> str:
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
    path = log_dir / "trajectory.json"
    log = {"label": label, "trajectory": trajectory}
    json.dump(log, open(path, "w"), indent=2)
    print(f"  logged -> {path}  ({len(trajectory)} generations)")


# ----------------------------------------------------------------------------- #
# Recording profiler (FSSP-GLS variant; mirrors reprod/eoh_tsp_gls.py structure).
# Score from EoH is negative makespan (-mean_makespan); display as positive.
# ----------------------------------------------------------------------------- #
class RecordingProfiler(EoHProfiler):
    def __init__(self, label: str, log_dir: pathlib.Path, pop_size: int,
                 n_instance: int, wandb_logger=None):
        super().__init__(log_dir=None, create_random_path=False)
        self.label = label
        self.log_dir = log_dir
        self.pop_size = pop_size
        self.n_instance = n_instance
        self.records: List[dict] = []
        self._heuristics: List[dict] = []
        self.times: List[float] = []
        self.populations: Dict[int, list] = {}
        self._t0 = time.time()
        self._last_pop_gen = 0
        self._gen_lock = Lock()
        self.wandb_logger = wandb_logger
        self._wandb_logged_rows = 0

    def record_parameters(self, *args, **kwargs): pass

    def register_function(self, func, program=None, *args, **kwargs):
        s = func.score if (func is not None and func.score is not None) else float("-inf")
        try:
            src = str(func) if func is not None else ""
        except Exception:
            src = ""
        # Per-candidate evaluation wall time (LLM4AD EoH sets func.evaluate_time =
        # wall of the whole subprocess eval over all n_instances).
        et = getattr(func, "evaluate_time", None)
        eval_time = float(et) if et is not None else 0.0
        with self._gen_lock:
            self.records.append({"gen_id": None, "score": float(s), "eval_time": eval_time})
            self._heuristics.append({"gen_id": None, "score": float(s), "source": src})
        elapsed = time.time() - self._t0
        self.times.append(elapsed)
        dt = elapsed - (self.times[-2] if len(self.times) > 1 else 0.0)
        if math.isfinite(s):
            mean_makespan = -float(s)
            tag = f"score={s:.4f} mean_makespan={mean_makespan:.4f}"
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
        survivors = []
        for f in pop.population:
            survivors.append({"score": (float(f.score) if f.score is not None
                                        else float("-inf"))})
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
            chunk = evo_records[k:k + self.pop_size]
            if len(chunk) < self.pop_size:
                break
            gid = 1 + k // self.pop_size
            groups[gid] = chunk
            heur_groups[gid] = evo_h[k:k + self.pop_size]

        trajectory = []
        best_mean_makespan = float("inf")
        best_cand = None
        cum_budget = 0
        cum_cpu = 0.0   # cumulative CPU seconds (summed across cores) over ALL candidates
        for gid in sorted(groups.keys()):
            chunk = groups[gid]
            cum_budget += len(chunk) * self.n_instance
            cum_cpu += sum((r.get("eval_time") or 0.0) for r in chunk)
            for i, r in enumerate(chunk):
                s = r["score"]
                if math.isfinite(s):
                    mean_makespan = -float(s)
                    if mean_makespan < best_mean_makespan:
                        best_mean_makespan = mean_makespan
                        best_cand = f"gen{gid:02d}_cand{i:02d}"
            score = best_mean_makespan if best_cand is not None else float("inf")
            trajectory.append({
                "gen_id": gid,
                "cand_id": best_cand,
                "score": float(score),
                "n_instances": self.n_instance,
                "used_budget": cum_budget,
                # Cumulative CPU seconds (summed across cores) burned by ALL
                # candidate evaluations up to this generation.
                "cpu_seconds": round(cum_cpu, 3),
            })
        save_run_log(self.log_dir, self.label, trajectory)
        for row in trajectory[self._wandb_logged_rows:]:
            self.wandb_logger.log_trajectory_row(row)
        self._wandb_logged_rows = len(trajectory)
        self._save_evaluations_per_gen(groups)
        self._save_heuristics(heur_groups)

    def _save_evaluations_per_gen(self, groups: Dict[int, list]) -> None:
        entries = []
        for gid in sorted(groups.keys()):
            n_heur = len(groups[gid])
            entries.append({
                "gen_id": gid,
                "num_heuristics": n_heur,
                "n_instance": self.n_instance,
                "total_instance_evals": n_heur * self.n_instance,
            })
        payload = {
            "label": self.label,
            "n_instance": self.n_instance,
            "cumulative_instance_evals": sum(e["total_instance_evals"] for e in entries),
            "generations": entries,
        }
        json.dump(payload, open(self.log_dir / "evaluations_per_gen.json", "w"), indent=2)

    def _save_heuristics(self, heur_groups: Dict[int, list]) -> None:
        out: List[dict] = []
        for gid in sorted(heur_groups.keys()):
            for i, h in enumerate(heur_groups[gid]):
                out.append({"cand_id": f"gen{gid:02d}_cand{i:02d}",
                            "gen_id": gid,
                            "score": float(h["score"]),
                            "source": h["source"]})
        payload = {"label": self.label, "total_sampled": len(out), "heuristics": out}
        json.dump(payload, open(self.log_dir / "heuristics.json", "w"), indent=2)

    def finish(self, *args, **kwargs): pass


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce EoH on FSSP-GLS (paper-faithful).")
    p.add_argument("--pop-size", type=int, default=10,
                   help="Population size.")
    p.add_argument("--max-generations", type=int, default=20,
                   help="Number of evolution generations.")
    p.add_argument("--n-instances", type=int, default=64,
                   help="Number of FSSP instances (paper: 64).")
    p.add_argument("--n-jobs", type=int, default=50,
                   help="Jobs per instance (paper: 50).")
    p.add_argument("--m-low", type=int, default=2,
                   help="Minimum machines per instance (paper: 2).")
    p.add_argument("--m-high", type=int, default=20,
                   help="Maximum machines per instance (paper: 20).")
    p.add_argument("--instance-source", type=str, default="pregenerated",
                   choices=["pregenerated", "synthetic"],
                   help="FSSP instance source. 'pregenerated' (default) loads the "
                        "fixed pre-generated benchmark instances from disk (integer "
                        "processing times); 'synthetic' generates U[0,1] instances "
                        "with --n-jobs jobs and machines drawn uniformly in "
                        "[--m-low, --m-high].")
    p.add_argument("--selection-num", type=int, default=5,
                   help="Parents p for E1/E2 prompts (paper: 5).")
    p.add_argument("--timeout-seconds", type=int, default=None,
                   help="Per-candidate subprocess timeout covering all n_instances sequentially. "
                        "Default: 65 × n_instances. Must exceed 60 × n_instances so well-behaved "
                        "candidates finish; kills hangs that would block the 60s/instance GLS check.")
    p.add_argument("--label", type=str, default="eoh/fssp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048, help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm"],
                   help="LLM backend. 'openrouter' (default), 'ollama' for "
                        "self-hosted, or 'mistral' for Mistral's API (needs MISTRAL_API_KEY).")
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next",
                   help="Model id for the selected backend.")
    p.add_argument("--ollama-host", type=str, default=None,
                   help="Ollama server host:port (default: localhost:11434). "
                        "Only used when --llm-backend=ollama.")
    p.add_argument("--num-threads", type=int, default=4,
                   help="Parallel samplers for LLM API.")
    p.add_argument("--num-cores", type=int, default=4,
                   help="Parallel evaluators (process pool).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fix-init-pop", action="store_true",
                   help="Evaluate src/init_pop/eoh_fssp_gls.json as generation 0 instead of sampling it from the LLM.")
    p.add_argument("--use-wandb", action="store_true", default=False,
                   help="Log trajectory to Weights & Biases. Requires WANDB_API_KEY.")
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
    log_dir = make_log_dir(args.log_root, args.label, dt_stamp, args.seed,
                           tag=_run_tag(args))
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mirror_stdout_to(log_dir / "terminal.txt")
    print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

    total_samples = args.pop_size * args.max_generations
    budget = total_samples * args.n_instances
    print(f"pop_size={args.pop_size}  max_generations={args.max_generations}  "
          f"n_instances={args.n_instances}  n_jobs={args.n_jobs}  "
          f"m_range=[{args.m_low}, {args.m_high}]")
    print(f"total candidates = {total_samples}, function evaluations = {budget}")

    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model,
                              timeout=args.llm_timeout)
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
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label,
                                  model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir,
                           prompt_log=log_dir / "llm_prompts.jsonl")
        llm = OpenRouterLLM4AD(cached)
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
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

    # GetData seeds np.random internally (seed=2024); re-seed after so EoH
    # randomness is controlled by args.seed, not the instance-generation state.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Override per-instance GLS time cap to 60s (matching TSP-GLS's
    # solve_without_time). The FSSP evaluation module defaults to time_max=10s;
    # iter_max=1000 is already correct.
    _fssp_eval_mod.time_max = 60.0
    _fssp_eval_mod.iter_max = 1000

    evaluation = FSSP_GLS_Evaluation()
    evaluation.n_instance = args.n_instances
    evaluation.problem_size = args.n_jobs
    # timeout_seconds is the wall-clock kill for the WHOLE per-candidate
    # subprocess (all n_instances sequentially). It must be large enough for
    # a well-behaved heuristic (≥ 60s × n_instances) but finite so a heuristic
    # that infinite-loops inside get_matrix_and_jobs (before the 60s check at
    # gls.py:215 can fire) doesn't hang the run forever.
    # Default: 65s × n_instances — 5s slack per instance over the GLS cap.
    timeout_seconds = args.timeout_seconds or (65 * args.n_instances)
    evaluation.timeout_seconds = timeout_seconds
    # Instance source is selected by --instance-source (default 'pregenerated' =
    # load the fixed benchmark files; 'synthetic' = generate U[0,1] with n_jobs
    # jobs and machines in [m_low, m_high]). n_jobs/m_low/m_high are ignored by
    # GetData in pregenerated mode (the files carry their own dimensions).
    evaluation._datasets = GetData(
        args.n_instances, n_jobs=args.n_jobs, m_low=args.m_low, m_high=args.m_high,
        use_pregenerated=(args.instance_source == "pregenerated"),
    ).generate_instances()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"  instances: n={args.n_instances}  n_jobs={args.n_jobs}  "
          f"machines=[{args.m_low},{args.m_high}]  "
          f"timeout={timeout_seconds}s/candidate (65s × {args.n_instances} instances)")

    prof = RecordingProfiler(label=args.label, log_dir=log_dir,
                             pop_size=args.pop_size, n_instance=args.n_instances,
                             wandb_logger=wandb_logger)
    method = EoH(
        llm=llm,
        profiler=prof,
        evaluation=evaluation,
        # +1 compensates for EoH's phantom survival generation.
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
        install_fixed_initial_population(method, ROOT / "src" / "init_pop" / "eoh_fssp_gls.json")

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
