"""Random Policy (Fixed-K) evaluation for LLaMEA on BBOB.

The "tiny" (Fixed-K) counterpart of ``reprod/llamea_bbob.py``: instead of scoring every
candidate on the FULL BBOB grid (72 = 24 fids x 3 iids, each x n_reps), each candidate is
evaluated on exactly ``K`` of the 72 ``(fid, iid)`` instances (still x n_reps each). This is
the LLaMEA-on-BBOB analog of ``tiny/eoh_tsp_gls.py``'s K-instance random policy, a fixed
allocation baseline. Fitness = mean AOCC over the K-subset (higher = better).

Design (mirrors ``racing/llamea_bbob.RacingLLaMEA(LLaMEA_BBOB)``): subclass the native ES
``reprod.llamea_bbob.LLaMEA_BBOB`` and override only ``_evaluate_population`` (the instance
pool) — LLaMEA's (mu+lambda)/(mu,lambda) selection, prompts, sampling and logging are reused.

``--instance-mode`` (like the EoH tiny runners):
  - ``fixed``:  one K-subset of the 72 (fid,iid) pairs, sampled once (seeded by ``--seed``)
    before evolution and reused for EVERY candidate.
  - ``random``: a fresh K-subset is drawn per candidate, seeded reproducibly by the
    candidate's source, so the same code always gets the same subset.

Hetero only (the 72-pair BBOB grid); ``--instance-pool-mode homo`` is not supported here.

After evolution, each run's incumbent trajectory is re-evaluated on the FULL 72-instance grid
via ``analyses.eval_bbob.evaluate_single`` (mode='valid') -> ``valid_trajectory.json``, matching
the ``tiny/eoh_*`` runners' ``_final_eval`` flow.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import math
import pathlib
import random
import sys

import numpy as np
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))

from llamea.utils import prepare_namespace, clean_local_namespace
from llamea.solution import Solution

# Reuse the native ES + all its machinery from reprod.
from reprod.llamea_bbob import (
    LLaMEA_BBOB,
    TrajectoryLogger,
    _eval_single_task,
    _FIDS,
    _IIDS,
    _N_UNIQUE_INSTANCES,
    _auto_eval_timeout,
    _build_llm,
    load_fixed_initial_population,
)

from utils.logger import make_log_dir, mirror_stdout_to
from utils import make_wandb_logger
from utils.source_key import canon_source


# The full hetero instance pool: the 72 unique (fid, iid) pairs (order fixed/stable).
_INSTANCE_POOL: list = [(fid, iid) for fid in _FIDS for iid in _IIDS]
assert len(_INSTANCE_POOL) == _N_UNIQUE_INSTANCES


# --------------------------------------------------------------------------- #
# K-instance subset selection (fixed | random) — same contract as tiny/eoh_*.
# --------------------------------------------------------------------------- #

def _sample_subset_idx(n_total: int, k: int, seed: int) -> list:
    """K distinct instance indices (sorted) sampled without replacement from
    range(n_total), reproducible given ``seed``."""
    rng = np.random.RandomState(int(seed) % (2 ** 16))
    return sorted(rng.choice(n_total, size=min(int(k), n_total), replace=False).tolist())


def _program_subset_seed(program_str: str, base_seed: int) -> int:
    """Stable per-candidate seed for 'random' mode. Uses hashlib (NOT Python's salted
    ``hash``) so the same source yields the same subset across processes/reruns."""
    h = int(hashlib.md5(program_str.encode("utf-8")).hexdigest(), 16)
    return (h ^ (int(base_seed) & 0xFFFFFFFF)) % (2 ** 16)


class TinyLLaMEA_BBOB(LLaMEA_BBOB):
    """``LLaMEA_BBOB`` restricted to a K-instance subset of the 72 (fid,iid) pairs.

    Overrides ONLY ``_evaluate_population`` (the instance pool) + adds survivors.json;
    everything else (selection, prompts, sampling, trajectory logging) is inherited.
    """

    def __init__(self, *, K: int, instance_mode: str = "fixed", **kwargs):
        if str(kwargs.get("instance_pool_mode", "hetero")).lower() != "hetero":
            raise ValueError("tiny/llamea_bbob supports only --instance-pool-mode hetero "
                             "(K is a subset of the 72 (fid,iid) pairs).")
        super().__init__(**kwargs)
        self.K = int(K)
        self.instance_mode = str(instance_mode).lower()
        # Fixed subset: sampled once (seeded by --seed), reused for every candidate.
        if self.instance_mode == "fixed":
            idx = _sample_subset_idx(len(_INSTANCE_POOL), self.K, self.seed)
            self._fixed_subset = [_INSTANCE_POOL[i] for i in idx]
            print(f"  [instance-mode=fixed] sampled K={len(idx)} of {len(_INSTANCE_POOL)} "
                  f"(fid,iid) instances once (seed={self.seed}): {self._fixed_subset}", flush=True)
        else:
            self._fixed_subset = None
            print(f"  [instance-mode=random] K={self.K} of {len(_INSTANCE_POOL)} (fid,iid) "
                  f"instances resampled per candidate (seeded by source ^ {self.seed})", flush=True)
        # The trajectory's used_budget counts evals over the K-subset, not the full grid.
        self._traj = TrajectoryLogger(
            log_dir=self.log_dir, label=self.label,
            evals_per_candidate=self.K * self.n_reps,
            wandb_logger=getattr(self._traj, "wandb_logger", None),
        )

    def _subset_for(self, code: str) -> list:
        """The K (fid,iid) pairs this candidate is scored on."""
        if self.instance_mode == "fixed":
            return self._fixed_subset
        seed = _program_subset_seed(code, self.seed)
        idx = _sample_subset_idx(len(_INSTANCE_POOL), self.K, seed)
        return [_INSTANCE_POOL[i] for i in idx]

    def _evaluate_population(self, population: list) -> None:
        """Assign ``.fitness`` = mean AOCC over this candidate's K-instance subset
        (x n_reps), instead of the full 72-grid. Reuses the parent's pool + per-task
        timeout machinery (``_map_with_timeout`` / ``_eval_single_task``)."""
        import collections
        tasks: list = []
        failed: set = set()
        per_cand_n: dict = {}   # idx -> number of tasks (K*n_reps; may vary only by failure)
        budget = self.budget_factor * self.dim
        for idx, sol in enumerate(population):
            issue = None
            try:
                global_ns, issue = prepare_namespace(sol.code, allowed=["numpy"], logger=None)
                ns: dict = {}
                exec(sol.code, global_ns, ns)
                ns = clean_local_namespace(ns, global_ns)
                subset = self._subset_for(sol.code)
                for (fid, iid) in subset:
                    for rep in range(self.n_reps):
                        tasks.append((idx, fid, iid, rep, self.dim, budget, sol.code, sol.name))
                per_cand_n[idx] = len(subset) * self.n_reps
            except Exception as e:
                sol.set_scores(self._worst, feedback=f" {issue}." if issue else "", error=e)
                failed.add(idx)

        if not tasks:
            return
        if self._eval_pool is not None:
            results = self._map_with_timeout(tasks)
        else:
            results = [_eval_single_task(t) for t in tasks]

        by_idx = collections.defaultdict(list)
        for r in results:
            if r is not None:
                val = r[1] if (r[1] is not None and math.isfinite(r[1])) else 0.0
                by_idx[r[0]].append(val)

        for idx, sol in enumerate(population):
            if idx in failed:
                continue
            aucs = by_idx[idx]
            expected = per_cand_n.get(idx, self.K * self.n_reps)
            if len(aucs) < expected:   # pad timed-out/unreturned tasks with worst AOCC
                aucs.extend([0.0] * (expected - len(aucs)))
            if not aucs or all(v == 0.0 for v in aucs):
                sol.set_scores(self._worst, "Algorithm failed to evaluate on BBOB instances.")
            else:
                m, s = float(np.mean(aucs)), float(np.std(aucs))
                sol.add_metadata("aucs", aucs)
                sol.set_scores(
                    m,
                    f"The algorithm {sol.name} got an average Area over the convergence "
                    f"curve (AOCC, 1.0 is the best) score of {m:0.4f} with standard "
                    f"deviation {s:0.4f}.",
                )

    # ---- survivors.json (environmental-selection survivor set per generation) ----

    def _record_generation(self, population: list) -> None:
        super()._record_generation(population)
        self._save_survivors()

    def _save_survivors(self) -> None:
        """Dump the current post-selection population (the (mu+lambda)->mu survivors
        that enter the next generation) to survivors.json, in the same schema as the
        EoH tiny runners, so eval_uniform_*_ranking can build the parent+offspring
        selection pool. cand_id is resolved by matching a survivor's code to its
        heuristics.json entry (keyed docstring-insensitively, for parity with EoH —
        LLaMEA logs raw ``sol.code`` so an exact match also holds)."""
        import json
        # source -> first cand_id (heuristics.json uses gen{gid}_cand{i} by record order).
        src_to_cid: dict = {}
        for h in self._traj.heuristics:
            src_to_cid.setdefault(canon_source(h.get("source", "")), h["cand_id"])
        rows = []
        for sol in self.population:
            fit = sol.fitness if math.isfinite(getattr(sol, "fitness", float("nan"))) else float("-inf")
            rows.append({"cand_id": src_to_cid.get(canon_source(sol.code)), "score": float(fit)})
        payload = {"label": self.label, "pop_size": self.n_parents,
                   "survivors_by_gen": {str(self.generation): rows}}
        # Accumulate across generations (re-read + merge so each gen adds one key).
        out_path = self.log_dir / "survivors.json"
        existing = {}
        if out_path.exists():
            try:
                existing = json.load(open(out_path)).get("survivors_by_gen", {})
            except Exception:
                existing = {}
        existing[str(self.generation)] = rows
        payload["survivors_by_gen"] = existing
        json.dump(payload, open(out_path, "w"), indent=2)


# --------------------------------------------------------------------------- #
# CLI + main (mirrors reprod/llamea_bbob, + --K / --instance-mode + _final_eval)
# --------------------------------------------------------------------------- #

_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "llm_model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "llm_backend": "vllm",
    "temperature": 0.8,
    "t_iter": 100,
    "max_generations": None,
    "n_parents": 10,
    "n_offspring": 10,
    "dim": 5,
    "budget_factor": 2000,
    "n_reps": 1,
    "n_runs": 5,
    "K": 1,
    "instance_mode": "fixed",
    "elitism": True,
    "evolution_mode": "population",
    "parent_selection": "random",
    "tournament_size": 3,
    "num_threads": 4,
    "num_cores": 4,
    "eval_timeout": -1.0,
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit", "K": "K", "instance_mode": "im",
    "n_parents": "mu", "n_offspring": "lam", "max_generations": "mg",
    "dim": "dim", "budget_factor": "bf", "n_reps": "nr",
    "elitism": "elit", "parent_selection": "psel",
    "num_threads": "nt", "num_cores": "nc", "llm_model": "model", "llm_backend": "llm",
}


def _run_tag(args: argparse.Namespace) -> str:
    parts = []
    for key, default in _BASH_DEFAULTS.items():
        if not hasattr(args, key):
            continue
        val = getattr(args, key)
        if val == default:
            continue
        ab = _ABBREV.get(key, key)
        if isinstance(val, bool):
            if val:
                parts.append(ab)
        elif key == "llm_model" and isinstance(val, str) and "/" in val:
            parts.append(f"{ab}{val.split('/')[-1]}")
        else:
            parts.append(f"{ab}{val}")
    return "_".join(parts) if parts else "default"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random Policy (Fixed-K) LLaMEA on BBOB (tiny).")
    p.add_argument("--llm-model", type=str, default=_BASH_DEFAULTS["llm_model"])
    p.add_argument("--llm-backend", type=str, default=_BASH_DEFAULTS["llm_backend"],
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--t-iter", type=int, default=100,
                   help="LLaMEA total CANDIDATE budget (incl. init pop). Ignored when "
                        "--max-generations is set.")
    p.add_argument("--max-generations", type=int, default=None,
                   help="Evolution generations; overrides --t-iter: budget = n_parents "
                        "+ max_generations x n_offspring.")
    p.add_argument("--n-parents", type=int, default=10)
    p.add_argument("--n-offspring", type=int, default=10)
    p.add_argument("--elitism", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evolution-mode", type=str, default="population",
                   choices=["population", "single"])
    p.add_argument("--K", type=int, default=1,
                   help="Number of (fid,iid) instances (of the 72) each candidate is scored on.")
    p.add_argument("--instance-mode", type=str, default="fixed", choices=["fixed", "random"],
                   help="K-subset selection: 'fixed' (one subset, seeded by --seed, reused for "
                        "all candidates) or 'random' (fresh subset per candidate, seeded by source).")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--budget-factor", type=int, default=2000)
    p.add_argument("--n-reps", type=int, default=1,
                   help="Repeated seeded runs per (fid,iid) instance (seeds 0..n_reps-1). "
                        "Total evals per candidate = K x n_reps.")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fix-init-pop", action="store_true",
                   help="Use src/init_pop/llamea_24_bbob.json for generation 0 instead of the LLM.")
    p.add_argument("--parent-selection", type=str, default="random",
                   choices=["random", "tournament", "roulette"])
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--label", type=str, default="tiny_llamea_bbob")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--eval-timeout", type=float, default=-1.0,
                   help="Per-(candidate, instance) wall-clock cap (s); -1 = auto-scale, 0 = off, >0 = fixed.")
    p.add_argument("--use-wandb", action="store_true", default=False)
    return p.parse_args(argv)


def _final_eval(log_dir: pathlib.Path, dim: int, budget_factor: int, n_rep: int,
                eval_timeout: float, num_cores: int) -> None:
    """Post-evolution validation: re-evaluate the trajectory on the FULL 72-instance BBOB
    grid via analyses.eval_bbob.evaluate_single (mode='valid') -> valid_trajectory.json.
    Mirrors the tiny/eoh_* runners' _final_eval (single-trajectory -> has_incumbent=False)."""
    from analyses.eval_bbob import evaluate_single, _to_serialisable
    import json
    print("=== Post-Evolution Final Evaluation (eval_bbob.py, full 72-grid) ===", flush=True)
    res = evaluate_single(exp_path=log_dir, dim=dim, budget_factor=budget_factor, n_rep=n_rep,
                          eval_timeout=eval_timeout, n_cores=num_cores,
                          mode="valid", has_incumbent=False)
    out_path = log_dir / "valid_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=_to_serialisable)
    print(f"Saved post-evolution validation trajectory -> {out_path}", flush=True)


def run_one(args: argparse.Namespace, seed: int, tag: str, log_dir: pathlib.Path,
            cache_dir: pathlib.Path, wandb_logger=None) -> Solution:
    random.seed(seed)
    np.random.seed(seed)

    if args.evolution_mode == "single":
        n_parents, n_offspring, elitism = 1, 1, True
    else:
        n_parents, n_offspring, elitism = args.n_parents, args.n_offspring, args.elitism
    print(f"  instance_mode={args.instance_mode}  K={args.K}  evolution_mode={args.evolution_mode}  "
          f"n_parents={n_parents}  n_offspring={n_offspring}  elitism={elitism}", flush=True)

    # Budget: two distinct terms (match reprod/llamea_bbob on the FULL 72-pool).
    #   target_budget (the reprod-equivalent budget CAP, computed once, full pool):
    #       = n_offspring * max_generations * _N_UNIQUE_INSTANCES * n_reps
    #     i.e. the (candidate, instance) evaluations reprod would spend over all 72
    #     (fid,iid) pairs in --max-generations evolution generations.
    #   Tiny evaluates only K instances/candidate, so we SCALE UP the generation count
    #   (effective_max_generations) to spend that SAME budget on the K-subset — mirrors
    #   tiny/eoh_*'s effective_max_generations = target_budget / (pop_size * K).
    #   used_budget during the run stays honest = candidates * K * n_reps (logged by
    #   the TrajectoryLogger, evals_per_candidate=K*n_reps).
    _full_pool = _N_UNIQUE_INSTANCES  # 72 (fid,iid) pairs
    if args.max_generations is not None and args.max_generations >= 0:
        target_budget = n_offspring * args.max_generations * _full_pool * args.n_reps
        effective_max_generations = math.ceil(
            target_budget / max(1, n_offspring * args.K * args.n_reps))
        llamea_budget = n_parents + effective_max_generations * n_offspring
        print(f"  [budget] full-pool target_budget = n_offspring({n_offspring}) x "
              f"max_generations({args.max_generations}) x pool({_full_pool}) x n_reps({args.n_reps}) "
              f"= {target_budget}  (the reprod-equivalent cap)")
        print(f"  [budget] K={args.K} -> effective_max_generations = ceil(target_budget / "
              f"(n_offspring x K x n_reps)) = {effective_max_generations}  "
              f"-> llamea_budget = {n_parents} + {effective_max_generations} x {n_offspring} "
              f"= {llamea_budget} candidates")
        print(f"  [budget] used_budget tracks K-subset spend = candidates x K x n_reps "
              f"(final ~{llamea_budget * args.K * args.n_reps}); full-pool cap = {target_budget}")
    else:
        llamea_budget = args.t_iter
        print(f"  [budget] --t-iter={llamea_budget} candidates (no max-generations scaling)", flush=True)

    if args.eval_timeout is not None and args.eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(args.budget_factor, args.dim)
        print(f"  [eval-timeout] auto = {eval_timeout:.0f}s per (candidate, instance)", flush=True)
    else:
        eval_timeout = args.eval_timeout

    llm = _build_llm(args, cache_dir, log_dir)
    fixed_init_population = None
    if args.fix_init_pop:
        fixed_path = ROOT / "src" / "init_pop" / "llamea_24_bbob.json"
        fixed_init_population = load_fixed_initial_population(fixed_path, n_parents)
        print(f"  [fixed-init-pop] loaded {len(fixed_init_population)} heuristics from {fixed_path}")

    es = TinyLLaMEA_BBOB(
        K=args.K,
        instance_mode=args.instance_mode,
        llm=llm,
        log_dir=log_dir,
        label=f"{args.llm_model.split('/')[-1]}_{tag}_seed{seed}",
        n_parents=n_parents,
        n_offspring=n_offspring,
        elitism=elitism,
        budget=llamea_budget,
        dim=args.dim,
        budget_factor=args.budget_factor,
        n_reps=args.n_reps,
        parent_selection=args.parent_selection,
        tournament_size=args.tournament_size,
        num_threads=args.num_threads,
        num_cores=args.num_cores,
        eval_timeout=eval_timeout,
        seed=seed,
        instance_pool_mode="hetero",
        wandb_logger=wandb_logger,
        fixed_init_population=fixed_init_population,
    )
    best = es.run()
    # Post-evolution full-pool validation (auto, like tiny/eoh_*). Best-effort.
    try:
        _final_eval(log_dir=log_dir, dim=args.dim, budget_factor=args.budget_factor,
                    n_rep=args.n_reps, eval_timeout=eval_timeout, num_cores=args.num_cores)
    except Exception as e:
        print(f"  [_final_eval] WARN: final evaluation failed: {type(e).__name__}: {e}", flush=True)
    return best


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")

    tag = _run_tag(args)
    now = _dt.datetime.now()
    dt_stamp = args.run_stamp or f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"

    print("=== Random Policy (Fixed-K) LLaMEA on BBOB (tiny) ===")
    print(f"Model       : {args.llm_model}")
    print(f"K / mode    : {args.K} / {args.instance_mode}  (of {_N_UNIQUE_INSTANCES} (fid,iid) pairs)")
    print(f"Dim         : {args.dim}   Per-inst bgt: {args.budget_factor} * {args.dim} = {args.budget_factor * args.dim}")
    print(f"Runs        : {args.n_runs}  (seeds {args.seed} .. {args.seed + args.n_runs - 1})")
    print(f"Tag         : {tag}")

    import yaml
    for i in range(args.n_runs):
        seed = args.seed + i
        print(f"\n--- Run {i + 1}/{args.n_runs}  (seed={seed}) ---")

        log_dir = make_log_dir(args.log_root, args.label, dt_stamp, seed, tag=tag)
        cache_dir = args.cache_root / dt_stamp / str(seed)
        cache_dir.mkdir(parents=True, exist_ok=True)

        args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
        with open(log_dir / "args.yaml", "w") as _f:
            yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

        mirror_stdout_to(log_dir / "terminal.txt")
        print(f"terminal output mirrored -> {log_dir / 'terminal.txt'}")

        run_name = f"tiny_llamea_{dt_stamp.replace('/', '_')}_{seed}_{tag}"
        wandb_logger = make_wandb_logger(
            enabled=args.use_wandb, project="llm4ad", name=run_name, config=args_dict)

        best = run_one(args, seed, tag, log_dir, cache_dir, wandb_logger)
        if best is not None:
            print(f"Best: {best.name}  AOCC={best.fitness}")
        else:
            print("Best: (none — all candidates failed)")
        wandb_logger.finish()

    print("\n=== All Runs Completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
