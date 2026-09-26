"""Elitist Successive Halving EoH on the LLM4AD TSP-GLS task.

Same outer LLM call surface as ``reprod/eoh_tsp_gls.py`` (LLM4AD's ``EoHPrompt``
+ ``EoHSampler``, ``TSP_GLS_2O_Evaluation_wo_Time``) but replaces full-pool
evaluation + mu+lambda truncation with a cache-aware Successive Halving race
over the union of elites and offspring.

Two independent time limits apply per-instance:
  - 60s/instance inside ``_guided_local_search`` (gls.py:149): breaks the GLS
    iteration loop after 60s and returns the best tour found so far.
  - ``eval_timeout`` per-call wall-clock cap passed to ``_eval_one_instance_elitist``
    (default: 65s, matching the per-instance batch timeout in reprod/).

See :mod:`racing.base` for the shared racing infrastructure.
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
import tempfile
import time
import uuid
import yaml
from dataclasses import dataclass
from typing import Any, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, ConfigAS, OpenRouterClient, GoogleClient, OllamaClient,
                   MistralClient, vLLMClient, make_wandb_logger)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD, GoogleLLM4AD
from utils.logger import make_log_dir
from llm4ad.base.code import TextFunctionProgramConverter
from llm4ad.method.eoh.prompt import EoHPrompt
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.task.optimization.tsp_gls_2O.evaluation import (
    TSP_GLS_2O_Evaluation_wo_Time, solve_without_time, calculate_cost)
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData

from racing.base import RacingBase, CandidateRecord, _Tee, save_run_log, _run_tag
from utils.config import RaceState
from utils.race import _eval_one_instance_elitist


# BIG_PENALTY is the finite default cost for a TIMEOUT (a slow-but-valid
# heuristic killed at eval_timeout). It is large enough to rank last but stays
# finite so the config is penalised, not rejected — mirroring irace's PAR
# penalty (boundMax * boundPar) at packages/irace/R/race.R:491-496.
BIG_PENALTY = 1e6

# REJECT_COST marks an INVALID result — a compile/runtime crash inside the
# generated heuristic (SyntaxError, NameError, an exception in the GLS run).
# irace uses Inf for this: a rejected configuration is eliminated immediately
# and loses elite protection (packages/irace/R/race.R:1064-1073, and the
# comment at :493 "Inf / -Inf represent rejection"). Keeping crashes (Inf) and
# timeouts (finite BIG_PENALTY) distinct is what lets the race reject a broken
# heuristic while merely penalising a slow one.
REJECT_COST = float("inf")

# Upper bound for a per-task random seed = INT32_MAX (max signed 32-bit int).
# Seeds are conventionally 32-bit, and this mirrors irace, which draws task
# seeds up to `.Machine$integer.max` (= 2147483647) via `sample.int(2147483647L)`
# (packages/irace/R/irace.R:562, race_state.R:65). `RandomState.randint(0, MAX)`
# yields a seed in [0, MAX).
_SEED_MAX = 2 ** 31 - 1  # 2_147_483_647 (INT32_MAX)


def _usage_fields(sampler) -> dict:
    """Token counts for the calling thread's most recent LLM call.

    Walks sampler -> LLM4AD adapter -> CachedLLM and reads the per-thread
    usage, so threaded sampling cannot attribute another thread's tokens to
    this record. Returns zeros for backends that report no usage (and never
    raises: logging must not be able to abort a run).
    """
    try:
        cached = getattr(getattr(sampler, "llm", None), "_cached", None)
        u = cached.take_last_usage() if cached is not None else {}
    except Exception:
        u = {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "cached_tokens": int(u.get("cached_tokens", 0) or 0),
        "reasoning_tokens": int(u.get("reasoning_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
    }

def _sh_schedule(pool_size: int, elite_size: int, reduction_factor: float,
                 min_instances: int, max_instances: int) -> list[int]:
    """Return strictly increasing SH resource levels ending at ``max_instances``.

    The number of levels follows the usual SH reduction trajectory.  The last
    level is always retained even when pruning already reaches ``elite_size``:
    that level is the final, full-resource comparison of the survivors.
    """
    if max_instances < 1:
        return []
    start = min(max(1, int(min_instances)), max_instances)
    eta = max(1.1, float(reduction_factor))
    rounds = (max(1, math.ceil(math.log(pool_size / elite_size, eta)))
              if pool_size > elite_size else 1)
    if rounds == 1:
        return [max_instances]
    levels = []
    for round_idx in range(rounds):
        ratio = (max_instances / start) ** (round_idx / (rounds - 1))
        level = min(max_instances, max(start + round_idx, int(round(start * ratio))))
        if not levels or level > levels[-1]:
            levels.append(level)
    if levels[-1] != max_instances:
        levels[-1] = max_instances
    return levels


@dataclass(frozen=True)
class SeededInstance:
    """One irace-style race task: a (base TSP instance, seed) pair.

    irace identifies a task by an (instanceID, seed) pair (race.R:120,
    `instances_log`); the seed is fixed per task and SHARED across every
    candidate evaluated on it, so the Friedman/t-test compares configs under
    identical random conditions. We mirror that here: `instance` is the shared,
    read-only TSP instance; `seed` is fixed for this (base instance, repetition)
    and passed into the GLS evaluation so a non-deterministic LLM heuristic is
    reproducible and configs stay comparable.

    `base_idx` / `rep` are bookkeeping: which base instance and which repetition
    this task is (for logging / pool extension), not used for scoring.
    """

    instance: Any
    seed: int
    base_idx: int
    rep: int


def _unwrap(task_or_inst) -> tuple:
    """Return (instance, seed) from a SeededInstance, or (inst, None) for a bare
    instance. Lets every scorer accept either form so the deterministic path
    (seed=None) keeps working unchanged."""
    if isinstance(task_or_inst, SeededInstance):
        return task_or_inst.instance, task_or_inst.seed
    return task_or_inst, None


_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "pop_size": 10,
    "max_generations": -1,
    "ref_max_generations": 20,
    "n_instances": 64,
    "problem_size": 100,
    "selection_num": 5,
    "t_first": 5,
    "t_each": 1,
    "sh_reduction_factor": 2,
    "sh_min_instances": 2,
    "elitist_new_instances": 1,
    "alpha": 0.05,
    "test_type": "friedman",
    "posthoc_test_type": "conover",
    "elitist_limit": 2,
    "eval_timeout": 65.0,
    "num_threads": 4,
    "num_cores": 4,
    "llm_model": "qwen/qwen3-coder-next",
    "llm_backend": "openrouter",
    "elitist": True,
    "save_pop": False,
    "early_stopping_non_elitist": False,
    "deal_with_crashed": "rejection",
    "deterministic": False,
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "pop_size": "ps",
    "max_generations": "mg",
    "ref_max_generations": "rmg",
    "n_instances": "ni",
    "problem_size": "prob",
    "selection_num": "sn",
    "t_first": "tf",
    "t_each": "te",
    "sh_reduction_factor": "srf",
    "sh_min_instances": "smi",
    "elitist_new_instances": "tni",
    "alpha": "a",
    "test_type": "tt",
    "posthoc_test_type": "phtt",
    "elitist_limit": "elimit",
    "eval_timeout": "eto",
    "num_threads": "nth",
    "num_cores": "nc",
    "llm_model": "model",
    "llm_backend": "llm",
    "elitist": "elitist",
    "save_pop": "savpop",
    "early_stopping_non_elitist": "es",
    "deal_with_crashed": "dwc",
    "deterministic": "det",
}


# --------------------------------------------------------------------------- #
# Concorde-based per-instance optimum
# --------------------------------------------------------------------------- #
_CONCORDE_COORD_SCALE = 1e6


def _silence_fds():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved


def _restore_fds(saved):
    os.dup2(saved[0], 1)
    os.dup2(saved[1], 2)
    os.close(saved[0])
    os.close(saved[1])


def _opt_cost(inst) -> float:
    from concorde.tsp import TSPSolver
    coords = np.asarray(inst[0] if isinstance(inst, tuple) else inst.positions)
    xs = (coords[:, 0] * _CONCORDE_COORD_SCALE).tolist()
    ys = (coords[:, 1] * _CONCORDE_COORD_SCALE).tolist()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        saved = _silence_fds()
        try:
            solver = TSPSolver.from_data(xs, ys, norm="EUC_2D",
                                         name=uuid.uuid4().hex)
            sol = solver.solve(verbose=False)
        finally:
            _restore_fds(saved)
            os.chdir(cwd)
    tour = list(sol.tour)
    cost = 0.0
    for a, b in zip(tour, tour[1:] + [tour[0]]):
        cost += float(np.linalg.norm(coords[a] - coords[b]))
    return cost


# --------------------------------------------------------------------------- #
# TSP-GLS per-instance scoring
# --------------------------------------------------------------------------- #

def score_tsp_inst(fn: callable, inst) -> float:
    """Run ``fn`` as ``update_edge_distance`` on one task; return tour cost.

    ``inst`` may be a :class:`SeededInstance` (carrying the per-task seed) or a
    bare TSP instance (legacy, seed=None). The seed is passed into the GLS
    evaluation so a non-deterministic LLM heuristic is reproducible for a given
    (instance, seed) task.

    A raised exception (a crash inside the generated heuristic or the GLS run)
    is an INVALID result and returns ``REJECT_COST`` (Inf) so the race can
    reject the configuration, matching irace's Inf-means-rejection convention.
    A non-finite tour cost from a successful run is likewise treated as invalid.
    """
    instance, seed = _unwrap(inst)
    try:
        cost = solve_without_time(instance, fn, seed=seed)
        return float(cost) if math.isfinite(float(cost)) else REJECT_COST
    except Exception:
        return REJECT_COST


def score_tsp_config(cfg: "ConfigAS", inst) -> tuple:
    """Picklable process-pool worker. ``inst`` is a SeededInstance or bare instance.

    Returns ``(cost, cpu_seconds)`` — ``cpu_seconds`` is this worker's own CPU
    time (``time.process_time``) for the GLS solve, so the race harness can sum
    it across worker processes into a run-level CPU odometer. GLS is CPU-bound
    single-threaded numba, so CPU ~= wall for a single eval."""
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    try:
        fn = cfg.callable
        cost = score_tsp_inst(fn, inst)
    except Exception:
        cost = REJECT_COST
    cpu = time.process_time() - cpu0
    if os.environ.get("EOH_WORKER_TRACE"):
        dt = time.perf_counter() - t0
        instance, seed = _unwrap(inst)
        raw_id = getattr(instance, "_id", None)
        iid = raw_id + 1 if isinstance(raw_id, int) else "?"
        seed_str = f"/seed={seed}" if seed is not None else ""
        print(f"    [eval-worker pid={os.getpid()}] {cfg.id}:{iid}{seed_str} "
              f"-> cost={cost:.4f}; cpu={cpu:.2f}s wall={dt:.2f}s", flush=True)
    return cost, cpu


# --------------------------------------------------------------------------- #
# SHEoH (TSP-GLS)
# --------------------------------------------------------------------------- #


class SHEoH(RacingBase):
    """EoH outer loop with Successive Halving race as fitness + selection for TSP-GLS."""

    _pool_runner = staticmethod(score_tsp_config)
    _big_penalty = BIG_PENALTY
    _track_cpu = True   # score_tsp_config returns (cost, cpu); log CPU odometer

    def __init__(
        self,
        *,
        evaluation,
        instances,
        label,
        log_dir,
        llm,
        pop_size=10,
        max_generations=None,
        selection_num=5,
        budget_cap=None,
        t_first=5,
        t_each=1,
        sh_reduction_factor=2,
        sh_min_instances=2,
        alpha=0.05,
        seed=1,
        opt_by_idx=None,
        mean_opt=None,
        num_threads=1,
        num_cores=1,
        eval_timeout=None,
        timeout_cost=BIG_PENALTY,
        test_type: str = "friedman",
        posthoc_test_type: str = "conover",
        elitist: bool = True,
        elitist_new_instances: int = 1,
        elitist_limit: int = 2,
        save_pop: bool = False,
        early_stopping_non_elitist: bool = False,
        deal_with_crashed: str = "rejection",
        deterministic: bool = False,
        wandb_logger=None,
    ):
        self.sh_reduction_factor = max(1.1, float(sh_reduction_factor))
        self.sh_min_instances = max(1, int(sh_min_instances))
        self._deterministic = deterministic
        if not deterministic:
            self._base_instances = list(instances)
            self._seed_rng = np.random.RandomState(int(seed))
            self._n_reps = 0
            instances = self._build_rep(self._base_instances)
        else:
            self._base_instances = None
        super().__init__(
            instances=instances,
            label=label,
            log_dir=log_dir,
            pop_size=pop_size,
            max_generations=max_generations,
            budget_cap=budget_cap,
            t_first=t_first,
            t_each=t_each,
            alpha=alpha,
            seed=seed,
            num_threads=num_threads,
            num_cores=num_cores,
            test_type=test_type,
            posthoc_test_type=posthoc_test_type,
            elitist=elitist,
            elitist_new_instances=elitist_new_instances,
            elitist_limit=elitist_limit,
            save_pop=save_pop,
            early_stopping_non_elitist=early_stopping_non_elitist,
            deal_with_crashed=deal_with_crashed,
            wandb_logger=wandb_logger,
        )
        self.llm = llm
        self.selection_num = selection_num
        self.opt_by_idx = opt_by_idx or {}
        self.mean_opt = mean_opt
        self._eval_timeout = (
            float(eval_timeout) if eval_timeout and float(eval_timeout) > 0 else None
        )
        self._big_penalty = float(timeout_cost)

        self._perf_log_path = self.log_dir / "instance_seed_perf.jsonl"
        self._prompt_log_path = self.log_dir / "llm_prompts.jsonl"
        self._perf_logged: set = set()
        self._llm_call_idx = 0

        self.template_str = evaluation.template_program
        self.task_desc = evaluation.task_description
        self.problem_size = getattr(evaluation, "problem_size", 100)
        self.template_fn = TextFunctionProgramConverter.text_to_function(
            self.template_str
        )
        self.sampler = EoHSampler(llm, self.template_str)

        self._best_score = float("inf")   # lower mean_cost = better
        self._best_gap = float("inf")
        self._needs_pre_eval_cleanup = False
        import itertools

        self._op_iter = itertools.cycle(("e1", "e2", "m1", "m2"))

    # ---- irace-style (instance, seed) task pool --------------------------

    def _build_rep(self, base_instances) -> list:
        rep = self._n_reps
        self._n_reps += 1
        tasks = []
        for base_idx, inst in enumerate(base_instances):
            seed = int(self._seed_rng.randint(0, _SEED_MAX))
            tasks.append(SeededInstance(instance=inst, seed=seed,
                                       base_idx=base_idx, rep=rep))
        return tasks

    def _ensure_task_pool(self, next_instance: int) -> None:
        if self._base_instances is None:
            return
        while next_instance > len(self.instances):
            new_rep = self._build_rep(self._base_instances)
            self.instances = self.instances + new_rep
            print(f"  [task-pool] extended to {len(self.instances)} tasks "
                  f"(rep {self._n_reps - 1}, "
                  f"{len(self._base_instances)} base x {self._n_reps} reps)")

    # ---- Elitist Successive Halving Sub-Race Engine -----------------------

    def _race(self, records: List[CandidateRecord], next_instance: int) -> dict:
        """Run one cache-aware, full-final-round successive-halving race."""
        if not records:
            return {
                "survivors": [], "all_records_sorted": [],
                "next_instance": next_instance, "experiments_used": 0,
                "break_reason": "no candidates", "table_rows": [],
                "step_trace": [], "cpu_seconds": 0.0,
                "seen_instances": [],
            }

        cfgs = [r.cfg for r in records]
        id2rec = {r.cfg.id: r for r in records}
        if getattr(self, "_needs_pre_eval_cleanup", False):
            for rec in records:
                rec.cfg.reset_history()
                rec.mean_cost = float("inf")
            self._needs_pre_eval_cleanup = False
            print(
                "  [pre-eval cleanup] cfg histories and mean_costs reset "
                f"for {len(records)} candidates before first race",
                flush=True,
            )

        t0 = time.time()
        K0 = len(cfgs)
        eta = self.sh_reduction_factor
        M = self.pop_size
        max_inst_available = len(self.instances)
        schedule = _sh_schedule(K0, M, eta, self.sh_min_instances,
                                max_inst_available)
        start_n = schedule[0] if schedule else 0
        budget_remaining = (self.budget_cap - self.budget_used
                            if self.budget_cap is not None else float("inf"))

        active_cfgs = list(cfgs)
        for c in cfgs:
            c._race_init_costs = set(c.costs_by_inst.keys())
        entry_elites = [c for c in cfgs if c.costs_by_inst and c.alive]
        experiments_used = 0
        seen_inst_indices: list[int] = []
        step_trace: list[dict] = []
        state = RaceState(configs=cfgs, instances=[], instances_log=self.instances,
                          metric="mean_cost")
        runner = self._make_runner()
        completed_rounds = 0
        budget_stopped = False

        print(f"\n# SH Sub-Race -- {K0} candidates ({M} elites target, eta={eta:g}, N_min={start_n}, N_max={max_inst_available})")
        print(f"  scheduled rounds ({len(schedule)}): target instances {schedule}")

        for round_idx, target_n_insts in enumerate(schedule):
            active_cfgs = [c for c in active_cfgs if c.alive]
            if not active_cfgs:
                break
            self._ensure_task_pool(target_n_insts)
            state.instances_log = self.instances

            required = sum(
                not c.has(inst_idx)
                for c in active_cfgs
                for inst_idx in range(1, target_n_insts + 1)
            )
            if required > budget_remaining:
                budget_stopped = True
                print(f"  [SH budget guard] need {required} fresh evaluations for "
                      f"round {round_idx + 1}, only {budget_remaining} remain; "
                      "leaving the previous completed level intact")
                break

            fresh_evals = cached_evals = 0
            for inst_idx in range(1, target_n_insts + 1):
                alive_before = len(active_cfgs)
                missing = [c for c in active_cfgs if not c.has(inst_idx)]
                cached_evals += len(active_cfgs) - len(missing)
                if missing:
                    self._eval_pool = _eval_one_instance_elitist(
                        state, inst_idx, self.instances[inst_idx - 1], missing,
                        runner, None, eval_pool=self._eval_pool,
                        pool_runner=self._pool_runner,
                        eval_timeout=self._eval_timeout,
                        timeout_cost=self._big_penalty,
                        crash_penalty=(None if self.deal_with_crashed == "rejection"
                                       else self._big_penalty),
                        pool_recreate=self._recreate_eval_pool,
                        task_step=inst_idx,
                    )
                    fresh_evals += len(missing)
                if inst_idx not in seen_inst_indices:
                    seen_inst_indices.append(inst_idx)
                step_trace.append({
                    "task_step": inst_idx, "inst_idx": inst_idx,
                    "n_alive_before": alive_before,
                    "n_alive_after": len([c for c in active_cfgs if c.alive]),
                    "eliminated_cand_ids": [c.id for c in active_cfgs if not c.alive],
                    "test_ran": False, "protection_active": False,
                    "p_value_omnibus": None, "q_var": float("nan"),
                    "kendall_w": float("nan"), "rho": float("nan"),
                    "cum_evaluations": state.total_evaluations,
                })
                active_cfgs = [c for c in active_cfgs if c.alive]
                if not active_cfgs:
                    break

            experiments_used += fresh_evals
            budget_remaining -= fresh_evals
            self.budget_used += fresh_evals

            def _mean_cost(c: ConfigAS) -> float:
                vals = [c.costs_by_inst[k] for k in range(1, target_n_insts + 1)
                        if k in c.costs_by_inst]
                return float(np.mean(vals)) if vals else float("inf")

            active_cfgs.sort(key=_mean_cost)
            completed_rounds += 1
            b_cap_str = f"{self.budget_cap}" if self.budget_cap is not None else "inf"
            print(f"  [SH Round {round_idx+1}/{len(schedule)}] candidates={len(active_cfgs)} | instances={target_n_insts} | fresh_evals={fresh_evals}, 0-cost cached={cached_evals} | total budget={self.budget_used}/{b_cap_str}")
            if cached_evals > 0:
                print(f"    [elitist cache] 0-cost cache hit on {cached_evals} evaluation(s) for carried-over elites (0 evaluation budget spent)")

            if round_idx < len(schedule) - 1:
                keep_k = max(M, int(math.floor(len(active_cfgs) / eta)))
                if keep_k < len(active_cfgs):
                    dropped = active_cfgs[keep_k:]
                    for d in dropped:
                        d.alive = False
                    active_cfgs = active_cfgs[:keep_k]
                    elim_ids = [d.id for d in dropped]
                    print(f"    [halving] eliminated {len(dropped)} candidates ({elim_ids}) -> {len(active_cfgs)} remaining")
                    next_target = schedule[round_idx + 1] if round_idx < len(schedule) - 1 else target_n_insts
                    print(f"    Surviving candidates carried to SH Round {round_idx + 2} (target instances: {next_target}):")
                    print("    +------+----------------------+-------------+---------------+------------------------------+")
                    print("    | Rank | Candidate ID         | Mean Cost   | Eval / Target | Elitist 0-Cost Cache         |")
                    print("    +------+----------------------+-------------+---------------+------------------------------+")
                    for rank_idx, c in enumerate(active_cfgs, 1):
                        m_cost = _mean_cost(c)
                        cost_str = f"{m_cost:.4f}" if math.isfinite(m_cost) else "inf"
                        n_eval = len([v for k, v in c.costs_by_inst.items() if k <= target_n_insts])
                        eval_str = f"{n_eval} / {target_n_insts}"
                        prior_cached = len([k for k in c.costs_by_inst.keys() if k <= target_n_insts and k in getattr(c, "_race_init_costs", set())])
                        cid = str(c.id)
                        if len(cid) > 20:
                            cid = cid[:17] + "..."
                        total_lifetime = len(c.costs_by_inst)
                        if prior_cached > 0:
                            cache_info = f"{prior_cached} cached ({total_lifetime} lifetime)"
                        else:
                            cache_info = f"0 cached (new offspring)"
                        print(f"    | {rank_idx:4d} | {cid:20s} | {cost_str:>11s} | {eval_str:^13s} | {cache_info:28s} |")
                    print("    +------+----------------------+-------------+---------------+------------------------------+")

        if completed_rounds == 0:
            active_cfgs = entry_elites

        if len(active_cfgs) > M:
            dropped = active_cfgs[M:]
            for d in dropped:
                d.alive = False
            active_cfgs = active_cfgs[:M]

        # 3. Finalize Records & Ranks
        alive_ids = {c.id for c in active_cfgs if c.alive}
        seen_list = sorted(seen_inst_indices)

        for c in cfgs:
            rec = id2rec[c.id]
            vals = [c.costs_by_inst[k] for k in seen_list if k in c.costs_by_inst]
            rec.mean_cost = float(np.mean(vals)) if vals else float("inf")
            rec.sum_ranks = rec.mean_cost
            rec.n_evals = len(vals)
            rec.evaluated_idxs = sorted(k - 1 for k in c.costs_by_inst if k in seen_list)
            rec.eliminated_at = -1 if c.id in alive_ids else rec.n_evals
            rec.survived = c.id in alive_ids

        survivor_cfgs = [c for c in cfgs if c.id in alive_ids]
        survivors = sorted(
            [id2rec[c.id] for c in survivor_cfgs],
            key=lambda r: r.mean_cost,
        )

        eliminated_cfgs = [c for c in cfgs if c.id not in alive_ids]
        eliminated_sorted = sorted(
            [id2rec[c.id] for c in eliminated_cfgs],
            key=lambda r: r.mean_cost,
        )

        all_records_sorted = survivors + eliminated_sorted

        dt_race = time.time() - t0
        break_reason = "SH budget cap reached before a complete round" if budget_stopped else "SH rounds complete"
        best_cand = survivors[0] if survivors else None
        best_str = f"{best_cand.cfg.id} (mean_cost={best_cand.mean_cost:.4f})" if best_cand else "None"
        print(f"  [SH Sub-Race Summary] {break_reason} | survivors={len(survivors)}/{K0} | best={best_str} | race_wall={dt_race:.2f}s")
        final_target = seen_list[-1] if seen_list else 0
        print(f"  Final Surviving Elites from SH Sub-Race (target instances: {final_target}):")
        print("  +------+----------------------+-------------+---------------+------------------------------+")
        print("  | Rank | Candidate ID         | Mean Cost   | Eval / Target | Elitist 0-Cost Cache         |")
        print("  +------+----------------------+-------------+---------------+------------------------------+")
        for rank_idx, r in enumerate(survivors, 1):
            c = r.cfg
            m_cost = r.mean_cost
            cost_str = f"{m_cost:.4f}" if math.isfinite(m_cost) else "inf"
            n_eval = len([v for k, v in c.costs_by_inst.items() if k <= final_target])
            eval_str = f"{n_eval} / {final_target}"
            prior_cached = len([k for k in c.costs_by_inst.keys() if k <= final_target and k in getattr(c, "_race_init_costs", set())])
            cid = str(c.id)
            if len(cid) > 20:
                cid = cid[:17] + "..."
            total_lifetime = len(c.costs_by_inst)
            if prior_cached > 0:
                cache_info = f"{prior_cached} cached ({total_lifetime} lifetime)"
            else:
                cache_info = "0 cached (new offspring)"
            print(f"  | {rank_idx:4d} | {cid:20s} | {cost_str:>11s} | {eval_str:^13s} | {cache_info:28s} |")
        print("  +------+----------------------+-------------+---------------+------------------------------+")

        return {
            "survivors": survivors,
            "all_records_sorted": all_records_sorted,
            "next_instance": next_instance + experiments_used,
            "experiments_used": experiments_used,
            "break_reason": break_reason,
            "table_rows": [],
            "step_trace": step_trace,
            "cpu_seconds": 0.0,
            "seen_instances": seen_list,
        }

    # ---- Instance metadata -----------------------------------------------

    def _task_meta(self, task_idx: int) -> tuple:
        if isinstance(task_idx, (list, tuple)):
            task_idx = task_idx[0]
        task = self.instances[task_idx - 1]
        if isinstance(task, SeededInstance):
            return task.base_idx, task.seed, task.rep
        return task_idx - 1, None, 0

    def _instance_label(self, inst_idx: int) -> str:
        base_idx, seed, _rep = self._task_meta(inst_idx)
        s = seed if seed is not None else self.seed
        return f"i{base_idx}_s{s}"

    def _append_perf_log(self, gen_id: int, race_records) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        order = list(getattr(self, "_last_instance_order", []) or [])
        lines = []
        for r in race_records:
            for task_idx, cost in r.cfg.costs_by_inst.items():
                key = (r.cfg.id, task_idx)
                if key in self._perf_logged:
                    continue
                self._perf_logged.add(key)
                base_idx, seed, rep = self._task_meta(task_idx)
                c = float(cost)
                extras = self._perf_row_extras(r.cfg, task_idx, cost, order)
                lines.append(json.dumps({
                    "gen": int(gen_id),
                    "cand_id": r.cfg.id,
                    "task_idx": int(task_idx),
                    "base_idx": int(base_idx),
                    "instance_idx_k": extras["instance_idx_k"],
                    "race_step": extras["race_step"],
                    "rep": int(rep),
                    "seed": (int(seed) if seed is not None else None),
                    "cost": (c if math.isfinite(c) else str(c)),
                    "status": extras["status"],
                    "wall_s": extras["wall_s"],
                    "ts": ts,
                }))
        if lines:
            with open(self._perf_log_path, "a") as f:
                f.write("\n".join(lines) + "\n")

    # ---- Pre-evaluation of initial population ----------------------------

    def _pre_evaluate(self, init_recs: List[CandidateRecord]) -> None:
        n_sample = min(self.t_first, len(self.instances))
        if n_sample == 0 or not init_recs:
            return
        idxs = self.rng.choice(
            len(self.instances), size=n_sample, replace=False
        ).tolist()
        sampled = [self.instances[i] for i in idxs]
        print(
            f"  [pre-eval] evaluating {len(init_recs)} candidates on "
            f"{n_sample} randomly sampled instances (indices {idxs})"
        )
        t0 = time.time()
        for rec in init_recs:
            costs = []
            try:
                fn = rec.cfg.callable
            except Exception as exc:
                print(
                    f"    [pre-eval] {rec.cfg.id} callable error -- {exc}; "
                    f"mean_cost stays inf"
                )
                continue
            cand_cpu = 0.0
            cand_wall = 0.0
            for inst, i_idx in zip(sampled, idxs):
                cpu0 = time.process_time()
                wall0 = time.perf_counter()
                c = score_tsp_inst(fn, inst)
                cpu = time.process_time() - cpu0
                wall = time.perf_counter() - wall0
                cand_cpu += cpu
                cand_wall += wall
                self._cpu_seconds_used += cpu
                costs.append(c)
                self.budget_used += 1
                print(f"    [pre-eval] {rec.cfg.id}:{i_idx + 1} "
                      f"-> cost={c:.4f}; cpu={cpu:.2f}s wall={wall:.2f}s",
                      flush=True)
            valid = [c for c in costs if math.isfinite(c) and c < BIG_PENALTY]
            rec.mean_cost = float(np.mean(valid)) if valid else float("inf")
            print(
                f"    [pre-eval] {rec.cfg.id}: mean_cost={rec.mean_cost:.4f} "
                f"({len(valid)}/{n_sample} valid evals)  "
                f"cpu_total={cand_cpu:.1f}s wall_total={cand_wall:.1f}s "
                f"(mean cpu={cand_cpu / n_sample:.2f}s/eval)",
                flush=True,
            )
        dt = time.time() - t0
        print(f"  [pre-eval] done in {dt:.1f}s  budget_used={self.budget_used}")
        self._needs_pre_eval_cleanup = True

    def _init_population(self) -> tuple:
        init_recs, next_instance = super()._init_population()
        self._pre_evaluate(init_recs)
        return init_recs, next_instance



    # ---- Abstract implementations ----------------------------------------

    def _make_runner(self):
        def runner(params, instance):
            cpu0 = time.process_time()
            cost = float(score_tsp_inst(params, instance))
            return cost, time.process_time() - cpu0

        return runner

    def _materialize(self, func, idx: int, gen: int = 0) -> CandidateRecord:
        prog = TextFunctionProgramConverter.function_to_program(func, self.template_str)
        src = str(prog)
        within = self._gen_cand_counter.get(gen, 0)
        self._gen_cand_counter[gen] = within + 1
        tag = "p" if gen == 0 else "c"
        cand_id = f"g{gen}_{tag}{within}"
        func.cand_id = cand_id
        cfg = ConfigAS(
            id=cand_id,
            source=src,
            entry_point=func.name,
            name=f"{func.name}#{idx}",
            global_ns={"np": np},
        )
        self._heuristics.append({"cand_id": cand_id, "gen_id": gen, "source": src})
        return CandidateRecord(cfg=cfg, candidate=func)

    def _sample_init(self, gen: int = 0):
        return self._sample("i1", gen=gen)

    def _raw_from_source(self, source: str):
        """Compile a fixed-init heuristic source into the EoH func form (--fix-init-pop)."""
        return TextFunctionProgramConverter.text_to_function(source)

    def _final_eval(self) -> dict:
        """Post-evolution validation via analyses/eval_tsp.py for BOTH incumbent rules ->
        valid_trajectory_mean_rank.json / valid_trajectory_mean_cost.json."""
        from analyses.eval_tsp import evaluate_single, _to_serialisable
        print(f"\n=== Post-Evolution Final Evaluation (eval_tsp.py) ===", flush=True)
        base_insts = self._base_instances if self._base_instances is not None else self.instances
        results: dict = {}
        for rule in ("mean_rank", "mean_cost"):
            try:
                res = evaluate_single(
                    exp_path=self.log_dir, n_instances=len(base_insts),
                    problem_size=self.problem_size, n_cores=self.num_cores,
                    mode="valid", has_incumbent=True, incumbent_key=f"incumbents_{rule}")
            except Exception as e:
                print(f"  [_final_eval/{rule}] skipped: {type(e).__name__}: {e}", flush=True)
                continue
            out_path = self.log_dir / f"valid_trajectory_{rule}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2, default=_to_serialisable)
            print(f"Saved {rule} validation -> {out_path}", flush=True)
            results[rule] = res
        return results

    def run(self):
        res = super().run()
        valid = self._final_eval()
        for rule in ("mean_rank", "mean_cost"):
            try:
                self._mlog.finalize_reliability(
                    getattr(self, f"incumbents_{rule}"), valid.get(rule), rule=rule)
            except Exception as e:
                print(f"  [manuscript-log/{rule}] WARN: reliability finalize failed: {e}", flush=True)
        return res

    def _compute_gap(self, rec: CandidateRecord) -> float:
        if rec.n_evals == 0 or not math.isfinite(rec.mean_cost) or not rec.evaluated_idxs:
            return float("inf")
        if self.mean_opt is None or self.mean_opt <= 0:
            return float("inf")
        return float((rec.mean_cost - self.mean_opt) / self.mean_opt)

    def _record_generation(self, gen_id, race_records, table_rows=None):
        race_order = list(self._last_instance_order)
        perfs = []
        for r in race_records:
            gap = self._compute_gap(r)
            if r.n_evals > 0 and math.isfinite(r.mean_cost) and r.mean_cost < BIG_PENALTY:
                score = float(r.mean_cost)
            else:
                score = float(BIG_PENALTY)
            # lower mean_cost = better
            if r.n_evals > 0 and math.isfinite(r.mean_cost) and score < self._best_score:
                self._best_score = score
                self._best_record = r
                if math.isfinite(gap):
                    self._best_gap = gap
            elif math.isfinite(gap) and gap < self._best_gap:
                self._best_gap = gap
            perfs.append(
                {
                    "cand_id": r.cfg.id,
                    "num_eval_instances": int(r.n_evals),
                    "instance_order": list(race_order),
                    "instances_evaluated": list(r.evaluated_idxs),
                    "score": score,
                    "gap": float(gap),
                    "survived": bool(r.survived),
                }
            )
        survivors_snapshot = [
            {
                "score": (
                    float(r.mean_cost)
                    if math.isfinite(r.mean_cost)
                    else float(BIG_PENALTY)
                )
            }
            for r in race_records
            if r.survived
        ]
        self.generations.append(
            {
                "gen_id": int(gen_id),
                "performances": perfs,
                "population": survivors_snapshot,
                "used_budget": int(self.budget_used),
            }
        )
        for row in table_rows or []:
            entry = {"gen_id": int(gen_id), **row}
            self.trajectory.append(entry)
            if self.wandb_logger:
                self.wandb_logger.log_trajectory_row(entry)
        save_run_log(self.log_dir, self.label, self.trajectory, self.final_eval,
                     incumbents_by_rule=self._incumbents_by_rule)
        self._save_heuristics()
        self._append_perf_log(gen_id, race_records)

    def _build_table_rows(self, race_out: dict, budget_before: int) -> list:
        rows = []
        for row in race_out.get("table_rows", []):
            mb = float(row["mean_best"])
            if self.mean_opt and self.mean_opt > 0 and math.isfinite(mb):
                gap = (mb - self.mean_opt) / self.mean_opt
            else:
                gap = float("inf")
            rows.append(
                {
                    "inst_idx": int(row["inst_idx"]),
                    "cand_id": row["cand_id"],
                    "score": mb,
                    "gap": float(gap),
                    "n_instances": int(row["n_instances"]),
                    "used_budget": budget_before + int(row["experiments_used"]),
                }
            )
        return rows

    def _run_header(self) -> str:
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        return (
            f"[{self.label}] elitist successive halving EoH TSP-GLS (budget_cap={cap_str}, "
            f"max_generations={gen_str}, T_first={self.t_first}, "
            f"T_each={self.t_each}, N_min={self.pop_size}, "
            f"elitist={self.elitist})"
        )

    # ---- EoH sampling ----------------------------------------------------

    def _log_llm_prompt(self, op, gen, prompt, parents=None, status="ok") -> None:
        parents_shown = [
            {"cand_id": getattr(p, "cand_id", None),
             "desc": (getattr(p, "algorithm", "") or "")[:120]}
            for p in (parents or [])
        ]
        with self._sample_lock:
            record = {
                "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                "call_idx": self._llm_call_idx,
                "gen": gen,
                "op": op,
                "parents": parents_shown,
                "prompt": prompt,
                "status": status,
                # Token usage for this call, read per-thread so threaded sampling
                # (num_threads>1) cannot misattribute another thread's counts.
                # Cost is derived from these in post-processing.
                **_usage_fields(self.sampler),
            }
            self._llm_call_idx += 1
            with open(self._prompt_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _sample(self, op, parents=None, gen=None):
        if op == "i1":
            prompt = EoHPrompt.get_prompt_i1(self.task_desc, self.template_fn)
        elif op == "e1":
            prompt = EoHPrompt.get_prompt_e1(self.task_desc, parents, self.template_fn)
        elif op == "e2":
            prompt = EoHPrompt.get_prompt_e2(self.task_desc, parents, self.template_fn)
        elif op == "m1":
            prompt = EoHPrompt.get_prompt_m1(
                self.task_desc, parents[0], self.template_fn
            )
        elif op == "m2":
            prompt = EoHPrompt.get_prompt_m2(
                self.task_desc, parents[0], self.template_fn
            )
        else:
            raise ValueError(op)
        t0 = time.time()
        try:
            thought, func = self.sampler.get_thought_and_function(prompt)
        except Exception as e:
            dt = time.time() - t0
            self._log_llm_prompt(op, gen, prompt, parents, status="exception")
            print(
                f"    [sample/{op}] exception after {dt:.2f}s: {type(e).__name__}: {e}"
            )
            self._timings["samples"].append(
                {
                    "op": op,
                    "gen": gen,
                    "dt": dt,
                    "status": "exception",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            return None
        dt = time.time() - t0
        if func is None:
            self._log_llm_prompt(op, gen, prompt, parents, status="parse_fail")
            print(f"    [sample/{op}] returned no function ({dt:.2f}s)")
            self._timings["samples"].append(
                {"op": op, "gen": gen, "dt": dt, "status": "parse_fail"}
            )
            return None
        self._log_llm_prompt(op, gen, prompt, parents, status="ok")
        print(f"    [sample/{op}] LLM call ok  ({dt:.2f}s)")
        self._timings["samples"].append(
            {"op": op, "gen": gen, "dt": dt, "status": "ok"}
        )
        func.algorithm = thought or ""
        func.operator = op
        func.parent_ids = [getattr(p, "cand_id", None) for p in (parents or [])] or None
        return func

    def _select_parents(self, pop, k):
        feasible = [r for r in pop if math.isfinite(r.mean_cost) and r.mean_cost < BIG_PENALTY]
        print(
            f"      selecting {k} parents from {len(pop)} candidates "
            f"({len(feasible)} feasible with finite mean_cost)"
        )
        if not feasible:
            return [self.rng.choice(pop) for _ in range(k)] if pop else []
        ranked = sorted(feasible, key=lambda r: r.mean_cost)  # lower cost = better rank
        p = np.array([1.0 / (i + len(ranked)) for i in range(len(ranked))])
        p = p / p.sum()
        idx = self.rng.choice(len(ranked), size=min(k, len(ranked)), replace=False, p=p)
        return [ranked[i] for i in idx]

    def _sample_offspring_seq(self, elites, gen, max_attempts):
        offspring: List[CandidateRecord] = []
        attempts = 0
        while len(offspring) < self.pop_size and attempts < max_attempts:
            attempts += 1
            op = next(self._op_iter)
            k = self.selection_num if op in ("e1", "e2") else 1
            parents = self._select_parents(elites, k)
            fn = self._sample(op, parents=[p.candidate for p in parents], gen=gen)
            if fn is None:
                print(f"    [generation {gen}/{op}] sample fail -- attempt {attempts}")
                continue
            offspring.append(self._materialize(fn, self._sample_idx, gen=gen))
            self._sample_idx += 1
        return offspring

    def _sample_offspring(self, elites, gen):
        """Do not spend LLM calls when a first SH level cannot fit."""
        if self.budget_cap is not None:
            first_level = min(self.sh_min_instances, len(self.instances))
            minimum_needed = self.pop_size * first_level
            remaining = self.budget_cap - self.budget_used
            if remaining < minimum_needed:
                print(
                    f"  [SH budget guard] {remaining} evaluations remain, but "
                    f"the next first level needs at least {minimum_needed}; stopping run"
                )
                return []
        return super()._sample_offspring(elites, gen)

    def _sample_one_threaded(self, elites, gen, attempt_no):
        with self._sample_lock:
            op = next(self._op_iter)
            k = self.selection_num if op in ("e1", "e2") else 1
            parents = self._select_parents(elites, k)
            parent_fns = [p.candidate for p in parents]
        fn = self._sample(op, parents=parent_fns, gen=gen)
        if fn is None:
            print(f"    [generation {gen}/{op}] sample fail -- attempt {attempt_no}")
            return None
        with self._sample_lock:
            idx = self._sample_idx
            self._sample_idx += 1
            rec = self._materialize(fn, idx, gen=gen)
        return rec

    # ---- Heuristic log ---------------------------------------------------

    def _save_heuristics(self) -> None:
        score_by_id = {}
        for gen in self.generations:
            for p in gen.get("performances", []):
                score_by_id[p["cand_id"]] = (
                    float(p.get("score")) if p.get("score") is not None else None
                )
        out = [
            {
                "cand_id": h["cand_id"],
                "gen_id": int(h["gen_id"]),
                "score": score_by_id.get(h["cand_id"]),
                "source": h["source"],
            }
            for h in self._heuristics
        ]
        payload = {"label": self.label, "total_sampled": len(out), "heuristics": out}
        json.dump(payload, open(self.log_dir / "heuristics.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elitist Successive Halving EoH on TSP-GLS.")
    p.add_argument("--pop-size", type=int, default=10)
    p.add_argument("--max-generations", type=int, default=-1,
                   help="Generation cap. Use -1 to disable (budget-cap only).")
    p.add_argument("--ref-max-generations", type=int, default=20,
                   help="Reference generations for budget calc when --budget-cap is auto.")
    p.add_argument("--n-instances", type=int, default=64)
    p.add_argument("--problem-size", type=int, default=100)
    p.add_argument("--selection-num", type=int, default=5)
    p.add_argument("--t-first", type=int, default=5)
    p.add_argument("--t-each", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--test-type", type=str, default="friedman",
                   choices=["friedman", "ttest"])
    p.add_argument("--posthoc-test-type", type=str, default="conover",
                   choices=["conover", "nemenyi"],
                   help="Friedman post-hoc test variant.")
    p.add_argument("--no-elitist", dest="elitist", action="store_false", default=True)
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None,
                   help="Total evaluation budget. None = auto (pop_size x ref_max_generations x n_instances). "
                        "-1 = disabled.")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="Treat the target as deterministic (no per-task seeding).")
    p.add_argument("--eval-timeout", type=float, default=65.0,
                   help="Per-instance wall-clock timeout (seconds). "
                        "Default 65s matches the GLS 60s/instance cap + 5s slack.")
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY,
                   help="Finite penalty cost recorded when a heuristic is killed at "
                        f"--eval-timeout. Default {BIG_PENALTY:g}.")
    p.add_argument("--label", type=str, default="sh/tsp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048,
                   help="Max tokens for LLM generation.")
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--save-pop", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from src/init_pop/eoh_tsp_gls.json "
                        "instead of sampling from the LLM.")
    p.add_argument("--early-stopping-non-elitist", action="store_true", default=False)
    p.add_argument("--deal-with-crashed", type=str, default="rejection",
                   choices=["rejection", "penalty"],
                   help="How the race treats a crashed candidate. "
                        "'rejection' (default): drop immediately. "
                        "'penalty': keep with worst-case cost.")
    p.add_argument("--sh-reduction-factor", type=float, default=1.33,
                   help="Successive Halving reduction factor eta (default 1.33)")
    p.add_argument("--sh-min-instances", type=int, default=5,
                   help="Minimum instances evaluated in round 0 of SH (default 5)")
    p.add_argument("--use-wandb", action="store_true", default=False)
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
        args.log_root,
        args.label,
        dt_stamp,
        args.seed,
        tag=_run_tag(args, _BASH_DEFAULTS, _ABBREV),
    )
    cache_dir = args.cache_root / dt_stamp / str(args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import atexit

    term_log_path = log_dir / "terminal.txt"
    term_log_file = open(term_log_path, "w", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, term_log_file)
    sys.stderr = _Tee(_orig_stderr, term_log_file)

    def _restore_streams():
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        term_log_file.close()

    atexit.register(_restore_streams)
    print(f"terminal output mirrored -> {term_log_path}")

    max_generations = (
        None
        if args.max_generations is not None and args.max_generations < 0
        else args.max_generations
    )
    if args.budget_cap is None:
        budget_cap = args.pop_size * args.ref_max_generations * args.n_instances
    else:
        budget_cap = None if args.budget_cap < 0 else args.budget_cap
    if max_generations is None and budget_cap is None:
        print(
            "error: at least one of --max-generations or --budget-cap must be >= 0",
            file=sys.stderr,
        )
        return 2

    print(
        f"pop_size={args.pop_size}  "
        f"max_generations={'off' if max_generations is None else max_generations}  "
        f"n_instances={args.n_instances}  problem_size={args.problem_size}"
    )
    print(
        f"budget_cap={'off' if budget_cap is None else budget_cap}  "
        f"(T_first={args.t_first}, T_each={args.t_each}, alpha={args.alpha})"
    )

    if args.llm_backend == "ollama":
        client = OllamaClient(
            host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout
        )
        cached = CachedLLM(client, cache_dir=cache_dir)
        llm = OllamaLLM4AD(cached)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir)
        llm = MistralLLM4AD(cached)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
        cached = CachedLLM(client, cache_dir=cache_dir)
        llm = vLLMLLM4AD(cached)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = GoogleClient(
            timeout=args.llm_timeout, x_title=args.label, model=args.llm_model
        )
        cached = CachedLLM(client, cache_dir=cache_dir)
        llm = GoogleLLM4AD(cached)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(
            timeout=args.llm_timeout, x_title=args.label, model=args.llm_model
        )
        cached = CachedLLM(client, cache_dir=cache_dir)
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

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args, _BASH_DEFAULTS, _ABBREV)}"
    wandb_logger = make_wandb_logger(
        enabled=args.use_wandb, project="llm4ad", name=run_name, config=args_dict
    )

    from utils.manuscript_log import write_run_meta
    write_run_meta(log_dir, {
        "run_id": run_name,
        "framework": "EoH", "domain": "TSP-GLS",
        "policy": "Elitist-SH",
        "seed": args.seed,
        "M": args.pop_size, "N": args.n_instances, "K": args.n_instances,
        "alpha": args.alpha, "test_type": args.test_type,
        "posthoc": args.posthoc_test_type,
        "T_first": args.t_first, "T_each": args.t_each,
        "elimit": args.elitist_limit,
        "refill": bool(getattr(args, "save_pop", False)),
        "incumbent_rule": "both",
        "B": budget_cap,
        "llm_model": cached.client.model,
    })

    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluation = TSP_GLS_2O_Evaluation_wo_Time()
    evaluation.n_instance = args.n_instances
    evaluation.problem_size = args.problem_size
    if (args.n_instances, args.problem_size) != (16, 100):
        evaluation._datasets = GetData(args.n_instances, args.problem_size).generate_instances()
    random.seed(args.seed)
    np.random.seed(args.seed)

    instances = list(evaluation._datasets)
    for i, inst in enumerate(instances):
        inst._id = i

    print(f"  instances: n={args.n_instances}  problem_size={args.problem_size}  "
          f"eval_timeout={args.eval_timeout}s/instance")

    print("Solving per-instance optima with Concorde...")
    opt_per_inst = [_opt_cost(inst) for inst in instances]
    mean_opt = float(np.mean(opt_per_inst))
    opt_by_idx = {i: v for i, v in enumerate(opt_per_inst)}
    print(f"  per-instance opt: min={min(opt_per_inst):.4f}  "
          f"max={max(opt_per_inst):.4f}  mean={mean_opt:.4f}")

    _racer = SHEoH(
        evaluation=evaluation,
        instances=instances,
        label=args.label,
        log_dir=log_dir,
        llm=llm,
        pop_size=args.pop_size,
        max_generations=max_generations,
        selection_num=args.selection_num,
        budget_cap=budget_cap,
        t_first=args.t_first,
        t_each=args.t_each,
        sh_reduction_factor=args.sh_reduction_factor,
        sh_min_instances=args.sh_min_instances,
        alpha=args.alpha,
        seed=args.seed,
        opt_by_idx=opt_by_idx,
        mean_opt=mean_opt,
        num_threads=args.num_threads,
        num_cores=args.num_cores,
        eval_timeout=args.eval_timeout,
        timeout_cost=args.timeout_cost,
        test_type=args.test_type, posthoc_test_type=args.posthoc_test_type,
        elitist=args.elitist,
        elitist_new_instances=args.elitist_new_instances,
        elitist_limit=args.elitist_limit,
        save_pop=args.save_pop,
        early_stopping_non_elitist=args.early_stopping_non_elitist,
        deal_with_crashed=args.deal_with_crashed,
        deterministic=args.deterministic,
        wandb_logger=wandb_logger,
    )
    if args.fix_init_pop:
        from utils.fixed_init_pop import load_fixed_initial_population
        _fp = ROOT / "src" / "init_pop" / "eoh_tsp_gls.json"
        _racer._fixed_init_sources = [
            h["source"] for h in load_fixed_initial_population(_fp, _racer.pop_size)]
        print(f"  [fixed-init-pop] loaded {len(_racer._fixed_init_sources)} "
              f"heuristics from {_fp}", flush=True)
    _racer.run()

    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
