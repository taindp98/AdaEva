"""Successive Halving LLaMEA on BBOB.

Same LLaMEA ES loop as ``racing/llamea_bbob.py`` (the native
``reprod/llamea_bbob.LLaMEA_BBOB`` skeleton + irace-style instance pool +
population-context mutation prompts + per-candidate ``Solution`` objects) but
replaces the Friedman/Nemenyi F-race (``_race``) with a Successive Halving
sub-race — the geometric instance-schedule + candidate-pruning pattern from
``sh/eoh_obp.py``.

In Successive Halving, every (heuristic, instance) pair runs exactly once (no
repetition / seeded stochastic replay), so the runner always operates in
DETERMINISTIC mode — the instance pool is the bare 72 BBOB base instances
(24 fids × 3 iids), each evaluated once per candidate with no per-task seed.

Key overrides from ``RacingLLaMEA``:
    ``_race``           →  SH sub-race (geometric schedule, halving prune)
    ``run``             →  print header, SH-specific budget/gen reporting

Everything else (prompt, sampling, materialize, _pre_evaluate,
_rescore_elites_shared, _final_eval, _record_generation, _log_manuscript,
instance pool management) is inherited unchanged.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures as _cf
import datetime as _dt
import json
import math
import os
import pathlib
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLaMEA"))

from ioh import get_problem, logger as ioh_logger

from racing.llamea_bbob import (
    RacingLLaMEA, LLaMEA_LLM_Adapter, _auto_eval_timeout,
    score_bbob_config, _score_bbob_inst, _unwrap, SeededInstance,
    BIG_PENALTY, REJECT_COST, _SEED_MAX,
    _FIDS, _IIDS, _N_UNIQUE_INSTANCES, _HOMO_FID, _HOMO_DIMS,
    _AOC_LOWER, _AOC_UPPER,
)
from sh.eoh_obp import _sh_schedule
from utils import ConfigAS, make_wandb_logger
from utils.config import RaceState
from utils.race import _eval_one_instance_elitist
from utils.llm import CachedLLM, OpenRouterClient, OllamaClient, MistralClient, vLLMClient, GoogleClient
from utils.logger import make_log_dir, _Tee


# --------------------------------------------------------------------------- #
# SHLLaMEA — Successive Halving on BBOB via the LLaMEA ES loop
# --------------------------------------------------------------------------- #

def _mean_cost(c: ConfigAS) -> float:
    """Mean cost over all evaluated instances (robust to empty)."""
    vals = list(c.costs_by_inst.values())
    return float(np.mean(vals)) if vals else float("inf")


def _round_mean_cost(cfg: ConfigAS, max_idx: int) -> float:
    """Mean cost over instances 1..max_idx."""
    vals = [cfg.costs_by_inst[k] for k in range(1, max_idx + 1)
            if k in cfg.costs_by_inst]
    return float(np.mean(vals)) if vals else float("inf")


class SHLLaMEA(RacingLLaMEA):
    """``RacingLLaMEA`` with Successive Halving as the sub-race strategy.

    The SH sub-race evaluates candidates on a geometrically increasing number
    of instances ``[N_min, ..., N_max]`` and prunes the weakest ``⌊K/η⌋`` at
    each level.  Elites carried over from the previous generation already hold
    evaluations on all instances from prior races; those are reused at zero
    budget cost (0-cost cache).

    Cost = ``-AOCC`` (lower = better), matching ``score_bbob_config``.
    """

    def __init__(self, *, sh_reduction_factor: float = 1.25,
                 sh_min_instances: int = 5, **kwargs):
        self.sh_reduction_factor = max(1.1, float(sh_reduction_factor))
        self.sh_min_instances = max(1, int(sh_min_instances))
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # SH sub-race (overrides RacingLLaMEA._race)
    # ------------------------------------------------------------------ #

    def _race(self, cfgs: List[ConfigAS], next_instance: int) -> dict:
        """Successive Halving sub-race over ``cfgs``.

        Parameters
        ----------
        cfgs : list[ConfigAS]
            The combined pool (elites + offspring) entering the race.
        next_instance : int
            1-based index of the next unused instance (inherited irace contract).

        Returns
        -------
        dict
            Same contract as ``RacingLLaMEA._race``: keys ``survivors``,
            ``all_records_sorted``, ``next_instance``, ``break_reason``,
            ``experiments_used``, ``cpu_seconds``, ``seen_instances``,
            ``step_trace``.
        """
        t0 = time.time()

        K0 = len(cfgs)
        M = self.n_parents
        eta = self.sh_reduction_factor
        max_inst_available = len(self.instances)
        start_n = min(self.sh_min_instances, max_inst_available)
        schedule = _sh_schedule(K0, M, eta, start_n, max_inst_available)
        b_cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        budget_remaining = (self.budget_cap - self.budget_used) if self.budget_cap is not None else float("inf")

        # Pre-eval cleanup: reset carried-over histories so the SH sub-race
        # starts from a clean slate (the elitist 0-cost cache still holds via
        # ``costs_by_inst`` — only ``mean_cost`` is recalculated from scratch).
        if getattr(self, "_needs_pre_eval_cleanup", False):
            self._needs_pre_eval_cleanup = False
            for c in cfgs:
                c.reset_history()
            print(f"  [pre-eval cleanup] cfg histories and mean_costs reset "
                  f"for {len(cfgs)} candidates before first race", flush=True)

        # Tag each config's starting instance set (for 0-cost cache reporting).
        for c in cfgs:
            c._race_init_costs = set(c.costs_by_inst.keys())
        active_cfgs = list(cfgs)
        entry_elites = [c for c in cfgs if c.costs_by_inst and c.alive]
        experiments_used = 0
        seen_inst_indices: list[int] = []
        step_trace: list[dict] = []
        state = RaceState(configs=cfgs, instances=[], instances_log=self.instances,
                          metric="mean_cost")
        runner = self._make_runner()
        completed_rounds = 0
        budget_stopped = False

        print(f"\n# SH Sub-Race -- {K0} candidates ({M} elites target, "
              f"eta={eta:g}, N_min={start_n}, N_max={max_inst_available})")
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
                        pool_runner=score_bbob_config,
                        eval_timeout=self._eval_timeout,
                        timeout_cost=self._timeout_cost,
                        crash_penalty=(None if self.deal_with_crashed == "rejection"
                                       else self._crash_penalty),
                        pool_recreate=self._recreate_pool,
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
                })
                # Drop crashed/eliminated candidates NOW so they are not re-evaluated
                # (and re-charged) on the round's remaining instances — mirrors the
                # per-instance alive re-filter in the SH EoH runners.
                active_cfgs = [c for c in active_cfgs if c.alive]

            experiments_used += fresh_evals
            self.budget_used += fresh_evals
            budget_remaining = (self.budget_cap - self.budget_used) if self.budget_cap is not None else float("inf")
            completed_rounds += 1

            # Rank candidates by mean cost over all instances evaluated in this round.
            active_cfgs.sort(key=lambda c: _round_mean_cost(c, target_n_insts))

            print(f"  [SH Round {round_idx+1}/{len(schedule)}] candidates={len(active_cfgs)} | "
                  f"instances={target_n_insts} | fresh_evals={fresh_evals}, "
                  f"0-cost cached={cached_evals} | total budget={self.budget_used}/{b_cap_str}")
            if cached_evals > 0:
                print(f"    [elitist cache] 0-cost cache hit on {cached_evals} "
                      f"evaluation(s) for carried-over elites (0 evaluation budget spent)")

            if round_idx < len(schedule) - 1:
                keep_k = max(M, int(math.floor(len(active_cfgs) / eta)))
                if keep_k < len(active_cfgs):
                    dropped = active_cfgs[keep_k:]
                    for d in dropped:
                        d.alive = False
                    active_cfgs = active_cfgs[:keep_k]
                    elim_ids = [d.id for d in dropped]
                    print(f"    [halving] eliminated {len(dropped)} candidates "
                          f"({elim_ids}) -> {len(active_cfgs)} remaining")
                    next_target = schedule[round_idx + 1] if round_idx < len(schedule) - 1 else target_n_insts
                    print(f"    Surviving candidates carried to SH Round {round_idx + 2} "
                          f"(target instances: {next_target}):")
                    self._print_table(active_cfgs, target_n_insts)

        if completed_rounds == 0:
            active_cfgs = entry_elites

        if len(active_cfgs) > M:
            dropped = active_cfgs[M:]
            for d in dropped:
                d.alive = False
            active_cfgs = active_cfgs[:M]

        # Finalize: flow race costs back to Solution fitness + feedback
        # so construct_prompt's population summary is meaningful.
        alive_ids = {c.id for c in active_cfgs if c.alive}
        seen_list = sorted(seen_inst_indices)

        for c in cfgs:
            mc = _mean_cost(c)

            sol = self._sol_by_id.get(c.id)
            if sol is None:
                continue
            if c.id in alive_ids and math.isfinite(mc):
                if self.prompt_mode == "original":
                    _aoccs = [-v for v in c.costs_by_inst.values()]
                    _std = float(np.std(_aoccs)) if _aoccs else 0.0
                    sol.set_scores(
                        -mc,
                        f"The algorithm {sol.name} got an average Area over the convergence "
                        f"curve (AOCC, 1.0 is the best) score of {-mc:0.4f} with standard "
                        f"deviation {_std:0.4f}.",
                    )
                else:
                    sol.set_scores(
                        -mc,    # fitness = AOCC = -cost
                        feedback=(f"The algorithm {sol.name} scored AOCC {-mc:0.4f} in the SH race "
                                  f"(evaluated over {len(c.costs_by_inst)} instances)."),
                    )
            elif c.id in alive_ids:
                err = self._capture_error(c)
                if self.prompt_mode == "original":
                    sol.set_scores(self._worst,
                                   "Algorithm failed to evaluate on BBOB instances.",
                                   error=err)
                else:
                    sol.set_scores(
                        self._worst,
                        feedback=f"The algorithm {sol.name} failed / was rejected during SH evaluation.",
                        error=err,
                    )
            else:
                sol.set_scores(
                    self._worst,
                    feedback=(f"The algorithm {sol.name} was ELIMINATED by the SH race "
                              f"(kept only for offspring diversity)."),
                )

        # Rank survivors on the instances THIS race actually covered (``seen_list``),
        # NOT each candidate's full/own coverage: elites carry cached instances from
        # earlier generations (coverage is non-uniform — 5/12/30/72), so the module
        # ``_mean_cost`` over disjoint sets is non-comparable and would let a
        # lightly-evaluated offspring (few easy instances) outrank a deeply-evaluated
        # elite. Mirrors the SH EoH runners' ``seen_list`` ranking.
        _seen_set = set(seen_list)

        def _seen_mean(c: ConfigAS) -> float:
            vals = [c.costs_by_inst[k] for k in _seen_set if k in c.costs_by_inst]
            return float(np.mean(vals)) if vals else float("inf")

        survivor_cfgs = sorted(
            [c for c in cfgs if c.id in alive_ids],
            key=_seen_mean,
        )
        eliminated_cfgs = sorted(
            [c for c in cfgs if c.id not in alive_ids],
            key=_seen_mean,
        )

        dt_race = time.time() - t0
        break_reason = ("SH budget cap reached before a complete round"
                        if budget_stopped else "SH rounds complete")
        best_cand = survivor_cfgs[0] if survivor_cfgs else None
        best_cost = _seen_mean(best_cand) if best_cand else float("inf")
        best_str = (f"{best_cand.id} (mean_cost={best_cost:.4f}, "
                    f"AOCC={-best_cost:.4f})" if best_cand else "None")
        print(f"  [SH Sub-Race Summary] {break_reason} | "
              f"survivors={len(survivor_cfgs)}/{K0} | best={best_str} | "
              f"race_wall={dt_race:.2f}s")
        final_target = seen_list[-1] if seen_list else 0
        print(f"  Final Surviving Elites from SH Sub-Race (target instances: {final_target}):")
        self._print_table(survivor_cfgs, final_target)

        self._append_perf_log(cfgs, seen_list)
        # WARNING for any (heuristic, instance) that hit the eval timeout.
        for c in cfgs:
            for inst_idx in sorted(getattr(c, "timed_out_insts", ())):
                self._warn_timeout(c.id, inst_idx, phase="sh-race")

        return {
            "survivors": survivor_cfgs,
            "all_records_sorted": survivor_cfgs + eliminated_cfgs,
            "next_instance": next_instance,
            "break_reason": break_reason,
            "step_trace": step_trace,
            "experiments_used": experiments_used,
            "cpu_seconds": 0.0,
            "seen_instances": seen_list,
        }

    def _make_runner(self):
        """Build the callable ``runner(params, instance)`` that the
        ``_eval_one_instance_elitist`` helper dispatches for in-process
        (non-pool) evaluation. Returns ``(cost, cpu_seconds)``."""
        def runner(params, instance):
            cpu0 = time.process_time()
            t0 = time.perf_counter()
            try:
                cost = _score_bbob_inst(params, instance)
            except Exception:
                cost = REJECT_COST
            cpu = time.process_time() - cpu0
            return cost, cpu
        return runner

    def _print_table(self, cfgs: list, target_n_insts: int) -> None:
        """Print a compact table of candidates with their mean cost / AOCC."""
        print("  +------+----------------------+-------------+--------+------------------------------+")
        print("  | Rank | Candidate ID         | Mean AOCC   | Eval/N | Elitist 0-Cost Cache         |")
        print("  +------+----------------------+-------------+--------+------------------------------+")
        for rank_idx, c in enumerate(cfgs, 1):
            mc = _mean_cost(c)
            aocc_str = f"{-mc:.4f}" if math.isfinite(mc) else "  inf"
            n_eval = len([k for k in c.costs_by_inst if k <= target_n_insts])
            eval_str = f"{n_eval}/{target_n_insts}"
            prior_cached = len([k for k in c.costs_by_inst
                                if k <= target_n_insts and k in getattr(c, "_race_init_costs", set())])
            cid = str(c.id)
            if len(cid) > 20:
                cid = cid[:17] + "..."
            total_lifetime = len(c.costs_by_inst)
            cache_info = (f"{prior_cached} cached ({total_lifetime} lifetime)"
                          if prior_cached > 0 else "0 cached (new offspring)")
            print(f"  | {rank_idx:4d} | {cid:20s} | {aocc_str:>11s} | {eval_str:^6s} | {cache_info:28s} |")
        print("  +------+----------------------+-------------+--------+------------------------------+")

    # ---- run() override for SH-specific header --------------------------

    def run(self):
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        print(f"[{self.label}] elitist successive halving LLaMEA BBOB "
              f"(budget_cap={cap_str}, max_generations={gen_str}, "
              f"T_first={self.t_first}, T_each={self.t_each}, mu={self.n_parents}, "
              f"lambda={self.n_offspring}, eta={self.sh_reduction_factor:g}, "
              f"N_min={self.sh_min_instances})", flush=True)
        print(f"  [elite-protection] active: survivor ranking tiers by instance "
              f"count (irace overall_ranks) — under-sampled offspring cannot "
              f"displace better-evaluated elites.", flush=True)
        print("# Markers:")
        print("     x No test is performed.")
        print("     - The test is performed and some configurations are discarded.")
        print("     = The test is performed but no configuration is discarded.")
        print("     ! The test is performed and configurations could be discarded "
              "but elite configurations are preserved.")
        print("     . Alive configurations were already evaluated on this instance "
              "and nothing is discarded.")
        print()

        init_sols = self._initialize_population()
        init_cfgs = [self._materialize(s, gen=0) for s in init_sols]
        self._elite_cfgs = init_cfgs[: self.n_parents]
        self._update_best(init_cfgs)
        next_instance = 1
        self.generation = 0
        print(f"  gen 00: sampled {len(init_cfgs)} initial parents; "
              f"used_budget={self.budget_used}", flush=True)

        # Pre-evaluate initial parents so offspring sampling + prompt have scores.
        if self.t_first > 0:
            self._pre_evaluate(init_cfgs)

        while not self._stop():
            gen_id = self.generation + 1
            elite_sols = [self._sol_by_id[c.id] for c in self._elite_cfgs]
            self.population = elite_sols
            offspring_sols = self._sample_offspring(elite_sols, gen=gen_id)
            offspring_cfgs = [self._materialize(s, gen=gen_id) for s in offspring_sols]
            if not offspring_cfgs:
                print(f"  gen {gen_id:02d}: 0 offspring — stopping", flush=True)
                break

            combined = (self._elite_cfgs + offspring_cfgs) if self.llamea_elitism else list(offspring_cfgs)
            t_race = time.time()

            print(f"\n################################################################################")
            print(f"# Generation {gen_id:02d} race — {len(combined)} configs "
                  f"({len(self._elite_cfgs)} elites + {len(offspring_cfgs)} offspring)")
            print(f"################################################################################")

            out = self._race(combined, next_instance)
            next_instance = out["next_instance"]
            # Budget-atomic SH won't start a round it cannot finish, so once the
            # remaining budget is smaller than even the first round's fresh-eval cost
            # the race spends nothing and self.budget_used freezes below budget_cap —
            # _stop() (which fires only at >= budget_cap) would then loop forever,
            # re-sampling offspring (LLM calls) that never get evaluated. Stop here so
            # run() proceeds to the full-suite _final_eval instead.
            if out.get("experiments_used", 0) == 0 and self.budget_cap is not None:
                print(f"  [budget exhausted] gen {gen_id} race made no fresh evaluations "
                      f"(remaining < one SH round); stopping search", flush=True)
                break
            survivors = out["survivors"]
            if self.save_pop:
                self._elite_cfgs = (out["all_records_sorted"][: self.n_parents]
                                    or self._elite_cfgs[: self.n_parents])
            else:
                self._elite_cfgs = survivors[: self.n_parents] or self._elite_cfgs[: self.n_parents]
            self._rescore_elites_shared()
            self.generation = gen_id
            self._evo_gens += 1
            self._record_generation(gen_id, offspring_cfgs)
            self._log_manuscript(gen_id, combined, offspring_cfgs, out, t_race)
            print(f"  gen {gen_id:02d}/{gen_str}: pool={len(combined)} "
                  f"({len(self._elite_cfgs)} elites+{len(offspring_cfgs)} offspring), "
                  f"survivors={len(out['survivors'])}, "
                  f"elites_kept={len(self._elite_cfgs)}, "
                  f"budget={self.budget_used}/{cap_str}, "
                  f"race_wall={time.time() - t_race:.1f}s "
                  f"({out['break_reason']})", flush=True)

        self._shutdown_pool()
        print(f"[{self.label}] done in {self._evo_gens} generations, "
              f"final budget={self.budget_used}, "
              f"best AOCC={self._best_aocc:.4f}", flush=True)
        valid = self._final_eval()
        for rule in ("mean_rank", "mean_cost"):
            try:
                self._mlog.finalize_reliability(
                    getattr(self, f"incumbents_{rule}"),
                    (valid or {}).get(rule), rule=rule)
            except Exception as e:
                print(f"  [manuscript-log/{rule}] WARN: reliability finalize failed: {e}", flush=True)
        return self.best


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_BASH_DEFAULTS: dict = {
    "fix_init_pop": False,
    "llm_model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "llm_backend": "vllm",
    "temperature": 0.8,
    "n_parents": 10,
    "n_offspring": 10,
    "max_generations": -1,
    "ref_max_generations": 20,
    "dim": 5,
    "budget_factor": 2000,
    "n_reps": 1,
    "t_first": 5,
    "t_each": 1,
    "sh_reduction_factor": 1.25,
    "sh_min_instances": 5,
    "alpha": 0.05,
    "test_type": "friedman",
    "posthoc_test_type": "conover",
    "race_elitist": True,
    "llamea_elitism": True,
    "save_pop": False,
    "deal_with_crashed": "rejection",
    "crash_penalty": 1e6,
    "elitist_new_instances": 1,
    "elitist_limit": 12,
    "eval_timeout": -1.0,
    "parent_selection": "random",
    "tournament_size": 3,
    "num_threads": 4,
    "num_cores": 4,
    "prompt_mode": "partial_eval",
    "instance_pool_mode": "hetero",
}
_ABBREV: dict = {
    "fix_init_pop": "fixinit",
    "llm_model": "model", "llm_backend": "llm", "temperature": "temp",
    "n_parents": "mu", "n_offspring": "lam", "max_generations": "mg",
    "ref_max_generations": "rmg", "dim": "d", "budget_factor": "bf", "n_reps": "nrep",
    "t_first": "tf", "t_each": "te",
    "sh_reduction_factor": "srf", "sh_min_instances": "smi",
    "alpha": "a", "test_type": "tt",
    "posthoc_test_type": "phtt",
    "race_elitist": "relit", "llamea_elitism": "eselit", "save_pop": "savpop",
    "deal_with_crashed": "dwc",
    "crash_penalty": "cp",
    "elitist_new_instances": "tni", "elitist_limit": "elimit", "eval_timeout": "et",
    "parent_selection": "ps",
    "tournament_size": "ts", "num_threads": "nt", "num_cores": "nc",
    "prompt_mode": "pm",
    "instance_pool_mode": "ipm",
}


def _run_tag(args: argparse.Namespace) -> str:
    overrides = []
    for key, default_val in _BASH_DEFAULTS.items():
        actual = getattr(args, key, None)
        if actual == default_val:
            continue
        abbr = _ABBREV.get(key, key)
        if isinstance(actual, bool):
            overrides.append(abbr if actual else f"no{abbr}")
        elif isinstance(actual, float):
            overrides.append(f"{abbr}{actual:g}")
        else:
            val_str = str(actual).replace("/", "_").replace(" ", "")
            overrides.append(f"{abbr}{val_str}")
    return "_".join(overrides) if overrides else "default"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Elitist Successive Halving LLaMEA on BBOB.")
    p.add_argument("--n-parents", type=int, default=10,
                   help="mu: population size (parents per generation).")
    p.add_argument("--n-offspring", type=int, default=10,
                   help="lambda: offspring per generation.")
    p.add_argument("--max-generations", type=int, default=-1,
                   help="Generation cap. Use -1 to disable (budget-cap only).")
    p.add_argument("--ref-max-generations", type=int, default=20,
                   help="Reference generations for budget calc when --budget-cap is auto.")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--budget-factor", type=int, default=2000,
                   help="Per-instance func-eval budget = budget_factor * dim.")
    p.add_argument("--n-reps", type=int, default=1,
                   help="Per-instance repetitions (default 1 for SH deterministic mode).")
    p.add_argument("--t-first", type=int, default=5,
                   help="Pre-eval: score initial mu parents on t_first sampled instances.")
    p.add_argument("--t-each", type=int, default=1)
    p.add_argument("--sh-reduction-factor", type=float, default=1.25,
                   help="Successive Halving reduction factor eta (default 1.25)")
    p.add_argument("--sh-min-instances", type=int, default=5,
                   help="Minimum instances evaluated in round 0 of SH (default 5)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--test-type", type=str, default="friedman",
                   choices=["friedman", "ttest"])
    p.add_argument("--posthoc-test-type", type=str, default="conover",
                   choices=["conover", "nemenyi"])
    p.add_argument("--no-race-elitist", dest="race_elitist", action="store_false", default=True)
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=12)
    p.add_argument("--no-llamea-elitism", dest="llamea_elitism", action="store_false", default=True,
                   help="Disable (mu+lambda) -> (mu,lambda).")
    p.add_argument("--save-pop", action="store_true", default=False)
    p.add_argument("--deal-with-crashed", type=str, default="rejection",
                   choices=["rejection", "penalty"])
    p.add_argument("--crash-penalty", type=float, default=BIG_PENALTY)
    p.add_argument("--eval-timeout", type=float, default=-1.0,
                   help="Per-(candidate, instance) wall-clock cap (s). "
                        "-1 (default) = AUTO-SCALE; 0 = disabled; >0 = fixed seconds.")
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY)
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="SH runs in deterministic mode by default (each "
                        "(heuristic, instance) pair evaluated exactly once).")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--parent-selection", type=str, default="random",
                   choices=["random", "tournament", "roulette"])
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--instance-pool-mode", type=str, default="hetero",
                   choices=["hetero", "homo"],
                   help="hetero: 24 fids × 3 iids = 72 instances (default). "
                        "homo: single fid × 3 dims × 3 iids = 9 instances.")
    p.add_argument("--prompt-mode", type=str, default="partial_eval",
                   choices=["partial_eval", "original"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None)
    p.add_argument("--label", type=str, default="sh_llamea_bbob")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048)
    p.add_argument("--llm-backend", type=str, default="vllm",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--llm-model", type=str,
                   default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from src/init_pop/llamea_24_bbob.json.")
    return p.parse_args(argv)


def _build_llm(args, cache_dir, log_dir):
    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "mistral":
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
        client = GoogleClient(timeout=args.llm_timeout, model=args.llm_model)
    else:
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
    cached = CachedLLM(client, cache_dir=cache_dir,
                       prompt_log=log_dir / "llm_prompts.jsonl")
    return LLaMEA_LLM_Adapter(cached, model_name=cached.client.model, temperature=args.temperature)


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

    term_log_path = log_dir / "terminal.txt"
    term_log_file = open(term_log_path, "w", buffering=1)
    _orig_out, _orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_out, term_log_file)
    sys.stderr = _Tee(_orig_err, term_log_file)

    def _restore():
        sys.stdout, sys.stderr = _orig_out, _orig_err
        term_log_file.close()
    atexit.register(_restore)
    print(f"terminal output mirrored -> {term_log_path}")

    max_generations = (None if args.max_generations is not None and args.max_generations < 0
                       else args.max_generations)
    n_unique = (len(_HOMO_DIMS) * len(_IIDS)) if args.instance_pool_mode == "homo" else _N_UNIQUE_INSTANCES
    if args.budget_cap is None:
        budget_cap = args.n_parents * args.ref_max_generations * n_unique
    else:
        budget_cap = None if args.budget_cap < 0 else args.budget_cap
    if max_generations is None and budget_cap is None:
        print("error: at least one of --max-generations or --budget-cap must be >= 0",
              file=sys.stderr)
        return 2

    # Resolve --eval-timeout: -1 => auto-scale with budget; 0 => disabled; >0 => fixed.
    if args.eval_timeout is not None and args.eval_timeout < 0:
        eval_timeout = _auto_eval_timeout(args.budget_factor, args.dim)
        print(f"  [eval-timeout] auto = 60 + budget_factor({args.budget_factor}) x "
              f"dim({args.dim})/100 = {eval_timeout:.0f}s per (config, instance)")
    else:
        eval_timeout = args.eval_timeout

    llm = _build_llm(args, cache_dir, log_dir)

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()}
    args_dict["llm_model"] = llm._cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args)}"
    from utils.manuscript_log import write_run_meta as _wrm
    _wrm(log_dir, {
        "run_id": run_name, "framework": "LLaMEA", "domain": "BBOB",
        "policy": "Elitist-SH",
        "seed": args.seed,
        "M": args.n_parents, "N": n_unique, "K": n_unique,
        "alpha": args.alpha, "test_type": args.test_type,
        "posthoc": args.posthoc_test_type,
        "T_first": args.t_first, "T_each": args.t_each,
        "elimit": args.elitist_limit,
        "refill": bool(args.save_pop),
        "incumbent_rule": "both",
        "B": budget_cap, "llm_model": args_dict.get("llm_model"),
    })
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    print(f"instance_pool_mode={args.instance_pool_mode}  "
          f"pop: mu={args.n_parents} lambda={args.n_offspring}  dim={args.dim}  "
          f"budget_cap={'off' if budget_cap is None else budget_cap}  "
          f"max_generations={'off' if max_generations is None else max_generations}")

    fixed_init_population = None
    if args.fix_init_pop:
        from utils.fixed_init_pop import load_fixed_initial_population
        _fp = ROOT / "src" / "init_pop" / "llamea_24_bbob.json"
        fixed_init_population = load_fixed_initial_population(_fp, args.n_parents)
        print(f"  [fixed-init-pop] loaded {len(fixed_init_population)} heuristics "
              f"from {_fp}", flush=True)

    SHLLaMEA(
        llm=llm, log_dir=log_dir, label=args.label,
        n_parents=args.n_parents, n_offspring=args.n_offspring,
        dim=args.dim, budget_factor=args.budget_factor, n_reps=args.n_reps,
        seed=args.seed,
        max_generations=max_generations, budget_cap=budget_cap,
        t_first=args.t_first, t_each=args.t_each,
        sh_reduction_factor=args.sh_reduction_factor,
        sh_min_instances=args.sh_min_instances,
        alpha=args.alpha, test_type=args.test_type,
        posthoc_test_type=args.posthoc_test_type,
        race_elitist=args.race_elitist,
        elitist_new_instances=args.elitist_new_instances,
        elitist_limit=args.elitist_limit,
        save_pop=args.save_pop,
        deal_with_crashed=args.deal_with_crashed,
        crash_penalty=args.crash_penalty,
        llamea_elitism=args.llamea_elitism,
        eval_timeout=eval_timeout,
        timeout_cost=args.timeout_cost,
        deterministic=args.deterministic,
        parent_selection=args.parent_selection,
        tournament_size=args.tournament_size,
        num_threads=args.num_threads, num_cores=args.num_cores,
        prompt_mode=args.prompt_mode,
        instance_pool_mode=args.instance_pool_mode,
        fixed_init_population=fixed_init_population,
        wandb_logger=wandb_logger,
    ).run()

    wandb_logger.finish()
    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
