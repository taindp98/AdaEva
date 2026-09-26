"""Racing EoH on the LLM4AD Online Bin Packing task.

Same outer LLM call surface as ``reprod_eoh_obp.py`` (LLM4AD's ``EoHPrompt``
+ ``EoHSampler``, ``OBPEvaluation``) but replaces full-pool evaluation +
mu+lambda truncation with a ``utils``-based Friedman/Nemenyi race over the
union of elites and offspring.

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
import time
import yaml
from dataclasses import dataclass
from typing import Any, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, ConfigAS, OpenRouterClient, OllamaClient,
                   MistralClient, vLLMClient, make_wandb_logger)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD
from utils.obp_utils import _obp_lower_bound
from utils.logger import make_log_dir
from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.eoh.prompt import EoHPrompt
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation

from racing.base import RacingBase, CandidateRecord, _Tee, save_run_log, _run_tag


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



# BIG_PENALTY is the finite default cost for a TIMEOUT (a slow-but-valid
# heuristic killed at eval_timeout). Kept finite so the config is penalised
# (and then disqualified via the timeout marker), not treated as a crash.
BIG_PENALTY = 1e6

# REJECT_COST marks an INVALID result — a compile/runtime crash inside the
# generated heuristic. irace uses Inf for this (packages/irace/R/race.R:493,
# 1064-1073): a rejected configuration is eliminated immediately and loses
# elite protection. Keeping crashes (Inf) distinct from timeouts (finite
# BIG_PENALTY) is what lets `utils.race._reject_invalid` reject a broken
# heuristic while a slow one is handled by the timeout path.
REJECT_COST = float("inf")

# Upper bound for a per-task random seed = INT32_MAX (max signed 32-bit int),
# mirroring irace's task seeds up to `.Machine$integer.max` (2147483647) via
# `sample.int` (packages/irace/R/irace.R:562, race_state.R:65).
_SEED_MAX = 2 ** 31 - 1  # 2_147_483_647 (INT32_MAX)


@dataclass(frozen=True)


class SeededInstance:
    """One irace-style race task: a (base OBP instance, seed) pair.

    irace identifies a task by an (instanceID, seed) pair; the seed is fixed per
    task and SHARED across every candidate evaluated on it, so the Friedman/t-test
    compares configs under identical random conditions. Here `instance` is the
    shared, read-only OBP instance dict; `seed` is fixed for this (base instance,
    repetition) and applied to np.random / random before the priority function
    runs, so a non-deterministic LLM heuristic is reproducible and comparable.

    `base_idx` / `rep` are bookkeeping (which base instance, which repetition),
    not used for scoring.
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
    "pop_size": 20,
    "max_generations": -1,
    "ref_max_generations": 20,
    "n_instances": 25,
    "n_items": 5000,
    "capacity": 100,
    "selection_num": 5,
    "timeout_seconds": 30,
    "t_first": 5,
    "t_each": 1,
    "elitist_new_instances": 1,
    "alpha": 0.05,
    "test_type": "friedman",
    "posthoc_test_type": "conover",
    "elitist_limit": 2,
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
    "n_items": "nit",
    "capacity": "cap",
    "selection_num": "sn",
    "timeout_seconds": "to",
    "t_first": "tf",
    "t_each": "te",
    "elitist_new_instances": "tni",
    "alpha": "a",
    "test_type": "tt",
    "posthoc_test_type": "phtt",
    "elitist_limit": "elimit",
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
# OBP scoring
# --------------------------------------------------------------------------- #


def _score_obp_inst(priority: callable, inst, _n_items: int) -> float:
    # `inst` may be a SeededInstance (carrying the per-task seed) or a bare OBP
    # instance dict (deterministic mode, seed=None). When seeded, np.random /
    # random are seeded before the priority function runs so a non-deterministic
    # LLM heuristic is reproducible for a given (instance, seed) task.
    inst, seed = _unwrap(inst)
    try:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        capacity = int(inst["capacity"])
        items = inst["items"]
        bins = np.array([capacity] * len(items), dtype=np.int64)
        for item in items:
            valid = np.nonzero((bins - item) >= 0)[0]
            priorities = priority(item, bins[valid])
            best = int(valid[int(np.argmax(priorities))])
            bins[best] -= int(item)
        return float((bins != capacity).sum())
    except Exception:
        # A crash inside the generated heuristic is an INVALID result -> Inf,
        # so the race rejects the configuration (mirrors irace's Inf-rejection;
        # see utils.race._reject_invalid). A timeout is handled separately and
        # records the finite penalty.
        return REJECT_COST


def score_obp_one(priority, inst, n_items: int) -> float:
    return _score_obp_inst(priority, inst, n_items)


def score_obp_config(cfg: "ConfigAS", inst: dict) -> tuple:
    """Picklable process-pool worker (module-level so it survives spawn). Returns
    ``(cost, wall_seconds)`` — the race's ``_split`` folds the wall into the CPU
    odometer AND records it as the per-(config, instance) ``wall_s`` (CPU-bound
    single-threaded OBP, so wall ~= CPU)."""
    t0 = time.perf_counter()
    try:
        fn = cfg.callable
        cost = _score_obp_inst(fn, inst, 0)
    except Exception:
        cost = REJECT_COST
    dt = time.perf_counter() - t0
    if os.environ.get("EOH_WORKER_TRACE"):
        inst_d, seed = _unwrap(inst)
        opt = inst_d.get("_opt")
        gap = (cost - opt) / opt if opt else float("nan")
        iid = inst_d.get("_id", "?")
        seed_str = f"/seed={seed}" if seed is not None else ""
        print(
            f"    [eval-worker pid={os.getpid()}] {cfg.id}:{iid}{seed_str} "
            f"-> cost={cost:.1f}; gap={gap:.4f}; runtime={dt:.2f}s",
            flush=True,
        )
    return cost, dt


# --------------------------------------------------------------------------- #
# RacingEoH
# --------------------------------------------------------------------------- #


class RacingEoH(RacingBase):
    """EoH outer loop with Friedman race as fitness + selection."""

    _pool_runner = staticmethod(score_obp_config)
    _big_penalty = BIG_PENALTY

    def __init__(
        self,
        *,
        evaluation,
        score_one,
        instances,
        label,
        log_dir,
        llm,
        pop_size,
        max_generations,
        selection_num,
        budget_cap,
        t_first,
        t_each,
        alpha,
        seed,
        opt_by_idx=None,
        mean_opt=None,
        problem_size=5000,
        capacity=100,
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
        # `deterministic` mirrors irace's scenario option: when False (the default,
        # stochastic target) evaluation uses irace-style (instance, seed) tasks —
        # the OBP eval is seeded per task so a non-deterministic LLM heuristic is
        # reproducible and configs stay comparable. The task pool starts as one
        # repetition of the base instances (each wrapped with its own seed) and
        # grows on demand via `_ensure_task_pool` (irace `ntimes` repetition); the
        # per-task seed stream is generated by `_seed_rng`, seeded from the run
        # `seed`. When True, instances stay bare and are evaluated once each — no
        # per-task seeding and no repetition (irace ntimes=1).
        self._deterministic = deterministic
        if not deterministic:
            self._base_instances = list(instances)          # bare OBP dicts
            self._seed_rng = np.random.RandomState(int(seed))
            self._n_reps = 0
            instances = self._build_rep(self._base_instances)  # rep 0
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
        self.score_one = score_one
        self.llm = llm
        self.capacity = int(capacity)
        self.selection_num = selection_num
        self.opt_by_idx = opt_by_idx or {}
        self.mean_opt = mean_opt
        self.problem_size = problem_size
        self._eval_timeout = (
            float(eval_timeout) if eval_timeout and float(eval_timeout) > 0 else None
        )
        # Finite cost assigned to a TIMEOUT (slow-but-valid heuristic killed at
        # eval_timeout). Kept finite; the timeout is then disqualified via the
        # explicit marker in utils.race. A crash returns Inf and is rejected.
        self._big_penalty = float(timeout_cost)

        # Streaming per-(instance, seed) performance log. One JSONL line per
        # genuine (heuristic, task) evaluation, appended as generations finish.
        # `costs_by_inst` is keyed by the 1-based race task index (`inst_idx`);
        # an elite's costs persist across generations, so `_perf_logged` de-dups
        # on (cand_id, task_idx) => each eval logged once.
        self._perf_log_path = self.log_dir / "instance_seed_perf.jsonl"
        self._perf_logged: set = set()

        self.template_str = evaluation.template_program
        self.task_desc = evaluation.task_description
        self.template_fn = TextFunctionProgramConverter.text_to_function(
            self.template_str
        )
        self.sampler = EoHSampler(llm, self.template_str)
        # Per-LLM-call prompt log (mirrors racing/eoh_bbob's llm_prompts.jsonl):
        # the exact rendered context each offspring was sampled from, so every
        # sample is reproducible. Written under self._sample_lock (threaded sampling).
        self._prompt_log_path = self.log_dir / "llm_prompts.jsonl"
        self._llm_call_idx = 0

        self._best_score = float("-inf")  # EoH: higher (less negative) is better
        self._best_gap = float("inf")
        self._needs_pre_eval_cleanup = False
        import itertools

        self._op_iter = itertools.cycle(("e1", "e2", "m1", "m2"))

    # ---- irace-style (instance, seed) task pool --------------------------

    def _build_rep(self, base_instances) -> list:
        """Wrap one repetition of the base instances as SeededInstance tasks,
        each with a fresh seed drawn from `self._seed_rng`. Increments the rep
        counter. Mirrors irace appending one more `ntimes` block to the
        instances_log, each row a distinct (instanceID, seed) pair."""
        rep = self._n_reps
        self._n_reps += 1
        tasks = []
        for base_idx, inst in enumerate(base_instances):
            seed = int(self._seed_rng.randint(0, _SEED_MAX))
            task = SeededInstance(instance=inst, seed=seed,
                                  base_idx=base_idx, rep=rep)
            tasks.append(task)
        return tasks

    def _ensure_task_pool(self, next_instance: int) -> None:
        """Grow the (instance, seed) task pool on demand, irace `ntimes`-style.

        The elitist race needs `next_instance + elitist_new_instances - 1`
        tasks available (race.py contract). If the current pool is shorter,
        append whole repetitions of the base instances (each with fresh seeds)
        until it is long enough. No-op in deterministic mode (no repetition)
        or when the pool is already large enough.
        """
        if self._deterministic or self._base_instances is None:
            return
        need = next_instance + max(1, self.elitist_new_instances) - 1
        while len(self.instances) < need:
            self.instances = self.instances + self._build_rep(self._base_instances)
            print(f"  [seed-pool] extended to {len(self.instances)} tasks "
                  f"({self._n_reps} reps of {len(self._base_instances)} base "
                  f"instances) — need {need}", flush=True)

    # ---- Per-(instance, seed) performance log ----------------------------

    def _task_meta(self, task_idx: int) -> tuple:
        """Resolve a 1-based `costs_by_inst` key (`inst_idx`) to
        `(base_idx, seed, rep)`. `self.instances` is the instances_log, indexed
        `task_idx - 1`. SeededInstance carries base_idx/seed/rep (0-based); a bare
        deterministic instance has no seed, so base_idx == task_idx-1, seed None,
        rep 0."""
        task = self.instances[task_idx - 1]
        if isinstance(task, SeededInstance):
            return task.base_idx, task.seed, task.rep
        return task_idx - 1, None, 0

    def _instance_label(self, inst_idx: int) -> str:
        """``i<base_idx>_s<seed>`` for the verbose>=2 debug grid: the underlying
        base-problem index (0-based, from the fixed instance set) and the per-task
        seed.  In deterministic mode there is no per-task seed, so the run's master
        seed (``self.seed``) is shown as a fixed placeholder."""
        base_idx, seed, _rep = self._task_meta(inst_idx)
        s = seed if seed is not None else self.seed
        return f"i{base_idx}_s{s}"

    def _append_perf_log(self, gen_id: int, race_records) -> None:
        """Append one JSONL line per newly-seen (heuristic, task) evaluation.
        Runs in the main process from `_record_generation` (post-race), so no
        concurrent-writer issue. Elite costs persist across generations; the
        `_perf_logged` set keeps each (cand_id, task_idx) to a single line."""
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        # Prefix position (instance_idx_k): where this task sat in the current
        # race's instance sequence — 0-based task indices in race order.
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
                    "task_idx": int(task_idx),   # 1-based (costs_by_inst key)
                    "base_idx": int(base_idx),   # 0-based (SeededInstance.base_idx)
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
        """Evaluate every initial candidate on a random sample of T_first instances.

        Sets ``rec.mean_cost`` so ``_select_parents`` has meaningful scores for
        the very first offspring-sampling step.  Counts each evaluation toward
        ``self.budget_used``.  Sets ``self._needs_pre_eval_cleanup = True`` so
        the first ``_race`` call resets all history before racing starts.
        """
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
                    f"    [pre-eval] {rec.cfg.id} callable error — {exc}; "
                    f"mean_cost stays inf"
                )
                continue
            for inst in sampled:
                c = self.score_one(fn, inst, self.problem_size)
                costs.append(c)
                self.budget_used += 1
            valid = [c for c in costs if math.isfinite(c) and c < BIG_PENALTY]
            rec.mean_cost = float(np.mean(valid)) if valid else float("inf")
            print(
                f"    [pre-eval] {rec.cfg.id}: mean_cost={rec.mean_cost:.4f} "
                f"({len(valid)}/{n_sample} valid evals)",
                flush=True,
            )
        dt = time.time() - t0
        print(f"  [pre-eval] done in {dt:.1f}s  budget_used={self.budget_used}")
        self._needs_pre_eval_cleanup = True

    def _init_population(self) -> tuple:
        init_recs, next_instance = super()._init_population()
        self._pre_evaluate(init_recs)
        return init_recs, next_instance

    def _race(self, records, next_instance):
        if self._needs_pre_eval_cleanup:
            for rec in records:
                rec.cfg.reset_history()
                rec.mean_cost = float("inf")
            self._needs_pre_eval_cleanup = False
            print(
                "  [pre-eval cleanup] cfg histories and mean_costs reset "
                f"for {len(records)} candidates before first race",
                flush=True,
            )
        return super()._race(records, next_instance)

    # ---- Abstract implementations ----------------------------------------

    def _make_runner(self):
        def runner(params, instance):
            return float(self.score_one(params, instance, self.problem_size))

        return runner

    def _materialize(self, func, idx: int, gen: int = 0) -> CandidateRecord:
        prog = TextFunctionProgramConverter.function_to_program(func, self.template_str)
        src = str(prog)
        within = self._gen_cand_counter.get(gen, 0)
        self._gen_cand_counter[gen] = within + 1
        tag = "p" if gen == 0 else "c"
        cand_id = f"g{gen}_{tag}{within}"
        func.cand_id = cand_id   # so this func, when reused as a parent, is resolvable
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
        """Post-evolution validation via analyses/eval_obp.py for BOTH incumbent rules ->
        valid_trajectory_mean_rank.json / valid_trajectory_mean_cost.json."""
        from analyses.eval_obp import evaluate_single, _to_serialisable
        print(f"\n=== Post-Evolution Final Evaluation (eval_obp.py) ===", flush=True)
        cap = getattr(self, "capacity", 100)
        base_insts = self._base_instances if self._base_instances is not None else self.instances
        results: dict = {}
        for rule in ("mean_rank", "mean_cost"):
            try:
                res = evaluate_single(
                    exp_path=self.log_dir, n_instances=len(base_insts),
                    n_items=self.problem_size, capacity=cap, n_cores=self.num_cores,
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
        valid = self._final_eval()   # {"mean_rank": ..., "mean_cost": ...}
        # Manuscript reliability pairing per incumbent rule (partial vs. full held-out).
        for rule in ("mean_rank", "mean_cost"):
            try:
                self._mlog.finalize_reliability(
                    getattr(self, f"incumbents_{rule}"), valid.get(rule), rule=rule)
            except Exception as e:
                print(f"  [manuscript-log/{rule}] WARN: reliability finalize failed: {e}", flush=True)
        return res

    def _compute_gap(self, rec: CandidateRecord) -> float:
        if (
            rec.n_evals == 0
            or not math.isfinite(rec.mean_cost)
            or not rec.evaluated_idxs
        ):
            return float("inf")
        if self.mean_opt is None or self.mean_opt <= 0:
            return float("inf")
        return float((rec.mean_cost - self.mean_opt) / self.mean_opt)

    def _record_generation(self, gen_id, race_records, table_rows=None):
        race_order = list(self._last_instance_order)
        perfs = []
        for r in race_records:
            gap = self._compute_gap(r)
            if r.n_evals > 0 and math.isfinite(r.mean_cost):
                score = float(-r.mean_cost)
            else:
                score = float(-BIG_PENALTY)
            if (
                r.n_evals > 0
                and math.isfinite(r.mean_cost)
                and score > self._best_score
            ):
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
                    float(-r.mean_cost)
                    if math.isfinite(r.mean_cost)
                    else float(-BIG_PENALTY)
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
            f"[{self.label}] racing EoH (budget_cap={cap_str}, "
            f"max_generations={gen_str}, T_first={self.t_first}, "
            f"T_each={self.t_each}, N_min={self.pop_size}, "
            f"elitist={self.elitist})"
        )

    # ---- EoH sampling ----------------------------------------------------

    def _log_llm_prompt(self, op, gen, prompt, parents=None, status="ok") -> None:
        """Append one JSON record per LLM call to ``llm_prompts.jsonl`` (mirrors
        racing/eoh_bbob), logged AFTER the call so the record can carry
        the call's token usage; every exit path (ok / parse_fail / exception) logs,
        so a crashed sample still leaves its rendered prompt on record. EoH prompts embed the parents' code, so
        a short per-parent algorithm description is logged for provenance."""
        parents_shown = [
            {"desc": (getattr(p, "algorithm", "") or "")[:120]}
            for p in (parents or [])
        ]
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "call_idx": None,   # assigned under the lock below (atomic read-and-increment)
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
        # Read+increment the shared counter AND write inside the lock: with threaded
        # sampling (num_threads>1) reading call_idx outside the lock races and yields
        # duplicate indices.
        with self._sample_lock:
            rec["call_idx"] = self._llm_call_idx
            self._llm_call_idx += 1
            with open(self._prompt_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

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
        feasible = [r for r in pop if math.isfinite(r.mean_cost)]
        ## DEBUG: check the mean_cost
        print(
            f"      selecting {k} parents from {len(pop)} candidates "
            f"({len(feasible)} feasible with finite mean_cost)"
        )
        if not feasible:
            return [self.rng.choice(pop) for _ in range(k)] if pop else []
        ranked = sorted(feasible, key=lambda r: r.mean_cost)
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
                print(f"    [generation {gen}/{op}] sample fail — attempt {attempts}")
                continue
            offspring.append(self._materialize(fn, self._sample_idx, gen=gen))
            self._sample_idx += 1
        return offspring

    def _sample_one_threaded(self, elites, gen, attempt_no):
        with self._sample_lock:
            op = next(self._op_iter)
            k = self.selection_num if op in ("e1", "e2") else 1
            parents = self._select_parents(elites, k)
            parent_fns = [p.candidate for p in parents]
        fn = self._sample(op, parents=parent_fns, gen=gen)
        if fn is None:
            print(f"    [generation {gen}/{op}] sample fail — attempt {attempt_no}")
            return None
        with self._sample_lock:
            idx = self._sample_idx
            self._sample_idx += 1
            rec = self._materialize(fn, idx, gen=gen)
        return rec

    # ---- Heuristic log (EoH has no description field) --------------------

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
    p = argparse.ArgumentParser(description="Racing EoH on Online Bin Packing.")
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Generation cap. Use -1 to disable.",
    )
    p.add_argument("--ref-max-generations", type=int, default=20)
    p.add_argument("--n-instances", type=int, default=5)
    p.add_argument("--n-items", type=int, default=5000)
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY,
                   help="Finite penalty cost recorded when a heuristic is killed at "
                        "--timeout-seconds (a slow-but-valid run). Kept finite; the timeout "
                        "is then disqualified via the timeout marker. A crash returns Inf and "
                        f"is rejected immediately. Default {BIG_PENALTY:g}.")
    p.add_argument("--selection-num", type=int, default=5)
    p.add_argument("--t-first", type=int, default=5)
    p.add_argument("--t-each", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--test-type", type=str, default="friedman", choices=["friedman", "ttest"]
    )
    p.add_argument("--posthoc-test-type", type=str, default="conover", choices=["conover", "nemenyi"],
                   help="Friedman post-hoc test variant. 'conover' (default): original R irace "
                        "formula (Conover 1999) vs best_idx only. 'nemenyi': scikit-posthocs all-pairs "
                        "Nemenyi test (far more conservative — rarely eliminates for large pools; "
                        "if used, raise --elitist-limit).")
    p.add_argument("--no-elitist", dest="elitist", action="store_false", default=True)
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=2,
                   help="In elitist irace, max elimination tests per race that do NOT eliminate a "
                        "configuration before the race early-stops. 0 = no limit. With "
                        "--posthoc-test-type nemenyi raise this (e.g. 10-12), else the race stops "
                        "after only a few instances.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="Treat the target as deterministic (irace scenario option). "
                        "By DEFAULT (omitted) evaluation is STOCHASTIC: irace-style "
                        "(instance, seed) tasks seed the OBP evaluation (np.random + "
                        "random) so a non-deterministic LLM heuristic is reproducible and "
                        "configs are compared on identical random conditions; the "
                        "(instance, seed) pool grows on demand (irace ntimes repetition) "
                        "and the per-task seed stream is derived from --seed. Pass "
                        "--deterministic to instead evaluate each instance ONCE with no "
                        "per-task seeding and no repetition (irace ntimes=1).")
    p.add_argument("--label", type=str, default="race/obp")
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
    )
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--num-cores", type=int, default=1)
    p.add_argument("--save-pop", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from the paired fixed population in "
                        "src/init_pop/ (re-raced this run) instead of sampling the "
                        "initial population from the LLM.")
    p.add_argument("--early-stopping-non-elitist", action="store_true", default=False)
    p.add_argument("--deal-with-crashed", type=str, default="rejection", choices=["rejection", "penalty"],
                   help="How the race treats a crashed candidate (non-finite/inf cost). "
                        "'rejection' (default): faithful irace — drop it immediately "
                        "(strip elite protection, remove from the alive set). 'penalty': keep "
                        "it alive with a finite worst-case cost so it undergoes the statistical "
                        "test — preserves a fragile population from collapsing when heuristics "
                        "crash often.")
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
        f"n_instances={args.n_instances}  n_items={args.n_items}  capacity={args.capacity}"
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

    # run_meta.json: the per-run join / reproducibility record every cross-cell
    # analysis keys on. framework/domain are fixed for this runner; policy reflects the
    # race variant; B/K/hyperparameters come from args.
    from utils.manuscript_log import write_run_meta
    write_run_meta(log_dir, {
        "run_id": run_name,
        "framework": "EoH", "domain": "OBP",
        "policy": ("raceAD" if getattr(args, "elitist", True) else "race-nonelitist"),
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
    evaluation = OBPEvaluation(
        timeout_seconds=None,
        n_instances=args.n_instances,
        n_items=args.n_items,
        capacity=args.capacity,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)

    inst_names = list(evaluation._datasets.keys())
    instances = list(evaluation._datasets.values())
    lb_per_inst = [
        _obp_lower_bound(inst["items"], inst["capacity"]) for inst in instances
    ]
    opt_by_idx = {i: lb for i, lb in enumerate(lb_per_inst)}
    mean_lb = float(np.mean(lb_per_inst))
    for i, inst in enumerate(instances):
        inst["_id"] = inst_names[i]
        inst["_opt"] = lb_per_inst[i]
    print(
        f"  per-instance lb: min={min(lb_per_inst)}  max={max(lb_per_inst)}  "
        f"mean={mean_lb:.2f}  "
        f"(over {args.n_instances} instances of {args.n_items} items, capacity={args.capacity})"
    )

    _racer = RacingEoH(
        evaluation=evaluation,
        score_one=score_obp_one,
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
        alpha=args.alpha,
        seed=args.seed,
        opt_by_idx=opt_by_idx,
        mean_opt=mean_lb,
        problem_size=args.n_items,
        capacity=args.capacity,
        num_threads=args.num_threads,
        num_cores=args.num_cores,
        eval_timeout=args.timeout_seconds,
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
        _fp = ROOT / "src" / "init_pop" / "eoh_obp.json"
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
