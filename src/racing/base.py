"""Shared base class and utilities for the Racing-* family.

Concrete subclasses must implement the abstract methods listed on
``RacingBase``.  Everything else — pool management, sampling dispatch,
the evolution loop, run orchestration, trajectory logging — lives here
and is inherited verbatim.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils import ConfigAS, elitist_race, print_markers_header
from utils.race import _config_scores


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #


class _Tee:
    """Write-through stream splitter: forwards every write to N streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


# --------------------------------------------------------------------------- #
# Incumbent identification (switchable via --incumbent-mode)
# --------------------------------------------------------------------------- #
def select_incumbent(elites, mode: str = "mean_rank", announce_gen=None):
    """Identify a generation's incumbent from the race elites (the survivors).

    Only candidates still ALIVE this generation are eligible: under ``--save-pop``
    the carried population is topped up with candidates ELIMINATED by the race
    (``alive == False``), and those refilled candidates — race-rejected and often
    only partially evaluated — must never be reported as the incumbent (via either
    rule). Also filters out any candidate carrying a crash/penalty cost (cost >= 1e5)
    so a crashed heuristic is never selected. Returns None if no eligible candidate
    remains (0 valid alive candidates). No-op when every elite is a survivor
    (non-save_pop, or callers that already pass an alive-only list).
    """
    if not elites:
        return None

    def _is_valid(e):
        cfg = getattr(e, "cfg", e)
        if not getattr(cfg, "alive", True):   # exclude race-eliminated (refilled) candidates
            return False
        costs = list(getattr(cfg, "costs_by_inst", {}).values())
        return len(costs) > 0 and all(math.isfinite(c) and c < 1e5 for c in costs)

    valid_elites = [e for e in elites if _is_valid(e)]
    if not valid_elites:
        return None

    if mode == "mean_rank":
        return valid_elites[0]
    if mode == "mean_cost":
        # Compare candidates on the COMMON instance set they SHARE, NOT each over its
        # own (possibly different) subset. Averaging over disjoint subsets is
        # non-comparable: a lightly-evaluated candidate on a few easy instances can
        # outrank a deeply-evaluated one. ``--deterministic`` makes it worse: costs are
        # frozen, so a biased small-subset mean never regresses. Same shared-set rule as
        # the sh/* runners (``seen_list`` in sh/eoh_obp.py, ``_seen_mean`` in sh/llamea_bbob.py).
        def _finite_keys(e):
            cfg = getattr(e, "cfg", e)
            return {k for k, c in getattr(cfg, "costs_by_inst", {}).items()
                    if math.isfinite(c) and c < 1e5}

        def _mean_over(e, keys):
            cfg = getattr(e, "cfg", e)
            vals = [cfg.costs_by_inst[k] for k in keys if k in cfg.costs_by_inst]
            return sum(vals) / len(vals) if vals else float("inf")

        common = set.intersection(*[_finite_keys(e) for e in valid_elites])
        if common:
            inc = min(valid_elites, key=lambda e: _mean_over(e, common))
        else:
            # No instance shared by ALL eligible elites (rare — e.g. fully disjoint
            # coverage). Fall back to each candidate's own lifetime mean rather than
            # pick arbitrarily.
            inc = min(valid_elites, key=lambda e: _mean_over(e, _finite_keys(e)))
        if announce_gen is not None:
            inc_id = getattr(getattr(inc, "cfg", inc), "id", "?")
            n_shared = len(common)
            note = "" if common else "  (NO shared instance — fell back to per-candidate lifetime mean)"
            print(f"  [incumbent/mean_cost] gen {announce_gen}: incumbent={inc_id} "
                  f"selected over {len(valid_elites)} eligible elites on "
                  f"{n_shared} shared common instance(s){note}", flush=True)
        return inc
    raise ValueError(f"unknown incumbent_mode: {mode!r} (known: 'mean_rank', 'mean_cost')")


def print_instance_order(instances, deterministic=None, seed=None) -> None:
    """DEBUG: print the initial (instance, seed) pool ORDER, so a run's instance
    ordering is auditable. Each task shows its instance identity (``_id``/name —
    which reveals any --seed shuffle) and per-task seed. Shared by RacingBase and the
    standalone LLaMEA runner (not a RacingBase subclass).

    NOTE: eoh_obp/tsp/fssp do NOT shuffle — the order is the fixed base-dataset order
    regardless of seed (only the per-task seeds change); only eoh_obp_hetero (with
    --data-file) permutes the order by --seed. For non-deterministic runs the pool
    grows on demand — each new rep repeats this base order with fresh seeds (see the
    '[seed-pool] extended' lines)."""
    if not instances:
        return

    def _fmt(t) -> str:
        inst = getattr(t, "instance", t)          # unwrap SeededInstance
        tseed = getattr(t, "seed", None)
        ident = None
        if isinstance(inst, dict):
            ident = inst.get("_id") or inst.get("name")
            if ident is None and "fid" in inst:
                ident = f"f{inst['fid']}i{inst.get('iid', '?')}"
        else:
            ident = getattr(inst, "_id", None) or getattr(inst, "name", None)
        if ident is None:
            ident = getattr(t, "base_idx", "?")
        return f"{ident}" + (f"/s{tseed}" if tseed is not None else "")

    order = [_fmt(t) for t in instances]
    print(f"  [instance-order] deterministic={deterministic} run_seed={seed} "
          f"pool_size={len(instances)} (initial pool)", flush=True)
    print(f"    order: [{', '.join(order)}]", flush=True)


def save_run_log(
    log_dir: pathlib.Path,
    label: str,
    trajectory: list,
    final_eval: Optional[dict] = None,
    incumbents_by_rule: Optional[dict] = None,
) -> None:
    """Write trajectory.json.  ``final_eval`` may contain a ``gap`` key.
    ``incumbents_by_rule`` maps an incumbent rule name -> its per-generation incumbent
    rows; each is written as ``incumbents_<rule>`` (e.g. ``incumbents_mean_rank`` /
    ``incumbents_mean_cost``). The old singular ``incumbents`` key is gone — both rules
    are always recorded."""
    path = log_dir / "trajectory.json"
    log = {"label": label, "trajectory": trajectory}
    for _rule, _rows in (incumbents_by_rule or {}).items():
        if _rows is not None:
            log[f"incumbents_{_rule}"] = _rows
    if final_eval is not None:
        log["final_eval"] = {
            "cand_id": final_eval["cand_id"],
            "score": float(final_eval["mean_cost"]),
            "gap": float(final_eval.get("gap", float("nan"))),
            "n_instances": int(final_eval["num_eval_instances"]),
        }
    json.dump(log, open(path, "w"), indent=2)
    print(f"  logged -> {path}  ({len(trajectory)} table rows)")


def _run_tag(args, bash_defaults: dict, abbrev: dict) -> str:
    """Return a compact folder-name tag encoding args that differ from defaults."""
    overrides = []
    for key, default_val in bash_defaults.items():
        actual_val = getattr(args, key, None)
        if actual_val == default_val:
            continue
        short = abbrev.get(key, key)
        if key == "llm_model" and isinstance(actual_val, str) and "/" in actual_val:
            actual_val = actual_val.split("/")[-1]
        if isinstance(actual_val, bool):
            overrides.append(short if actual_val else f"no{short}")
        else:
            overrides.append(f"{short}{actual_val}")
    expected_bc = args.pop_size * args.ref_max_generations * args.n_instances
    if args.budget_cap is not None and args.budget_cap != expected_bc:
        overrides.append(f"bc{args.budget_cap}")
    return "_".join(overrides) if overrides else "default"


# --------------------------------------------------------------------------- #
# Shared data record
# --------------------------------------------------------------------------- #


@dataclass
class CandidateRecord:
    """Wraps a ConfigAS with racing bookkeeping fields.

    ``candidate`` stores whatever the subclass needs for re-prompting
    (an EoH ``func`` object for RacingEoH, a LLaMEA ``Solution`` for
    RacingLLaMEA).
    """

    cfg: "ConfigAS"
    candidate: object
    mean_cost: float = float("inf")
    sum_ranks: float = float("inf")
    n_evals: int = 0
    eliminated_at: int = -1
    survived: bool = False
    evaluated_idxs: List[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #


class RacingBase(ABC):
    """Abstract base for Racing-* experiments.

    Subclasses must set the following class attributes before calling
    super().__init__:
        _pool_runner   -- module-level picklable function passed to elitist_race
                          as ``pool_runner``.  Set to None to disable the
                          process pool path.

    Subclasses must implement:
        _make_runner()           -> callable(params, instance) -> float
        _materialize(raw, idx, gen) -> CandidateRecord
        _sample_init(gen)        -> raw LLM output or None
        _sample_offspring_seq(elites, gen, max_attempts) -> List[CandidateRecord]
        _sample_one_threaded(elites, gen, attempt_no) -> CandidateRecord | None
        _compute_gap(rec)        -> float  (problem-specific gap vs. optimum)
        _cost_to_score(rec)      -> float  (sign convention: EoH negates, LLaMEA keeps)
        _is_new_best(rec, score) -> bool
        _final_eval(rec)         -> dict
        _record_generation(gen_id, race_records, table_rows) -> None
        _ensure_task_pool(next_instance) -> None  (default: no-op)
    """

    # Subclasses set this to a module-level picklable callable.
    _pool_runner = None
    # Subclasses may override to set per-eval timeout (seconds, or None).
    _eval_timeout: Optional[float] = None
    # Subclasses whose runner reports per-eval CPU time (returns (cost, cpu))
    # set this True to accumulate a run-level CPU odometer and stamp it into
    # each incumbent row. Off by default (OBP rows stay unchanged).
    _track_cpu: bool = False

    def __init__(
        self,
        *,
        instances,
        label,
        log_dir,
        pop_size,
        max_generations,
        budget_cap,
        t_first,
        t_each,
        alpha,
        seed,
        num_threads=1,
        num_cores=1,
        test_type: str = "friedman",
        posthoc_test_type: str = "conover",
        elitist: bool = True,
        elitist_new_instances: int = 1,
        elitist_limit: int = 2,
        save_pop: bool = False,
        early_stopping_non_elitist: bool = False,
        deal_with_crashed: str = "rejection",
        wandb_logger=None,
    ):
        if max_generations is None and budget_cap is None:
            raise ValueError("at least one of max_generations / budget_cap must be set")
        self.instances = instances
        self.label = label
        self.log_dir = log_dir
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.budget_cap = budget_cap
        self.t_first, self.t_each, self.alpha = t_first, t_each, alpha
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._n_races = 0
        self.num_threads = max(1, int(num_threads))
        self.num_cores = max(1, int(num_cores))
        if test_type not in ("friedman", "ttest"):
            raise ValueError(
                f"test_type must be 'friedman' or 'ttest', got {test_type!r}"
            )
        self.test_type = test_type
        self.posthoc_test_type = posthoc_test_type
        self.elitist = bool(elitist)
        self.elitist_new_instances = int(elitist_new_instances)
        self.elitist_limit = int(elitist_limit)
        self.save_pop = bool(save_pop)
        self.early_stopping_non_elitist = bool(early_stopping_non_elitist)
        if deal_with_crashed not in ("rejection", "penalty"):
            raise ValueError(
                f"deal_with_crashed must be 'rejection' or 'penalty', got {deal_with_crashed!r}"
            )
        self.deal_with_crashed = str(deal_with_crashed)
        self.wandb_logger = wandb_logger

        # Thread pool for parallel LLM sampling
        self._sampler_pool = None
        if self.num_threads > 1:
            import concurrent.futures as _cf2
            import threading as _th

            self._sampler_pool = _cf2.ThreadPoolExecutor(
                max_workers=self.num_threads, thread_name_prefix="racing-sampler"
            )
            self._sample_lock = _th.Lock()
        else:
            import threading as _th

            self._sample_lock = _th.Lock()

        # Process pool for parallel per-instance evaluation
        self._eval_pool = None
        if self.num_cores > 1:
            import atexit as _atexit
            import concurrent.futures as _cf3

            self._eval_pool = _cf3.ProcessPoolExecutor(max_workers=self.num_cores)
            _atexit.register(self._shutdown_eval_pool)

        self.budget_used = 0
        self._best_record: Optional[CandidateRecord] = None
        self._sample_idx = 0
        self.generations: list = []
        self.trajectory: list = []
        # BOTH per-generation incumbent series are recorded every run (the
        # --incumbent-mode global knob was removed): one row per generation per rule.
        # mean_rank = the irace overall_ranks best elite (elites[0]); mean_cost = the
        # survivor with the best lifetime mean objective. Each is validated separately
        # into valid_trajectory_mean_rank.json / valid_trajectory_mean_cost.json.
        self.incumbents_mean_rank: list = []
        self.incumbents_mean_cost: list = []
        # Structured JSONL logs (see utils/manuscript_log.py).
        from utils.manuscript_log import ManuscriptLogger
        self._n_full_instances = len(self.instances)
        self._mlog = ManuscriptLogger(self.log_dir, self._n_full_instances)
        # Cumulative CPU seconds (summed across cores) spent on ALL evaluations
        # so far; stamped into each incumbent row when ``_track_cpu`` is set.
        self._cpu_seconds_used: float = 0.0
        self._last_instance_order: list = []
        self._heuristics: list = []
        self._gen_cand_counter: dict = {}
        self.final_eval: Optional[dict] = None
        self._timings: dict = {"samples": [], "generations": [], "final_eval": None}
        self._op_iter = itertools.cycle(["refine"])

    # ---- Pool management -------------------------------------------------

    def _shutdown_eval_pool(self) -> None:
        pool = self._eval_pool
        if pool is None:
            return
        self._eval_pool = None
        procs = list((getattr(pool, "_processes", None) or {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()

    def _recreate_eval_pool(self, old_pool) -> object:
        self._eval_pool = old_pool
        self._shutdown_eval_pool()
        import concurrent.futures as _cf3

        self._eval_pool = _cf3.ProcessPoolExecutor(max_workers=self.num_cores)
        return self._eval_pool

    def _save_timings(self) -> None:
        json.dump(self._timings, open(self.log_dir / "timings.json", "w"), indent=2)

    # ---- Abstract interface ----------------------------------------------

    @abstractmethod
    def _make_runner(self):
        """Return a callable ``runner(params, instance) -> float``."""

    @abstractmethod
    def _materialize(self, raw, idx: int, gen: int = 0) -> CandidateRecord:
        """Turn a raw LLM output into a ``CandidateRecord``."""

    @abstractmethod
    def _sample_init(self, gen: int = 0):
        """Return a raw LLM output for initial population, or None on failure."""

    def _raw_from_source(self, source: str):
        """Compile a fixed-init heuristic ``source`` into the same 'raw' form that
        ``_sample_init`` returns, so ``_materialize`` can wrap it (for --fix-init-pop).
        Override in the framework runner; the default refuses fixed-init."""
        raise NotImplementedError(
            "This runner does not support --fix-init-pop (no _raw_from_source).")

    @abstractmethod
    def _sample_offspring_seq(
        self, elites: List[CandidateRecord], gen: int, max_attempts: int
    ) -> List[CandidateRecord]:
        """Sequential offspring sampling."""

    @abstractmethod
    def _sample_one_threaded(
        self, elites: List[CandidateRecord], gen: int, attempt_no: int
    ) -> Optional[CandidateRecord]:
        """One threaded offspring sample."""

    @abstractmethod
    def _final_eval(self, rec: CandidateRecord) -> dict:
        """Full final evaluation of the best record."""

    @abstractmethod
    def _record_generation(
        self, gen_id: int, race_records: List[CandidateRecord], table_rows=None
    ) -> None:
        """Log one generation's results."""

    def _ensure_task_pool(self, next_instance: int) -> None:
        """Optionally extend self.instances before _race.  No-op by default."""

    def _instance_label(self, inst_idx: int) -> str:
        """Display label for a 1-based race-task index, used by the verbose>=2
        debug grid (``race.py`` ``_render_debug_table``).  Default: the bare task
        index.  Subclasses that carry per-task seeds override this to show
        ``i<base_idx>_s<seed>`` (see ``racing.eoh_obp.RacingEoH``)."""
        return str(inst_idx)

    # ---- Heuristic log ---------------------------------------------------

    def _save_heuristics(self) -> None:
        score_by_id = {}
        for gen in self.generations:
            for p in gen.get("performances", []):
                score_by_id[p["cand_id"]] = (
                    float(p.get("score")) if p.get("score") is not None else None
                )
        out = []
        for h in self._heuristics:
            entry = {
                "cand_id": h["cand_id"],
                "gen_id": int(h["gen_id"]),
                "score": score_by_id.get(h["cand_id"]),
                "source": h["source"],
            }
            if "description" in h:
                entry["description"] = h["description"]
            out.append(entry)
        json.dump(
            {"label": self.label, "total_sampled": len(out), "heuristics": out},
            open(self.log_dir / "heuristics.json", "w"),
            indent=2,
        )

    # ---- Population initialization ----------------------------------------

    def _init_population(self) -> tuple:
        """Sample pop_size initial candidates.  No evaluation."""
        fence = "#" * 80
        print(f"""
{fence}
# Generation 00 race — initial population ({self.pop_size} candidates)
{fence}""")
        init_recs: List[CandidateRecord] = []
        fixed = getattr(self, "_fixed_init_sources", None)
        if fixed:
            # --fix-init-pop: materialize the pre-existing heuristics as generation 0
            # instead of sampling them from the LLM. They are (re-)raced on THIS run's
            # instances like any candidate — only the LLM initialisation is skipped.
            for src in fixed:
                raw = self._raw_from_source(src)
                if raw is None:
                    print("    [fixed-init-pop] WARNING: a fixed heuristic failed to "
                          "parse; skipped")
                    continue
                rec = self._materialize(raw, self._sample_idx, gen=0)
                self._sample_idx += 1
                init_recs.append(rec)
            print(f"  [fixed-init-pop] seeded {len(init_recs)} initial candidates from "
                  f"the fixed population (no LLM sampling)", flush=True)
        else:
            total_attempts, max_total = 0, 4 * self.pop_size
            while len(init_recs) < self.pop_size and total_attempts < max_total:
                total_attempts += 1
                raw = self._sample_init(gen=0)
                if raw is None:
                    print(f"    [initialization] parse fail — attempt {total_attempts}")
                    continue
                rec = self._materialize(raw, self._sample_idx, gen=0)
                self._sample_idx += 1
                init_recs.append(rec)

        if len(init_recs) < self.pop_size:
            print(
                f"    [initialization] WARNING: only {len(init_recs)}/{self.pop_size} "
                f"sampled after {total_attempts} attempts"
            )

        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        print(
            f"  init: pool={len(init_recs)} (no evaluation yet), "
            f"budget={self.budget_used}/{cap_str}"
        )
        return list(init_recs), 1

    # ---- Offspring sampling ----------------------------------------------

    def _sample_offspring(
        self, elites: List[CandidateRecord], gen: int
    ) -> List[CandidateRecord]:
        max_attempts = 4 * self.pop_size
        if self._sampler_pool is None:
            return self._sample_offspring_seq(elites, gen, max_attempts)
        return self._sample_offspring_par(elites, gen, max_attempts)

    def _sample_offspring_par(self, elites, gen, max_attempts):
        import concurrent.futures as _cf

        offspring: List[CandidateRecord] = []
        attempts = 0
        print(f"=== Parallel offspring sampling with {self.num_threads} threads ===")
        while len(offspring) < self.pop_size and attempts < max_attempts:
            need = self.pop_size - len(offspring)
            budget = max_attempts - attempts
            batch_size = max(1, min(self.num_threads, need, budget))
            futures = []
            for _ in range(batch_size):
                attempts += 1
                futures.append(
                    self._sampler_pool.submit(
                        self._sample_one_threaded, elites, gen, attempts
                    )
                )
            for fut in _cf.as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(
                        f"    [generation {gen}] sampler thread exception: "
                        f"{type(e).__name__}: {e}"
                    )
                    rec = None
                if rec is not None and len(offspring) < self.pop_size:
                    offspring.append(rec)
        return offspring

    # ---- Race ------------------------------------------------------------

    def _race(self, records: List[CandidateRecord], next_instance: int) -> dict:
        """Run one elitist race via ``elitist_race``.

        Delegates the entire race to the library; maps results back onto
        ``CandidateRecord`` wrappers. Subclasses extend behaviour through:
          - ``_ensure_task_pool``  (called before ``elitist_race``)
          - ``_make_runner``        (the (params, instance)->cost closure)
          - ``_pool_runner``        (module-level picklable counterpart)
          - ``_eval_timeout``       (per-eval wall-clock cap, or None)
        """
        if not records:
            return {
                "survivors": [],
                "next_instance": next_instance,
                "experiments_used": 0,
                "break_reason": "no candidates",
            }

        cfgs = [r.cfg for r in records]
        id2rec = {r.cfg.id: r for r in records}

        if not self.elitist:
            for c in cfgs:
                c.reset_history()

        self._ensure_task_pool(next_instance)

        max_exp = len(cfgs) * len(self.instances) + 1
        runner = self._make_runner()
        seed = self.seed + self._n_races
        race_idx = self._n_races
        self._n_races += 1

        # race_log.jsonl phase-1: snapshot the roster ENTERING the race (after any
        # non-elitist reset_history above), before elitist_race mutates the cfgs.
        from utils.race_log import snapshot_candidates, append_race_record
        _race_budget_before = self.budget_used
        _race_before_roster = snapshot_candidates(cfgs)

        out = elitist_race(
            configurations=cfgs,
            target_runner=runner,
            instances_log=self.instances,
            max_experiments=max_exp,
            next_instance=next_instance,
            elitist=self.elitist,
            elitist_new_instances=self.elitist_new_instances,
            elitist_limit=self.elitist_limit,
            early_stopping_non_elitist=self.early_stopping_non_elitist,
            first_test=self.t_first,
            each_test=self.t_each,
            test_type=self.test_type,
            posthoc_test_type=self.posthoc_test_type,
            alpha=self.alpha,
            metric="sum_ranks",
            min_survival=self.pop_size,
            sample_instances=True,
            seed=seed,
            verbose=2,
            eval_pool=self._eval_pool,
            pool_runner=self._pool_runner,
            eval_timeout=self._eval_timeout,
            timeout_cost=getattr(self, "_big_penalty", 1e6),
            crash_penalty=(None if self.deal_with_crashed == "rejection"
                           else getattr(self, "_big_penalty", 1e6)),
            pool_recreate=self._recreate_eval_pool,
            instance_label=self._instance_label,
        )

        budget_before = self.budget_used
        self.budget_used += out["experiments_used"]
        if self._track_cpu:
            self._cpu_seconds_used += float(out.get("cpu_seconds", 0.0))
        seen = out["seen_instances"]
        self._last_instance_order = [i - 1 for i in out["race_instances"]]

        alive_ids = {c.id for c in out["survivors"]}
        all_scores = _config_scores(list(cfgs), seen, "sum_ranks")
        seen_set = set(seen)
        for c in cfgs:
            rec = id2rec[c.id]
            race_keys = [k for k in c.costs_by_inst if k in seen_set]
            vals = [c.costs_by_inst[k] for k in race_keys]
            rec.mean_cost = float(sum(vals) / len(vals)) if vals else float("inf")
            rec.sum_ranks = float(all_scores.get(id(c), float("inf")))
            rec.n_evals = len(race_keys)
            rec.evaluated_idxs = sorted(k - 1 for k in race_keys)
            rec.eliminated_at = -1 if c.id in alive_ids else rec.n_evals
            rec.survived = c.id in alive_ids

        survivor_cfgs = [c for c in cfgs if c.alive]
        surv_scores = _config_scores(survivor_cfgs, seen, "sum_ranks")
        survivors = sorted(
            (id2rec[c.id] for c in survivor_cfgs),
            key=lambda r: (surv_scores[id(r.cfg)], r.mean_cost),
        )

        # --- Elite-protection diagnostic -------------------------------------
        # `_config_scores` ranks survivors by instance-count tier first (irace
        # `overall_ranks`): a config evaluated on more instances always ranks
        # ahead of one with fewer, so an under-sampled offspring cannot displace
        # a well-evaluated elite. Print only when tiering is actually in play
        # (survivors span >1 instance count) so single-tier generations stay
        # quiet. This is how you confirm the fix is active in a run's logs.
        seen_set_diag = set(seen)
        n_seen_by_id = {
            r.cfg.id: sum(1 for k in r.cfg.costs_by_inst if k in seen_set_diag)
            for r in survivors
        }
        tier_counts = sorted({n for n in n_seen_by_id.values()}, reverse=True)
        if len(tier_counts) > 1:
            top = survivors[: self.pop_size]
            n_top = [n_seen_by_id[r.cfg.id] for r in top]
            tail = survivors[self.pop_size:]
            # An under-sampled config "held back" = a config with fewer instances
            # that landed outside the elite slots while a better-evaluated config
            # holds a slot. This is exactly what the fix protects against.
            held_back = [
                r.cfg.id for r in tail
                if n_seen_by_id[r.cfg.id] < max(n_top, default=0)
            ]
            print(
                f"  [elite-protection] survivor instance-count tiers={tier_counts} "
                f"-> top-{self.pop_size} kept on n_instances={n_top}"
                + (f"; held below better-evaluated elites: {held_back}" if held_back else "")
            )

        all_records_sorted = survivors + [
            id2rec[c.id] for c in out["all_candidates_sorted"] if not c.alive
        ]

        table_rows = self._build_table_rows(out, budget_before)

        # race_log.jsonl phase-2: snapshot the SETTLED roster (restricted to the
        # instances this race used) + the budget bracket. Best-effort; survivors
        # first, then eliminated, matching all_records_sorted's order.
        try:
            _after_cfgs = [r.cfg for r in all_records_sorted]
            append_race_record(self.log_dir, {
                "race_idx": int(race_idx),
                "gen_id": int(getattr(self, "_cur_gen_id", -1)),
                "operator": getattr(self, "_cur_operator", None),
                "seed": int(seed),
                "phase_before": {
                    "used_budget": int(_race_budget_before),
                    "n_candidates": len(_race_before_roster),
                    "candidates": _race_before_roster,
                },
                "phase_after": {
                    "used_budget": int(self.budget_used),
                    "experiments_used": int(out["experiments_used"]),
                    "break_reason": out.get("break_reason", ""),
                    "n_survivors": len(survivors),
                    # Pass the race's computed rank-sum dict (all_scores, keyed by id(cfg)):
                    # the snapshot reads ConfigAS objects, whose sum_ranks property is 0 in
                    # elitist mode (ranks live only in all_scores / _config_scores), so without
                    # this the logged sum_ranks would be 0 for every candidate.
                    "candidates": snapshot_candidates(_after_cfgs, set(seen), scores=all_scores),
                },
            })
        except Exception as e:
            print(f"  [race-log] WARN: base._race logging failed: {e}", flush=True)

        return {
            "survivors": survivors,
            "all_records_sorted": all_records_sorted,
            "next_instance": out["next_instance"],
            "experiments_used": out["experiments_used"],
            "break_reason": out["break_reason"],
            "table_rows": table_rows,
            "step_trace": out.get("step_trace", []),
            "cpu_seconds": float(out.get("cpu_seconds", 0.0)),
            "seen_instances": out.get("seen_instances", []),
        }

    def _build_table_rows(self, race_out: dict, budget_before: int) -> list:
        """Convert raw race table rows to trajectory entries.  Override to add gap."""
        rows = []
        for row in race_out.get("table_rows", []):
            mb = float(row["mean_best"])
            rows.append(
                {
                    "inst_idx": int(row["inst_idx"]),
                    "cand_id": row["cand_id"],
                    "score": mb,
                    "gap": float("nan"),
                    "n_instances": int(row["n_instances"]),
                    "used_budget": budget_before + int(row["experiments_used"]),
                }
            )
        return rows

    # ---- Evolution loop --------------------------------------------------

    def _post_generation_hook(self, elites: List[CandidateRecord]) -> None:
        """Called after elite selection each generation.  No-op by default."""

    @property
    def _incumbents_by_rule(self) -> dict:
        """The two per-generation incumbent series, keyed by rule — for save_run_log
        (``incumbents_mean_rank`` / ``incumbents_mean_cost``) and the two-file validation."""
        return {"mean_rank": self.incumbents_mean_rank,
                "mean_cost": self.incumbents_mean_cost}

    @property
    def incumbents(self) -> list:
        """Alias for the mean_rank incumbent series. trajectory.json stores both series
        (see save_run_log)."""
        return self.incumbents_mean_rank

    def _finalize_reliability_both(self, valid: dict) -> None:
        """Pair each incumbent RULE's per-gen partial score with its full held-out score,
        writing fitness_reliability_log_<rule>.jsonl. ``valid`` = {rule: eval result}."""
        for rule in ("mean_rank", "mean_cost"):
            try:
                self._mlog.finalize_reliability(
                    getattr(self, f"incumbents_{rule}"), (valid or {}).get(rule), rule=rule)
            except Exception as e:
                print(f"  [manuscript-log/{rule}] WARN: reliability finalize failed: {e}",
                      flush=True)

    def _record_incumbent(self, gen_id: int, elites: List[CandidateRecord]) -> None:
        """Snapshot the per-generation incumbent (solution A, irace-faithful).

        The incumbent is the current ``overall_ranks`` elite #1 — ``elites[0]``,
        since ``_race`` sorts survivors by ``_config_scores(..., "sum_ranks")`` — i.e.
        exactly irace's best (``which.min(race_ranks)`` / ``getFinalElites(n=1)``,
        ``race.R:1228``). It is recomputed each generation over the elites'
        accumulated results and is allowed to change or regress (NOT advance-only);
        a monotone best-so-far view is a display choice left to the notebook
        (``np.minimum.accumulate``). No extra evaluation is spent: ``score`` is the
        incumbent's mean over its ALREADY-recorded per-task costs (the race metric —
        relative gap for the hetero variant), matching irace's
        ``mean(Results[, best])``. Appends one row to ``self.incumbents``.
        """
        if not elites:
            return
        # Record BOTH incumbent rules every generation (the --incumbent-mode knob was
        # removed): mean_rank = elites[0] (irace overall_ranks best, coverage-protected);
        # mean_cost = survivor with the best lifetime mean objective. Same elites, two
        # labels — no extra search, both series validated separately at the end.
        inc_rank = select_incumbent(elites, "mean_rank")
        if inc_rank is not None:
            self.incumbents_mean_rank.append(self._incumbent_row(gen_id, inc_rank))
        inc_cost = select_incumbent(elites, "mean_cost", announce_gen=gen_id)
        if inc_cost is not None:
            self.incumbents_mean_cost.append(self._incumbent_row(gen_id, inc_cost))

    def _incumbent_row(self, gen_id: int, inc: CandidateRecord) -> dict:
        """Build one incumbent trajectory row from an elite. ``score`` is the
        incumbent's mean over its ALREADY-recorded per-task costs (no re-eval)."""
        costs = list(inc.cfg.costs_by_inst.values())
        finite = [c for c in costs if math.isfinite(c)]
        value = float(sum(finite) / len(finite)) if finite else float("inf")
        row = {
            "gen_id": int(gen_id),
            "cand_id": inc.cfg.id,
            "used_budget": int(self.budget_used),
        }
        if self._track_cpu:
            # Cumulative CPU seconds (summed across cores) burned by ALL
            # evaluations (every heuristic × instance, not just this incumbent)
            # up to this generation — "how costly it was to discover this
            # incumbent". Logged next to used_budget.
            row["cpu_seconds"] = round(self._cpu_seconds_used, 3)
        row["score"] = value                # lifetime mean cost (no re-eval)
        row["n_instances"] = len(costs)     # incumbent's coverage at this generation
        return row

    def _evolve(
        self, elites: List[CandidateRecord], next_instance: int
    ) -> List[CandidateRecord]:
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        g = 0
        while self.max_generations is None or g < self.max_generations:
            if self.budget_cap is not None and self.budget_used >= self.budget_cap:
                print(
                    f"  [budget cap reached] {self.budget_used}/{cap_str} — stopping run"
                )
                break
            offspring = self._sample_offspring(elites, gen=g + 1)
            if not offspring:
                print(f"  gen {g+1:02d}: 0 offspring this round — stopping run")
                break
            n_elites_in = len(elites)
            combined = elites + offspring
            print(
                f"\n# Generation {g+1} race — {len(combined)} configs "
                f"({n_elites_in} elites + {len(offspring)} offspring)"
            )
            t_race = time.time()
            self._cur_gen_id = g + 1   # for race_log.jsonl (eoh/llamea: 1 race/gen)
            self._cur_operator = None
            out = self._race(combined, next_instance)
            dt_race = time.time() - t_race
            survivors = out["survivors"]
            next_instance = out["next_instance"]
            # No-progress guard (matters for budget-atomic SH): if the race spent no
            # fresh evaluations, the remaining budget is smaller than even the minimal
            # round, so budget_used has frozen BELOW budget_cap and the
            # `budget_used >= budget_cap` stop above would loop forever re-sampling
            # offspring. Stop so run() proceeds to _final_eval. Harmless for the F-race
            # (it always spends >0 fresh evaluations per generation).
            if out.get("experiments_used", 0) == 0 and self.budget_cap is not None:
                print(f"  [budget exhausted] race made no fresh evaluations "
                      f"(remaining budget < one round) — stopping run", flush=True)
                break
            if self.save_pop and len(survivors) < self.pop_size:
                elites = out["all_records_sorted"][: self.pop_size]
                print(
                    f"  [refill] top up elites with eliminated candidates "
                    f"(survivors={len(survivors)} < pop_size={self.pop_size})"
                )
            else:
                elites = survivors[: self.pop_size]
            self._post_generation_hook(elites)
            self._record_incumbent(g + 1, elites)
            self._record_generation(g + 1, combined, out.get("table_rows"))
            self._log_manuscript_eoh(g + 1, combined, offspring, out, dt_race, elites)
            print(
                f"  gen {g+1:02d}/{gen_str}: pool={len(combined)} "
                f"({n_elites_in} elites+{len(offspring)} offspring), "
                f"survivors={len(survivors)}, elites_kept={len(elites)}, "
                f"budget={self.budget_used}/{cap_str}, "
                f"race_wall={dt_race:.1f}s ({out['break_reason']})"
            )
            self._timings["generations"].append(
                {
                    "gen_id": g + 1,
                    "pool_size": len(combined),
                    "n_survivors": len(survivors),
                    "n_elites_kept": len(elites),
                    "race_wall": dt_race,
                }
            )
            self._save_timings()
            g += 1
        return elites

    # ---- Run orchestration -----------------------------------------------

    def _run_header(self) -> str:
        """Override to customise the opening log line."""
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        return (
            f"[{self.label}] racing (budget_cap={cap_str}, "
            f"max_generations={gen_str}, T_first={self.t_first}, "
            f"T_each={self.t_each}, N_min={self.pop_size}, "
            f"elitist={self.elitist})"
        )

    # ---- Manuscript structured logging (shared by all eoh runners) ------

    def _perf_row_extras(self, cfg, task_idx, cost, order) -> dict:
        """Thin wrapper over ``utils.manuscript_log.perf_row_extras`` (shared with the
        standalone LLaMEA runner), passing this runner's penalty threshold."""
        from utils.manuscript_log import perf_row_extras
        return perf_row_extras(cfg, task_idx, cost, order,
                               float(getattr(self, "_big_penalty", 1e5)))

    def _llm_token_totals(self):
        """(cumulative calls, prompt_tokens, completion_tokens) from the ``CachedLLM``
        behind ``self.llm``. Robust to ``self.llm`` being the LLM4AD wrapper (exposes
        ``._cached``) or a ``CachedLLM`` directly (the HiFo bridge path)."""
        llm = getattr(self, "llm", None)
        cached = getattr(llm, "_cached", None) or llm
        return (int(getattr(cached, "total_calls", 0) or 0),
                int(getattr(cached, "total_prompt_tokens", 0) or 0),
                int(getattr(cached, "total_completion_tokens", 0) or 0))

    def _instance_meta_eoh(self, inst_idx) -> dict:
        """Resolve a 1-based race inst_idx to instance metadata for the trace."""
        if not inst_idx or not (0 < inst_idx <= len(self.instances)):
            return {}
        meta: dict = {}
        try:
            if getattr(self, "_instance_label", None):
                lbl = self._instance_label(inst_idx)
                if lbl:
                    meta["instance_id"] = lbl
        except Exception:
            pass
        task = self.instances[inst_idx - 1]
        d = getattr(task, "instance", task)
        if isinstance(d, dict):
            for k in ("fid", "iid"):
                if k in d:
                    meta[k] = int(d[k])
        if hasattr(task, "seed"):
            meta["seed"] = getattr(task, "seed")
        if hasattr(task, "rep"):
            meta["rep"] = getattr(task, "rep")
        return meta

    def _log_manuscript_eoh(self, gen_id, combined, offspring, out, dt_race, elites,
                            log_steps: bool = True) -> None:
        """Emit race_step_trace / race_summary / candidate_log / diversity_log for
        this generation (eoh path; records wrap ConfigAS in CandidateRecord).
        Best-effort — never breaks the run.

        ``log_steps=False`` skips the step trace only. HiFo runs several sub-races
        per generation and logs each one's trace itself (tagged with ``op``) as the
        sub-race finishes; this call would otherwise append the last sub-race's
        steps a second time. The other three artefacts stay per generation.
        """
        try:
            from utils.manuscript_log import code_hash
            ml = self._mlog
            elite_ids = {r.cfg.id for r in elites}

            def _crashed(rec):
                return any((not math.isfinite(v)) or v >= 1e5
                           for v in rec.cfg.costs_by_inst.values())

            def _timed(rec):
                return bool(getattr(rec.cfg, "timed_out_insts", None))

            def _status(rec):
                if rec.cfg.alive:
                    return "survived_elite" if rec.cfg.id in elite_ids else "survived"
                if rec.cfg.id in elite_ids:
                    return "carried_diversity"
                if _timed(rec):
                    return "timeout"
                return "crashed" if _crashed(rec) else "eliminated"

            n_crashed = sum(1 for r in combined if (not r.cfg.alive) and _crashed(r) and not _timed(r))
            n_timeout = sum(1 for r in combined if _timed(r))
            n_elim = sum(1 for r in combined if (not r.cfg.alive) and not _crashed(r) and not _timed(r))

            if log_steps:
                ml.log_race_steps(gen_id, out.get("step_trace", []),
                                  instance_meta=self._instance_meta_eoh)

            inc = self.incumbents_mean_rank[-1] if self.incumbents_mean_rank else {}
            # Refill: elites kept that did NOT survive the race are the
            # topped-up (refilled) candidates.
            survivor_ids = {r.cfg.id for r in out.get("survivors", [])}
            refilled = [e for e in elites if e.cfg.id not in survivor_ids]
            refill_keys = {e.cfg.id: (e.mean_cost if math.isfinite(e.mean_cost) else float("inf"))
                           for e in refilled}
            # Both incumbent rules, every generation.
            try:
                inc_rank = select_incumbent(elites, "mean_rank")
                inc_cost = select_incumbent(elites, "mean_cost")
            except Exception:
                inc_rank = inc_cost = None
            # Cumulative LLM cost from the CachedLLM behind self.llm.
            _calls, _ptok, _ctok = self._llm_token_totals()
            ml.log_race_summary(ml.race_summary_record(
                gen_id=gen_id, n_candidates_init=len(combined),
                n_survivors=len(out.get("survivors", [])),
                n_refilled=len(refilled),
                refilled_ids=[e.cfg.id for e in refilled],
                refill_keys=refill_keys,
                instances_evaluated_max=len(out.get("seen_instances", [])),
                total_evaluations_spent=int(out.get("experiments_used", 0)),
                budget_consumed_cum=int(self.budget_used),
                topup_evaluations=0, stop_reason=out.get("break_reason", ""),
                race_wall_seconds=dt_race, cpu_seconds=float(out.get("cpu_seconds", 0.0)),
                llm_calls_cum=_calls, llm_prompt_tokens_cum=_ptok,
                llm_completion_tokens_cum=_ctok,
                incumbent_cand_id=inc.get("cand_id"),
                incumbent_mode="both",
                incumbent_rank_id=(inc_rank.cfg.id if inc_rank is not None else None),
                incumbent_cost_id=(inc_cost.cfg.id if inc_cost is not None else None),
                incumbent_partial_score=inc.get("score"),
                incumbent_coverage=int(inc.get("n_instances", 0)),
                crashed_count=n_crashed, timeout_count=n_timeout, eliminated_count=n_elim))

            for r in offspring:
                ml.log_candidate({
                    "cand_id": r.cfg.id, "gen_born": gen_id,
                    "parent_ids": getattr(r.candidate, "parent_ids", None),
                    "operator": getattr(r.candidate, "operator", None),
                    "status": _status(r),
                    "n_instances_evaluated": len(r.cfg.costs_by_inst),
                    "lifetime_mean_cost": (r.mean_cost if math.isfinite(r.mean_cost) else float("inf")),
                    "is_elite_carried": r.cfg.id in elite_ids,
                    "elite_credit_span": int(getattr(r.cfg, "is_elite_credit", 0)),
                    "code_hash": code_hash(r.cfg.source),
                })

            alive_elites = [r for r in elites if r.cfg.alive]
            dead_elites = [r for r in elites if not r.cfg.alive]
            srcs = [r.cfg.source for r in alive_elites]
            fits = [(-r.mean_cost if math.isfinite(r.mean_cost) else float("nan")) for r in alive_elites]
            ml.log_diversity(ml.diversity_record(
                gen_id=gen_id, survivor_sources=srcs, survivor_fitnesses=fits,
                crashed_fraction=(n_crashed + n_timeout) / max(1, len(combined)),
                eliminated_carried_count=len(dead_elites)))
        except Exception as e:
            print(f"  [manuscript-log] WARN: gen {gen_id} eoh logging failed: {e}", flush=True)

    def _finalize_best(self):
        """Select the run's winning record and build its final-eval dict.

        Returns ``(winner_record, final_eval_dict)`` or ``(None, None)``.  Default
        behaviour: the record tracked as best by ``mean_cost`` (``self._best_record``),
        re-evaluated on the full instance pool via ``_final_eval``.  Subclasses may
        override to select by irace rank (``overall_ranks``) and/or report the winner's
        already-recorded costs without re-running the heuristic.
        """
        if self._best_record is None:
            return None, None
        return self._best_record, self._final_eval(self._best_record)

    def _debug_instance_order(self) -> None:
        """DEBUG: print the initial (instance, seed) pool order (see module-level
        ``print_instance_order``)."""
        print_instance_order(getattr(self, "instances", None),
                             getattr(self, "_deterministic", None),
                             getattr(self, "seed", None))

    def run(self):
        print(self._run_header())
        if self.elitist:
            print(
                "  [elite-protection] active: survivor ranking tiers by "
                "instance count (irace overall_ranks) — under-sampled offspring "
                "cannot displace better-evaluated elites."
            )
        print_markers_header()
        self._debug_instance_order()
        t0 = time.time()
        try:
            elites, next_instance = self._init_population()
            elites = self._evolve(elites, next_instance)
        finally:
            if self._sampler_pool is not None:
                self._sampler_pool.shutdown(wait=True)
                self._sampler_pool = None
            self._shutdown_eval_pool()
        dt = time.time() - t0
        self.final_elites = elites
        print(
            f"[{self.label}] done in {dt:.1f}s, {len(self.generations)} generations, "
            f"final budget={self.budget_used}, surviving elites={len(elites)}"
        )
        # No post-evolution final evaluation: the per-generation incumbent
        # trajectory logged during the run is the sole output (final_eval stays
        # None, so save_run_log omits the final_eval block).
        self._timings["total_wall"] = dt
        save_run_log(self.log_dir, self.label, self.trajectory, self.final_eval,
                     incumbents_by_rule=self._incumbents_by_rule)
        self._save_heuristics()
        self._save_timings()
        if self.wandb_logger:
            self.wandb_logger.finish()
        return self
