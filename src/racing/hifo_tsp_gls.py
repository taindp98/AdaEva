"""Racing HiFo-Prompt on the LLM4AD TSP-GLS task.

The TSP counterpart of ``racing/hifo_obp.py``. It reuses the SAME racing engine
(:class:`racing.base.RacingBase` + the Friedman/Nemenyi ``elitist_race``) and the
SAME TSP-GLS scoring / (instance, seed) task pool (imported verbatim from
``racing.eoh_tsp_gls``), but swaps EoH's stateless prompts for HiFo-Prompt's
stateful machinery:

  * Foundational operators e1/e2/m1/m2/m3 (incl. HiFo's ``m3`` simplification),
    built via HiFo's ``Evolution.get_prompt_*`` over a TSP ``update_edge_distance``
    prompt spec (``_TSPPrompts``, identical to the one in ``reprod/hifo_tsp_gls`` —
    kept as a local copy so racing has no cross-family import; keep the two in sync).
  * Hindsight — an ``InsightPool`` whose tips are injected into every prompt and
    credited (Eq. 3 credit assignment + EMA) by offspring performance each gen.
  * Foresight — an ``EvolutionaryNavigator`` that picks a regime + design directive
    from the incumbent history and population diversity.

TSP-GLS specifics (vs OBP): the generated heuristic is ``update_edge_distance``
(the GLS penalty update), scored by running Guided Local Search and returning the
tour cost (lower better; gap = (cost-opt)/opt vs a Concorde optimum). Evaluation is
CPU-bound single-threaded numba, so ``_track_cpu`` is on and the runner returns
``(cost, cpu_seconds)``.

Two design choices:
  * ``--fitness-mode`` (``--prompt-mode`` kept as alias) selects the SCALAR that
    drives HiFo's Hindsight/Foresight/parent-selection — it does NOT change the
    prompt text (HiFo prompts are score-free). ``partial_eval`` (default) uses a
    SHARED-instance mean cost (comparable across candidates, the racing-LLaMEA
    fair-scoring fix); ``original`` uses each candidate's plain ``mean_cost`` over
    its own (unequal) raced instances.
  * NESTED per-operator racing: each HiFo generation runs 5 sub-races, one per
    operator (e1->e2->m1->m2->m3). Each sub-race races the current elites + that
    operator's ``pop_size`` offspring and keeps the survivors, which feed the next
    operator (steady-state — the race REPLACES HiFo's per-operator
    ``population_management``). So a generation produces ``n_op x pop_size``
    offspring (heuristics ``g<gen>_c0..c{n_op*pop_size-1}``), records ONE incumbent
    (from the final m3-race survivors), and Hindsight/Foresight update after each
    sub-race.

The race selection + logged trajectory stay identical to ``racing/eoh_tsp_gls`` (the
fitness-mode only steers the LLM-facing search, never the F-race test/incumbent).
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
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages" / "LLM4AD"))
sys.path.insert(0, str(ROOT / "packages" / "HiFo-Prompt" / "hifo" / "src"))

import numpy as np
from dotenv import load_dotenv

from utils import (CachedLLM, ConfigAS, OpenRouterClient, GoogleClient, OllamaClient,
                   MistralClient, vLLMClient, make_wandb_logger)
from utils.llm import OpenRouterLLM4AD, OllamaLLM4AD, MistralLLM4AD, vLLMLLM4AD, GoogleLLM4AD
from utils.logger import make_log_dir
from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.task.optimization.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation_wo_Time
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData

from racing.base import RacingBase, CandidateRecord, _Tee, save_run_log, _run_tag
# Reuse the TSP-GLS racing scorers / (instance, seed) task machinery verbatim so the
# evaluation is byte-for-byte identical to racing/eoh_tsp_gls.
from racing.eoh_tsp_gls import (SeededInstance, _unwrap, score_tsp_inst,
                                score_tsp_config, _opt_cost, BIG_PENALTY,
                                REJECT_COST, _SEED_MAX)

# HiFo-Prompt components (standalone: prompts, Hindsight pool, Foresight navigator).
import hifo.methods.hifo.hifo_evolution as _hifo_evolution
# The prompt builders (Evolution.get_prompt_*) never touch the LLM interface, but
# Evolution.__init__ constructs one — stub it so we can reuse the builders offline.
_hifo_evolution.InterfaceLLM = lambda *a, **k: None
from hifo.methods.hifo.hifo_evolution import Evolution as _HiFoEvolution
from hifo.methods.hifo.insight_pool import InsightPool
from hifo.methods.hifo.evolutionary_navigator import EvolutionaryNavigator
import hifo.methods.hifo.hifo_hp as _hifo_hp

_OPERATORS = ["e1", "e2", "m1", "m2", "m3"]
# HiFo evaluates this many pop_size batches for the initial population
# (hifo population_generation n_create=2); used only to size the budget cap so it
# matches the HiFo generation structure (init + n_op batches/generation).
_N_INIT_BATCHES = 2


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

class _TSPPrompts:
    """GetPrompts-compatible prompt spec for TSP-GLS. IDENTICAL to the ``_TSPPrompts``
    in ``reprod/hifo_tsp_gls`` (duplicated here so racing carries no reprod import —
    keep the two byte-for-byte in sync). The generated function is
    ``update_edge_distance`` (the GLS penalty-matrix update). ``func_name``
    (``update_edge_distance``) differs from ``func_outputs`` (``updated_edge_distance``)
    as HiFo's code reconstruction requires."""

    def __init__(self):
        self.prompt_task = (
            "Given an edge distance matrix and a local optimal route, design a strategy "
            "to update the distance matrix so that Guided Local Search escapes the local "
            "optimum and ultimately finds a shorter tour. Create a heuristic that updates "
            "the edge distance matrix."
        )
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ["edge_distance", "local_opt_tour", "edge_n_used"]
        self.prompt_func_outputs = ["updated_edge_distance"]
        self.prompt_inout_inf = (
            "'edge_distance' is a Numpy matrix of pairwise edge distances; "
            "'local_opt_tour' is a Numpy array of node IDs giving the current "
            "local-optimum tour; 'edge_n_used' is a Numpy matrix counting how often "
            "each edge was used across the search. Return 'updated_edge_distance', a "
            "Numpy matrix the same shape as 'edge_distance' holding the updated "
            "(penalised) distances."
        )
        self.prompt_other_inf = (
            "All inputs and outputs are Numpy arrays. Keep the function signature, "
            "inputs and outputs unchanged, avoid division-by-zero on 'edge_n_used', and "
            "include 'import numpy as np' at the top of the code."
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
    "fitness_mode": "partial_eval",
    # --- Layer B: HiFo method internals (paper defaults; see hifo_hp.py) ----- #
    "pool_capacity": 30,
    "novelty_threshold": 0.7,
    "selection_count": 3,
    "usage_penalty_weight": 0.1,
    "recency_bonus": 0.2,
    "recency_window": 2,
    "ema_alpha": 0.3,
    "decay_rate": 0.01,
    "probation_usage": 3,
    "credit_best": 0.8,
    "credit_inc": 0.2,
    "credit_pen": -0.3,
    "progress_eps": 1e-4,
    "stagnation_threshold": 3,
    "progress_threshold": 2,
    "diversity_threshold": 0.3,
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
    "fitness_mode": "fm",
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


# --------------------------------------------------------------------------- #
# RacingHiFo
# --------------------------------------------------------------------------- #


class RacingHiFo(RacingBase):
    """HiFo-Prompt outer loop with a Friedman race as fitness + selection.

    Same race/eval as RacingEoH; the difference is candidate GENERATION (HiFo
    composite prompts) and the per-generation Hindsight/Foresight updates."""

    _pool_runner = staticmethod(score_tsp_config)
    _big_penalty = BIG_PENALTY
    _track_cpu = True   # score_tsp_config returns (cost, cpu); log the CPU odometer

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
        problem_size=100,
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
        fitness_mode: str = "partial_eval",
        wandb_logger=None,
    ):
        # irace-style (instance, seed) stochastic tasks (see racing/eoh_obp).
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
        self.score_one = score_one
        self.llm = llm
        self.selection_num = selection_num
        self.opt_by_idx = opt_by_idx or {}
        self.mean_opt = mean_opt
        self.problem_size = problem_size
        self.fitness_mode = str(fitness_mode)
        self._eval_timeout = (
            float(eval_timeout) if eval_timeout and float(eval_timeout) > 0 else None
        )
        self._big_penalty = float(timeout_cost)

        self._perf_log_path = self.log_dir / "instance_seed_perf.jsonl"
        self._perf_logged: set = set()
        self._shared_insts: list = []
        # Per-LLM-call prompt log (mirrors racing/llamea_bbob's llm_prompts.jsonl).
        self._prompt_log_path = self.log_dir / "llm_prompts.jsonl"
        self._llm_call_idx = 0
        self._prompt_lock = threading.Lock()
        # Hindsight insight-distillation prompts/responses (separate file).
        self._insight_log_path = self.log_dir / "insight_extraction.jsonl"
        self._insight_call_idx = 0

        self.template_str = evaluation.template_program
        self.task_desc = evaluation.task_description
        self.sampler = EoHSampler(llm, self.template_str)

        # HiFo prompt builders + Hindsight + Foresight. Evolution only supplies the
        # get_prompt_* string builders (its InterfaceLLM is stubbed above).
        self._evo = _HiFoEvolution(
            api_endpoint="bridged", api_key="bridged", model_LLM="bridged",
            llm_use_local=False, llm_local_url=None, debug_mode=False,
            prompts=_TSPPrompts(),
        )
        self.insight_pool = InsightPool(max_size=_hifo_hp.POOL_CAPACITY)
        self.navigator = EvolutionaryNavigator()
        # Foresight state (fed to navigator.get_guidance; updated post-race).
        self._best_hist: list = []
        self._avg_hist: list = []
        self._div_hist: list = []

        self._best_score = float("-inf")
        self._best_gap = float("inf")
        self._needs_pre_eval_cleanup = False

    # ---- irace-style (instance, seed) task pool (mirror racing/eoh_obp) ----

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
        if self._deterministic or self._base_instances is None:
            return
        need = next_instance + max(1, self.elitist_new_instances) - 1
        while len(self.instances) < need:
            self.instances = self.instances + self._build_rep(self._base_instances)
            print(f"  [seed-pool] extended to {len(self.instances)} tasks "
                  f"({self._n_reps} reps of {len(self._base_instances)} base "
                  f"instances) — need {need}", flush=True)

    def _task_meta(self, task_idx: int) -> tuple:
        task = self.instances[task_idx - 1]
        if isinstance(task, SeededInstance):
            return task.base_idx, task.seed, task.rep
        return task_idx - 1, None, 0

    def _instance_label(self, inst_idx: int) -> str:
        base_idx, seed, _rep = self._task_meta(inst_idx)
        s = seed if seed is not None else self.seed
        return f"i{base_idx}_s{s}"

    def _append_perf_log(self, gen_id: int, race_records, op: str | None = None) -> None:
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
                    # `op` identifies the sub-race. HiFo races 5 times per
                    # generation (e1..m3), so `gen` alone does not scope a
                    # candidate to one race -- without this the replay pools
                    # candidates that never competed against each other.
                    "gen": int(gen_id), "op": (str(op) if op is not None else None),
                    "cand_id": r.cfg.id,
                    "task_idx": int(task_idx), "base_idx": int(base_idx),
                    "instance_idx_k": extras["instance_idx_k"], "race_step": extras["race_step"],
                    "rep": int(rep), "seed": (int(seed) if seed is not None else None),
                    "cost": (c if math.isfinite(c) else str(c)),
                    "status": extras["status"], "wall_s": extras["wall_s"], "ts": ts,
                }))
        if lines:
            with open(self._perf_log_path, "a") as f:
                f.write("\n".join(lines) + "\n")

    # ---- Pre-evaluation of initial population (mirror racing/eoh_obp) ------

    def _pre_evaluate(self, init_recs: List[CandidateRecord]) -> None:
        n_sample = min(self.t_first, len(self.instances))
        if n_sample == 0 or not init_recs:
            return
        idxs = self.rng.choice(len(self.instances), size=n_sample, replace=False).tolist()
        sampled = [self.instances[i] for i in idxs]
        print(f"  [pre-eval] evaluating {len(init_recs)} candidates on "
              f"{n_sample} randomly sampled instances (indices {idxs})")
        t0 = time.time()
        for rec in init_recs:
            costs = []
            try:
                fn = rec.cfg.callable
            except Exception as exc:
                print(f"    [pre-eval] {rec.cfg.id} callable error — {exc}; mean_cost stays inf")
                continue
            for inst in sampled:
                cpu0 = time.process_time()
                costs.append(self.score_one(fn, inst))   # TSP: score_tsp_inst(fn, inst)
                self._cpu_seconds_used += time.process_time() - cpu0  # GLS CPU -> odometer
                self.budget_used += 1
            valid = [c for c in costs if math.isfinite(c) and c < BIG_PENALTY]
            rec.mean_cost = float(np.mean(valid)) if valid else float("inf")
            rec._fair_cost = rec.mean_cost   # no shared set yet
            print(f"    [pre-eval] {rec.cfg.id}: mean_cost={rec.mean_cost:.4f} "
                  f"({len(valid)}/{n_sample} valid evals)", flush=True)
        print(f"  [pre-eval] done in {time.time()-t0:.1f}s  budget_used={self.budget_used}")
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
            print("  [pre-eval cleanup] cfg histories and mean_costs reset "
                  f"for {len(records)} candidates before first race", flush=True)
        return super()._race(records, next_instance)

    # ---- Abstract implementations ----------------------------------------

    def _make_runner(self):
        def runner(params, instance):
            # Return (cost, cpu_seconds) so the sequential race path (num_cores=1)
            # feeds the CPU odometer just like the pooled path (score_tsp_config) does.
            cpu0 = time.process_time()
            cost = float(self.score_one(params, instance))
            return cost, time.process_time() - cpu0
        return runner

    def _materialize(self, func, idx: int, gen: int = 0) -> CandidateRecord:
        prog = TextFunctionProgramConverter.function_to_program(func, self.template_str)
        src = str(prog)
        within = self._gen_cand_counter.get(gen, 0)
        self._gen_cand_counter[gen] = within + 1
        tag = "p" if gen == 0 else "c"
        cand_id = f"g{gen}_{tag}{within}"
        cfg = ConfigAS(id=cand_id, source=src, entry_point=func.name,
                       name=f"{func.name}#{idx}", global_ns={"np": np})
        self._heuristics.append({"cand_id": cand_id, "gen_id": gen, "source": src})
        rec = CandidateRecord(cfg=cfg, candidate=func)
        rec._fair_cost = float("inf")
        # Insights this candidate was generated with (for post-race credit).
        rec._insights = list(getattr(func, "_hifo_insights", []) or [])
        return rec

    # ---- HiFo prompt construction + sampling -----------------------------

    def _parent_dict(self, rec: CandidateRecord) -> dict:
        return {"algorithm": (getattr(rec.candidate, "algorithm", "") or ""),
                "code": rec.cfg.source}

    def _parent_score(self, rec: CandidateRecord) -> float:
        """Scalar used for HiFo parent selection — mode-gated (see class doc)."""
        if self.fitness_mode == "partial_eval":
            fc = getattr(rec, "_fair_cost", float("inf"))
            if math.isfinite(fc):
                return fc
        return rec.mean_cost

    def _select_parents(self, pop, k):
        feasible = [r for r in pop if math.isfinite(self._parent_score(r))]
        if not feasible:
            return [self.rng.choice(pop) for _ in range(k)] if pop else []
        ranked = sorted(feasible, key=self._parent_score)
        p = np.array([1.0 / (i + len(ranked)) for i in range(len(ranked))])
        p = p / p.sum()
        idx = self.rng.choice(len(ranked), size=min(k, len(ranked)), replace=False, p=p)
        return [ranked[i] for i in idx]

    def _guidance(self):
        """Navigator regime + directive from the CURRENT Foresight history."""
        try:
            return self.navigator.get_guidance(
                pop=None,
                best_fitness_history=self._best_hist,
                avg_fitness_history=self._avg_hist,
                diversity_history=self._div_hist,
            )
        except Exception:
            return "balanced", None

    def _build_prompt(self, op, indivs, insights, directive, regime):
        if op == "i1":
            return self._evo.get_prompt_i1(insights, directive, regime)
        if op == "e1":
            return self._evo.get_prompt_e1(indivs, insights, directive, regime)
        if op == "e2":
            return self._evo.get_prompt_e2(indivs, insights, directive, regime)
        if op == "m1":
            return self._evo.get_prompt_m1(indivs[0], insights, directive, regime)
        if op == "m2":
            return self._evo.get_prompt_m2(indivs[0], insights, directive, regime)
        if op == "m3":
            return self._evo.get_prompt_m3(indivs[0], insights, directive, regime)
        raise ValueError(op)

    def _sample_raw(self, op, prompt, insights, gen):
        """One LLM call over a PREBUILT prompt -> (func with .algorithm +
        ._hifo_insights) or None."""
        t0 = time.time()
        try:
            thought, func = self.sampler.get_thought_and_function(prompt)
        except Exception as e:
            dt = time.time() - t0
            print(f"    [sample/{op}] exception after {dt:.2f}s: {type(e).__name__}: {e}")
            self._timings["samples"].append({"op": op, "gen": gen, "dt": dt,
                                             "status": "exception",
                                             "error": f"{type(e).__name__}: {e}"})
            return None
        dt = time.time() - t0
        if func is None:
            print(f"    [sample/{op}] returned no function ({dt:.2f}s)")
            self._timings["samples"].append({"op": op, "gen": gen, "dt": dt,
                                             "status": "parse_fail"})
            return None
        print(f"    [sample/{op}] LLM call ok  ({dt:.2f}s)")
        self._timings["samples"].append({"op": op, "gen": gen, "dt": dt, "status": "ok"})
        func.algorithm = thought or ""
        func._hifo_insights = list(insights or [])
        return func

    def _log_llm_prompt(self, op, gen, parents, insights, directive, regime, prompt) -> None:
        """Append one JSON record per LLM call to ``llm_prompts.jsonl`` (mirrors
        racing/llamea_bbob), so the exact context each offspring was sampled from is
        reproducible. HiFo-specific fields: the operator, the Foresight regime +
        design directive, and the injected Hindsight insights. HiFo prompts show the
        parents' code/description but NOT per-candidate scores, so
        ``scores_exposed=False`` (the parents' fair/mean costs are still logged here
        for debugging, they are just not shown to the LLM)."""
        def _num(x):
            return x if isinstance(x, (int, float)) and math.isfinite(x) else str(x)
        parents_shown = []
        for p in parents:
            parents_shown.append({
                "cand_id": p.cfg.id,
                "desc": (getattr(p.candidate, "algorithm", "") or "")[:120],
                "fair_cost": _num(getattr(p, "_fair_cost", float("inf"))),
                "mean_cost": _num(p.mean_cost),
                "coverage": len(p.cfg.costs_by_inst),
            })
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "gen": gen,
            "operator": op,
            "fitness_mode": self.fitness_mode,
            "scores_exposed": False,   # HiFo prompts expose parent code/desc, not scores
            "regime": regime,
            "design_directive": directive,
            "insights": list(insights or []),
            "shared_instance_ids": list(getattr(self, "_shared_insts", []) or []),
            "shared_instance_count": len(getattr(self, "_shared_insts", []) or []),
            "parent_cand_ids": [p.cfg.id for p in parents],
            "parents_shown": parents_shown,
            "prompt": prompt,
            # Token usage for this call, read per-thread so threaded sampling
            # (num_threads>1) cannot misattribute another thread's counts.
            # Cost is derived from these in post-processing.
            **_usage_fields(self.sampler),
        }
        with self._prompt_lock:
            rec["call_idx"] = self._llm_call_idx
            self._llm_call_idx += 1
            with open(self._prompt_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _sample_one(self, elites, op, gen):
        """Sample one offspring for a FIXED operator (thread-safe pieces locked)."""
        with self._sample_lock:
            k = self.selection_num if op in ("e1", "e2") else 1
            parents = self._select_parents(elites, k) if op != "i1" else []
            indivs = [self._parent_dict(p) for p in parents]
            regime, directive = self._guidance()
            insights = self.insight_pool.get_tips(k=_hifo_hp.SELECTION_COUNT)
        prompt = self._build_prompt(op, indivs, insights, directive, regime)
        self._log_llm_prompt(op, gen, parents, insights, directive, regime, prompt)
        func = self._sample_raw(op, prompt, insights, gen)
        if func is None:
            return None
        # candidate_log (base) reads these off the func; set them so operator /
        # lineage are populated (parents are records -> use their cfg ids).
        func.operator = op
        func.parent_ids = [p.cfg.id for p in parents] or None
        with self._sample_lock:
            idx = self._sample_idx
            self._sample_idx += 1
            rec = self._materialize(func, idx, gen=gen)
        return rec

    def _sample_op_batch(self, elites, gen_id, op):
        """pop_size offspring for ONE operator in HiFo generation ``gen_id``.

        cand_ids are ``g{gen_id}_c{cumulative}`` — the per-generation counter
        accumulates across the 5 operators, so a full sweep yields
        ``g{gen_id}_c0 .. g{gen_id}_c49``. Threaded via the sampler pool."""
        batch: List[CandidateRecord] = []
        max_attempts = 4 * self.pop_size
        if self._sampler_pool is None:
            attempts = 0
            while len(batch) < self.pop_size and attempts < max_attempts:
                attempts += 1
                rec = self._sample_one(elites, op, gen_id)
                if rec is not None:
                    batch.append(rec)
            return batch
        import concurrent.futures as _cf
        attempts = 0
        while len(batch) < self.pop_size and attempts < max_attempts:
            need = self.pop_size - len(batch)
            bs = max(1, min(self.num_threads, need, max_attempts - attempts))
            futs = []
            for _ in range(bs):
                attempts += 1
                futs.append(self._sampler_pool.submit(self._sample_one, elites, op, gen_id))
            for fut in _cf.as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"    [gen {gen_id}/{op}] sampler thread exc: {type(e).__name__}: {e}")
                    rec = None
                if rec is not None and len(batch) < self.pop_size:
                    batch.append(rec)
        return batch

    # ABC-required hooks; unused because we override _evolve (the NESTED mapping
    # runs 5 per-operator sub-races INSIDE one generation). Safe fallbacks.
    def _sample_offspring_seq(self, elites, gen, max_attempts):
        return self._sample_op_batch(elites, gen, _OPERATORS[0])

    def _sample_one_threaded(self, elites, gen, attempt_no):
        return self._sample_one(elites, _OPERATORS[0], gen)

    def _raw_from_source(self, source: str):
        """Compile a fixed-init heuristic source into the EoH func form (--fix-init-pop)."""
        return TextFunctionProgramConverter.text_to_function(source)

    def _sample_init(self, gen: int = 0):
        with self._sample_lock:
            regime, directive = self._guidance()
            insights = self.insight_pool.get_tips(k=_hifo_hp.SELECTION_COUNT)
        prompt = self._build_prompt("i1", [], insights, directive, regime)
        self._log_llm_prompt("i1", gen, [], insights, directive, regime, prompt)
        return self._sample_raw("i1", prompt, insights, gen)

    # ---- Hindsight / Foresight updates (post-race, once per generation) ----

    def _rescore_shared(self, records) -> None:
        """Set ``rec._fair_cost`` for every record — mode-gated.

        ``partial_eval``: mean cost over the instances ALL alive survivors SHARE
        (intersection of ``costs_by_inst``) — comparable across candidates, reusing
        race evals already paid for. ``original``: each candidate's own ``mean_cost``.
        A record with no shared coverage falls back to its ``mean_cost``."""
        if self.fitness_mode != "partial_eval":
            for r in records:
                r._fair_cost = r.mean_cost
            self._shared_insts = []
            return
        survivors = [r for r in records if r.survived
                     and any(math.isfinite(v) for v in r.cfg.costs_by_inst.values())]
        inst_sets = [set(r.cfg.costs_by_inst.keys()) for r in survivors]
        shared = set.intersection(*inst_sets) if inst_sets else set()
        self._shared_insts = sorted(shared)
        for r in records:
            keys = [k for k in shared if k in r.cfg.costs_by_inst]
            vals = [r.cfg.costs_by_inst[k] for k in keys
                    if math.isfinite(r.cfg.costs_by_inst[k])
                    and r.cfg.costs_by_inst[k] < BIG_PENALTY]
            if vals:
                r._fair_cost = float(np.mean(vals))
            else:
                r._fair_cost = r.mean_cost if math.isfinite(r.mean_cost) else float("inf")

    def _print_fitness_debug(self, gen_id, op, records) -> None:
        """Debug: print the scalar each candidate contributes to Hindsight/Foresight/
        parent-selection under the active fitness mode. ``used(fair)`` is what the
        search actually optimizes on (``_fair_cost``); ``own(mean)`` is the candidate's
        mean over ITS OWN raced instances. In ``partial_eval`` the two differ when a
        candidate's coverage extends beyond the shared set (fair == shared-instance
        mean); in ``original`` they are identical (fair == own mean) by construction —
        which is exactly the fair-vs-unfair contrast to inspect."""
        def _f(x):
            return f"{x:8.3f}" if isinstance(x, (int, float)) and math.isfinite(x) else "     inf"
        shared_n = len(getattr(self, "_shared_insts", []) or [])
        print(f"    [fitness/{self.fitness_mode}] gen {gen_id} op {op} | "
              f"shared_insts={shared_n} | scalar fed to Hindsight/Foresight/parent-sel "
              f"(used=fair_cost, own=mean over own instances):")
        for r in sorted(records, key=lambda x: getattr(x, "_fair_cost", float("inf"))):
            fc = getattr(r, "_fair_cost", float("inf"))
            mc = r.mean_cost
            diff = ("  <-- fair!=own" if (math.isfinite(fc) and math.isfinite(mc)
                                          and abs(fc - mc) > 1e-9) else "")
            print(f"        {r.cfg.id:<10s} used={_f(fc)}  own={_f(mc)}  "
                  f"cov={len(r.cfg.costs_by_inst):<3d} {'surv' if r.survived else 'elim'}{diff}")

    def _credit_effectiveness(self, off_obj, pop_objs) -> float:
        """HiFo Eq. 3 credit (objective = mean bins, lower better)."""
        if off_obj is None or not math.isfinite(off_obj):
            return -0.5
        valid = [o for o in pop_objs if o is not None and math.isfinite(o)]
        if not valid:
            return 0.0
        best, worst = min(valid), max(valid)
        avg = sum(valid) / len(valid)
        if worst == best:
            return 0.1
        norm = (worst - off_obj) / (worst - best)
        if off_obj <= best:
            eff = _hifo_hp.CREDIT_BEST + 0.2 * norm
        elif off_obj <= avg:
            eff = _hifo_hp.CREDIT_INC + 0.6 * norm
        else:
            eff = _hifo_hp.CREDIT_PEN + 0.5 * norm
        return max(-1.0, min(1.0, eff))

    def _update_hindsight(self, op_offspring, records) -> None:
        """Credit the insights each just-generated offspring used, by its fair-cost
        relative to the current sub-race population (elites + this operator's batch)."""
        pop_objs = [getattr(r, "_fair_cost", float("inf")) for r in records]
        for r in op_offspring:
            insights = getattr(r, "_insights", None) or []
            if not insights:
                continue
            eff = self._credit_effectiveness(getattr(r, "_fair_cost", float("inf")), pop_objs)
            for tip in insights:
                self.insight_pool.update_tip_stats(tip, eff)

    def _update_foresight(self, gen, records) -> None:
        """Advance the navigator's history (incumbent trend + textual diversity)."""
        survivors = [r for r in records if r.survived]
        objs = [getattr(r, "_fair_cost", float("inf")) for r in survivors]
        valid = [o for o in objs if math.isfinite(o)]
        if valid:
            self._best_hist.append(min(valid))
            self._avg_hist.append(sum(valid) / len(valid))
            for h in (self._best_hist, self._avg_hist):
                if len(h) > 50:
                    del h[:-50]
        # Phenotypic diversity: fraction of survivor pairs with non-identical text
        # (HiFo Eq. 7; update_population_metrics), using the algorithm description.
        texts = [(getattr(r.candidate, "algorithm", "") or r.cfg.source) for r in survivors]
        if len(texts) >= 2:
            diff = sum(1 for i in range(len(texts)) for j in range(i + 1, len(texts))
                       if texts[i] != texts[j])
            self._div_hist.append(diff / (len(texts) * (len(texts) - 1) / 2))
            if len(self._div_hist) > 50:
                del self._div_hist[:-50]
        self.insight_pool.update_generation(gen)

    def _extract_insights(self, gen_id, op, records) -> None:
        """Distil 1-2 new insights from the top survivors into the pool (one LLM
        call). Mirrors hifo_interface_EC.extract_insights_from_population. The
        distillation prompt + raw response + parsed/admitted insights are logged to
        ``insight_extraction.jsonl`` (separate from the generation prompts in
        ``llm_prompts.jsonl``)."""
        pop = [r for r in records if r.survived
               and math.isfinite(getattr(r, "_fair_cost", float("inf")))]
        if len(pop) < 3:
            return
        pop.sort(key=lambda r: getattr(r, "_fair_cost", float("inf")))
        top = pop[:max(1, int(len(pop) * 0.3))]
        prompt = ("The following are core descriptions of high-performance "
                  "optimization algorithms evolved recently:\n")
        for i, r in enumerate(top):
            desc = (getattr(r.candidate, "algorithm", "") or "").strip()
            if desc and len(desc) > 8:
                content = desc
            else:
                code = r.cfg.source
                content = (code[:800] + "...\n# (truncated for brevity)"
                           if len(code) > 1000 else code)
            prompt += f"{i+1}. Algorithm: {content}\n"
        prompt += ("\nPlease extract 1-2 concise, generic, and performance-positive "
                   "[design principles] or [effective patterns] from the above algorithms."
                   "\nThese principles should be applicable to various combinatorial "
                   "optimization problems, not just the specific problem domain."
                   "\nWhen formulating these principles, it is essential to draw insights "
                   "from *both* the conceptual natural language descriptions *and* their "
                   "corresponding code implementations. Focus on identifying the underlying "
                   "strategic design choices and algorithmic methodologies rather than "
                   "superficial characteristics or specific implementation minutiae."
                   "\nEach principle/pattern should be expressed as an independent sentence "
                   "in the following format:"
                   "\n- Balance local optimization with global solution structure when making decisions."
                   "\n- Prioritize choices that maintain flexibility for future decision-making steps."
                   "\n- Implement adaptive mechanisms that respond to problem instance characteristics.")
        response = None
        items: list = []
        added = 0
        error = None
        try:
            response = self.llm.draw_sample(prompt)
            items = [ln.strip()[1:].strip() for ln in str(response).split("\n")
                     if ln.strip().startswith("-")]
            for it in items:
                if it and len(it) > 10 and self.insight_pool.add_tip(
                        it, tags=["extracted", "high_performance"]):
                    added += 1
            print(f"    [hindsight] extracted {len(items)} insights, "
                  f"{added} admitted (pool={len(self.insight_pool.tips)})")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            print(f"    [hindsight] insight extraction failed: {error}")
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "gen": gen_id,
            "operator": op,          # the sub-race operator after which distillation ran
            "n_top": len(top),
            "top_cand_ids": [r.cfg.id for r in top],
            "prompt": prompt,
            "response": (str(response) if response is not None else None),
            "extracted": items,      # parsed candidate insights (pre-admission)
            "admitted": added,       # how many passed the Jaccard-novelty admission
            "pool_size_after": len(self.insight_pool.tips),
            "error": error,
        }
        with self._prompt_lock:
            rec["call_idx"] = self._insight_call_idx
            self._insight_call_idx += 1
            with open(self._insight_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _op_update(self, gen_id, op, op_offspring, records) -> None:
        """Per-operator Hindsight/Foresight update, run after EACH sub-race inside a
        HiFo generation (mirrors HiFo's per-operator get_algorithm updates):
        rescore the fair (shared-instance) cost, credit this operator's offspring,
        advance the Navigator history, and distil insights from the survivors. The
        rescored fair cost feeds the NEXT operator's parent selection (steady-state)."""
        self._rescore_shared(records)
        self._print_fitness_debug(gen_id, op, records)
        self._update_hindsight(op_offspring, records)
        self._update_foresight(gen_id, records)
        self._extract_insights(gen_id, op, records)

    # ---- Final eval + run (mirror racing/eoh_tsp_gls) --------------------

    def _final_eval(self) -> dict:
        """Validate BOTH incumbent rules -> valid_trajectory_mean_rank/mean_cost.json."""
        from analyses.eval_tsp import evaluate_single, _to_serialisable
        print("\n=== Post-Evolution Final Evaluation (eval_tsp.py) ===", flush=True)
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
        self._finalize_reliability_both(self._final_eval())
        return res

    def _evolve(self, elites, next_instance):
        """NESTED generation loop: ONE iteration == one HiFo generation, containing 5
        per-operator sub-races (e1->e2->m1->m2->m3). Each sub-race races the current
        elites + that operator's pop_size offspring and keeps the survivors, which
        feed the next operator (steady-state — the race REPLACES HiFo's per-operator
        population_management). Hindsight/Foresight update after each sub-race. The
        generation is logged ONCE, so heuristics are g{gen}_c0..g{gen}_c49 and there
        is exactly one incumbent per HiFo generation."""
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        g = 0
        while self.max_generations is None or g < self.max_generations:
            if self.budget_cap is not None and self.budget_used >= self.budget_cap:
                print(f"  [budget cap reached] {self.budget_used}/{cap_str} — stopping run")
                break
            gen_id = g + 1
            seen: dict = {r.cfg.id: r for r in elites}   # every record seen this generation
            all_offspring: List[CandidateRecord] = []
            last_out = None
            t_gen = time.time()
            dead_zone = False
            for op in _OPERATORS:
                if self.budget_cap is not None and self.budget_used >= self.budget_cap:
                    print(f"  [budget cap reached mid-generation] {self.budget_used}/{cap_str}")
                    break
                offspring = self._sample_op_batch(elites, gen_id, op)
                if not offspring:
                    print(f"    [gen {gen_id}/{op}] 0 offspring — skipping operator")
                    continue
                all_offspring.extend(offspring)
                combined = elites + offspring
                print(f"\n# Generation {gen_id} / operator {op} race — {len(combined)} "
                      f"configs ({len(elites)} elites + {len(offspring)} offspring)")
                self._cur_gen_id = gen_id   # for race_log.jsonl (hifo: 1 race per operator)
                self._cur_operator = op
                out = self._race(combined, next_instance)
                next_instance = out["next_instance"]
                survivors = out["survivors"]
                if self.save_pop and len(survivors) < self.pop_size:
                    elites = out["all_records_sorted"][: self.pop_size]
                else:
                    elites = survivors[: self.pop_size]
                for r in combined:
                    seen[r.cfg.id] = r
                last_out = out
                # Log THIS sub-race before the next operator overwrites `out`.
                # A HiFo generation runs one race per operator, but only the last
                # used to reach the logs, so 4 of every 5 races (their step traces,
                # p-values and eliminations) were silently discarded. Each trace is
                # tagged with `op` so the replay can tell the sub-races apart; the
                # perf rows are appended here too, for the same reason.
                try:
                    self._mlog.log_race_steps(
                        gen_id, out.get("step_trace", []),
                        instance_meta=self._instance_meta_eoh, op=op)
                except Exception as e:                      # never break the run
                    print(f"  [manuscript-log] WARN: gen {gen_id}/{op} step trace "
                          f"failed: {e}", flush=True)
                self._append_perf_log(gen_id, combined, op=op)
                # Dead-zone guard: an SH sub-race (sh/hifo_*) can spend 0 experiments when
                # the remaining budget is smaller than even round-0's required evaluations
                # (SHEoH._race breaks on `required > budget_remaining` before evaluating).
                # Then budget_used never advances, so the budget-cap checks above can never
                # fire and this generation loop spins forever. If a sub-race makes no
                # progress under a budget cap, the budget is effectively exhausted -> stop.
                # (No-op for the Friedman race, which never returns 0 experiments.)
                if self.budget_cap is not None and out.get("experiments_used", 0) == 0:
                    print(f"    [gen {gen_id}/{op}] sub-race spent 0 experiments "
                          f"(budget {self.budget_used}/{cap_str}: remaining < a round's "
                          f"cost) — budget effectively exhausted, stopping run", flush=True)
                    dead_zone = True
                    break
                # Per-operator Hindsight/Foresight: the rescored fair cost feeds the
                # NEXT operator's parent selection (steady-state).
                self._op_update(gen_id, op, offspring, combined)
            if dead_zone:
                break
            if not all_offspring:
                print(f"  gen {gen_id:02d}: 0 offspring across all operators — stopping run")
                break
            dt_gen = time.time() - t_gen
            combined_all = list(seen.values())
            self._post_generation_hook(elites)
            self._record_incumbent(gen_id, elites)          # once per HiFo generation
            self._record_generation(gen_id, combined_all,
                                    last_out.get("table_rows") if last_out else None)
            # log_steps=False: each sub-race already logged its own trace (with
            # `op`) inside the operator loop above.
            self._log_manuscript_eoh(gen_id, combined_all, all_offspring,
                                     last_out or {}, dt_gen, elites,
                                     log_steps=False)
            print(f"  gen {gen_id:02d}/{gen_str}: {len(all_offspring)} offspring across "
                  f"{len(_OPERATORS)} operators, elites_kept={len(elites)}, "
                  f"budget={self.budget_used}/{cap_str}, gen_wall={dt_gen:.1f}s")
            self._timings["generations"].append({
                "gen_id": gen_id, "n_offspring": len(all_offspring),
                "n_elites_kept": len(elites), "gen_wall": dt_gen,
            })
            self._save_timings()
            g += 1
        return elites

    # NESTED mapping: _evolve is overridden so ONE _evolve iteration == one HiFo
    # generation (5 internal per-operator sub-races). The base's _record_incumbent
    # (one incumbent per _evolve iteration) therefore already fires once per HiFo
    # generation, from that generation's final (m3) survivors — no override needed.

    def _compute_gap(self, rec: CandidateRecord) -> float:
        if (rec.n_evals == 0 or not math.isfinite(rec.mean_cost)
                or not rec.evaluated_idxs):
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
            if (r.n_evals > 0 and math.isfinite(r.mean_cost)
                    and score > self._best_score):
                self._best_score = score
                self._best_record = r
                if math.isfinite(gap):
                    self._best_gap = gap
            elif math.isfinite(gap) and gap < self._best_gap:
                self._best_gap = gap
            perfs.append({
                "cand_id": r.cfg.id,
                "num_eval_instances": int(r.n_evals),
                "instance_order": list(race_order),
                "instances_evaluated": list(r.evaluated_idxs),
                "score": score,
                "gap": float(gap),
                "survived": bool(r.survived),
            })
        survivors_snapshot = [
            {"score": (float(-r.mean_cost) if math.isfinite(r.mean_cost)
                       else float(-BIG_PENALTY))}
            for r in race_records if r.survived
        ]
        self.generations.append({
            "gen_id": int(gen_id),
            "performances": perfs,
            "population": survivors_snapshot,
            "used_budget": int(self.budget_used),
        })
        for row in table_rows or []:
            entry = {"gen_id": int(gen_id), **row}
            self.trajectory.append(entry)
            self.wandb_logger.log_trajectory_row(entry)
        save_run_log(self.log_dir, self.label, self.trajectory, self.final_eval,
                     incumbents_by_rule=self._incumbents_by_rule)
        self._save_heuristics()
        # NOTE: perf rows are appended per sub-race inside `_evolve` (tagged with
        # `op`), so this generation-level call would only re-scan already-logged
        # (cand_id, task_idx) keys. Kept as a no-op safety net for any record that
        # somehow never passed through a sub-race.
        self._append_perf_log(gen_id, race_records)
        # NOTE: HiFo Hindsight/Foresight is updated per operator sub-race in
        # `_evolve` (via `_op_update`), not here — `_record_generation` runs once per
        # HiFo generation and only logs.

    def _build_table_rows(self, race_out: dict, budget_before: int) -> list:
        rows = []
        for row in race_out.get("table_rows", []):
            mb = float(row["mean_best"])
            if self.mean_opt and self.mean_opt > 0 and math.isfinite(mb):
                gap = (mb - self.mean_opt) / self.mean_opt
            else:
                gap = float("inf")
            rows.append({
                "inst_idx": int(row["inst_idx"]),
                "cand_id": row["cand_id"],
                "score": mb,
                "gap": float(gap),
                "n_instances": int(row["n_instances"]),
                "used_budget": budget_before + int(row["experiments_used"]),
            })
        return rows

    def _run_header(self) -> str:
        cap_str = "off" if self.budget_cap is None else str(self.budget_cap)
        gen_str = "off" if self.max_generations is None else str(self.max_generations)
        return (f"[{self.label}] racing HiFo NESTED (fitness_mode={self.fitness_mode}, "
                f"{len(_OPERATORS)}x{self.pop_size} offspring/gen via per-operator "
                f"sub-races e1..m3, budget_cap={cap_str}, "
                f"max_generations={gen_str} (HiFo gens), T_first={self.t_first}, "
                f"T_each={self.t_each}, elitist={self.elitist})")

    def _save_heuristics(self) -> None:
        score_by_id = {}
        for gen in self.generations:
            for p in gen.get("performances", []):
                score_by_id[p["cand_id"]] = (
                    float(p.get("score")) if p.get("score") is not None else None)
        out = [{"cand_id": h["cand_id"], "gen_id": int(h["gen_id"]),
                "score": score_by_id.get(h["cand_id"]), "source": h["source"]}
               for h in self._heuristics]
        payload = {"label": self.label, "total_sampled": len(out), "heuristics": out}
        json.dump(payload, open(self.log_dir / "heuristics.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Racing HiFo-Prompt on TSP-GLS.")
    p.add_argument("--pop-size", type=int, default=10)
    p.add_argument("--max-generations", type=int, default=20,
                   help="HiFo-generation cap (each generation = 5 per-operator "
                        "sub-races). Use -1 to disable.")
    p.add_argument("--ref-max-generations", type=int, default=20,
                   help="Reference number of HiFo GENERATIONS for the budget cap "
                        "(each = n_op operator sub-races). Budget cap (when --budget-cap "
                        "is unset) = (2 init + ref x n_op) x pop_size x n_instances. NOTE: "
                        "this counts HiFo generations; for a same-budget head-to-head with "
                        "racing/eoh_tsp_gls set --budget-cap equal on both.")
    p.add_argument("--n-instances", type=int, default=64)
    p.add_argument("--problem-size", type=int, default=100,
                   help="Number of nodes per TSP instance.")
    p.add_argument("--eval-timeout", type=float, default=65.0,
                   help="Per-(candidate, instance) GLS wall-clock cap (s); a slow-but-"
                        "valid heuristic is killed and recorded as --timeout-cost.")
    p.add_argument("--timeout-cost", type=float, default=BIG_PENALTY,
                   help=f"Finite penalty for a heuristic killed at --eval-timeout. "
                        f"A crash returns Inf and is rejected. Default {BIG_PENALTY:g}.")
    p.add_argument("--selection-num", type=int, default=5)
    p.add_argument("--t-first", type=int, default=5)
    p.add_argument("--t-each", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--test-type", type=str, default="friedman", choices=["friedman", "ttest"])
    p.add_argument("--posthoc-test-type", type=str, default="conover",
                   choices=["conover", "nemenyi"])
    p.add_argument("--no-elitist", dest="elitist", action="store_false", default=True)
    p.add_argument("--elitist-new-instances", type=int, default=1)
    p.add_argument("--elitist-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-cap", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--fitness-mode", "--prompt-mode", dest="fitness_mode", type=str,
                   default="partial_eval", choices=["original", "partial_eval"],
                   help="Scalar driving HiFo's Hindsight/Foresight/parent-selection "
                        "(this does NOT change the prompt text — HiFo prompts are "
                        "score-free — only what the search optimizes on). "
                        "'partial_eval' (default): SHARED-instance mean cost (comparable "
                        "across candidates; the racing-LLaMEA fair-scoring fix). 'original': "
                        "each candidate's plain mean_cost over its own (unequal) raced "
                        "instances. Does NOT affect the F-race test or logged trajectory. "
                        "(--prompt-mode is kept as a backward-compat alias.)")
    p.add_argument("--label", type=str, default="race/hifo_tsp_gls")
    p.add_argument("--log-root", type=pathlib.Path, default=ROOT)
    p.add_argument("--cache-root", type=pathlib.Path, default=ROOT / ".llm_cache")
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    p.add_argument("--llm-max-tokens", type=int, default=2048)
    p.add_argument("--llm-backend", type=str, default="openrouter",
                   choices=["openrouter", "ollama", "mistral", "vllm", "google"])
    p.add_argument("--llm-model", type=str, default="qwen/qwen3-coder-next")
    p.add_argument("--ollama-host", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--num-cores", type=int, default=1)
    p.add_argument("--save-pop", action="store_true", default=False)
    p.add_argument("--fix-init-pop", action="store_true", default=False,
                   help="Seed generation 0 from src/init_pop/hifo_tsp_gls.json (2*pop_size fixed "
                        "heuristics, re-raced this run) instead of sampling from the LLM.")
    p.add_argument("--early-stopping-non-elitist", action="store_true", default=False)
    p.add_argument("--deal-with-crashed", type=str, default="rejection",
                   choices=["rejection", "penalty"])
    p.add_argument("--use-wandb", action="store_true", default=False)

    # --- Layer B: HiFo method internals (default == paper; injected via hifo_hp) #
    g_ip = p.add_argument_group("HiFo Insight Pool (Hindsight)")
    g_ip.add_argument("--pool-capacity", type=int, default=30)
    g_ip.add_argument("--novelty-threshold", type=float, default=0.7)
    g_ip.add_argument("--selection-count", type=int, default=3)
    g_ip.add_argument("--usage-penalty-weight", type=float, default=0.1)
    g_ip.add_argument("--recency-bonus", type=float, default=0.2)
    g_ip.add_argument("--recency-window", type=int, default=2)
    g_ip.add_argument("--ema-alpha", type=float, default=0.3)
    g_ip.add_argument("--decay-rate", type=float, default=0.01)
    g_ip.add_argument("--probation-usage", type=int, default=3)
    g_cr = p.add_argument_group("HiFo credit-assignment tiers (Eq. 3 intercepts)")
    g_cr.add_argument("--credit-best", type=float, default=0.8)
    g_cr.add_argument("--credit-inc", type=float, default=0.2)
    g_cr.add_argument("--credit-pen", type=float, default=-0.3)
    g_nav = p.add_argument_group("HiFo Evolutionary Navigator (Foresight)")
    g_nav.add_argument("--progress-eps", type=float, default=1e-4)
    g_nav.add_argument("--stagnation-threshold", type=int, default=3)
    g_nav.add_argument("--progress-threshold", type=int, default=2)
    g_nav.add_argument("--diversity-threshold", type=float, default=0.3)
    return p.parse_args(argv)


def _inject_hifo_hp(args) -> None:
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
                           tag=_run_tag(args, _BASH_DEFAULTS, _ABBREV))
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

    max_generations = (None if args.max_generations is not None and args.max_generations < 0
                       else args.max_generations)
    if args.budget_cap is None:
        # Size the budget by HiFo's generation STRUCTURE: _N_INIT_BATCHES*pop_size
        # candidates for the initial population plus n_op*pop_size per HiFo generation
        # (one pop_size batch per operator sub-race), each over n_instances. So
        # --ref-max-generations counts HiFo GENERATIONS (each = n_op sub-races), NOT
        # racing-generations. (racing/eoh_tsp_gls's pop_size*rmg*ni formula omits the
        # n_op operator batches + init; for a same-budget head-to-head set --budget-cap
        # equal on both instead of relying on ref.)
        cand_evals = (_N_INIT_BATCHES + args.ref_max_generations * len(_OPERATORS)) * args.pop_size
        budget_cap = cand_evals * args.n_instances
        print(f"budget_cap {budget_cap} = ({_N_INIT_BATCHES} init + "
              f"{args.ref_max_generations} HiFo-gen x {len(_OPERATORS)} ops) x "
              f"{args.pop_size} pop x {args.n_instances} instances")
    else:
        budget_cap = None if args.budget_cap < 0 else args.budget_cap
    if max_generations is None and budget_cap is None:
        print("error: at least one of --max-generations or --budget-cap must be >= 0",
              file=sys.stderr)
        return 2

    print(f"pop_size={args.pop_size}  "
          f"max_generations={'off' if max_generations is None else max_generations}  "
          f"n_instances={args.n_instances}  problem_size={args.problem_size}")
    print(f"offspring/gen = {len(_OPERATORS)} operators x {args.pop_size} = "
          f"{len(_OPERATORS) * args.pop_size} (NESTED: per-operator sub-races inside "
          f"each HiFo generation; one incumbent + heuristics g<gen>_c0..c{len(_OPERATORS)*args.pop_size-1} "
          f"per generation)   fitness_mode={args.fitness_mode}")
    print(f"budget_cap={'off' if budget_cap is None else budget_cap}  "
          f"(T_first={args.t_first}, T_each={args.t_each}, alpha={args.alpha})")

    if args.llm_backend == "ollama":
        client = OllamaClient(host=args.ollama_host, model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir); llm = OllamaLLM4AD(cached)
    elif args.llm_backend == "mistral":
        if not os.environ.get("MISTRAL_API_KEY"):
            print("MISTRAL_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = MistralClient(model=args.llm_model, timeout=args.llm_timeout)
        cached = CachedLLM(client, cache_dir=cache_dir); llm = MistralLLM4AD(cached)
    elif args.llm_backend == "vllm":
        client = vLLMClient(model=args.llm_model, timeout=args.llm_timeout, max_tokens=args.llm_max_tokens)
        cached = CachedLLM(client, cache_dir=cache_dir); llm = vLLMLLM4AD(cached)
    elif args.llm_backend == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = GoogleClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir); llm = GoogleLLM4AD(cached)
    else:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
            return 2
        client = OpenRouterClient(timeout=args.llm_timeout, x_title=args.label, model=args.llm_model)
        cached = CachedLLM(client, cache_dir=cache_dir); llm = OpenRouterLLM4AD(cached)
    print(f"Backend: {args.llm_backend}  model: {cached.client.model}  cache: {cache_dir}")

    args_dict = {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()}
    args_dict["llm_model"] = cached.client.model
    with open(log_dir / "args.yaml", "w") as _f:
        yaml.dump(args_dict, _f, default_flow_style=False, sort_keys=True)

    run_name = f"{args.label}_{dt_stamp.replace('/', '_')}_{args.seed}_{_run_tag(args, _BASH_DEFAULTS, _ABBREV)}"
    from utils.manuscript_log import write_run_meta as _wrm
    _wrm(log_dir, {
        "run_id": run_name, "framework": "HiFo-Prompt", "domain": "TSP-GLS",
        "policy": ("raceAD" if getattr(args, "elitist", True) else "race-nonelitist"),
        "seed": args.seed, "M": getattr(args, "pop_size", None),
        "N": getattr(args, "n_instances", None), "K": getattr(args, "n_instances", None),
        "alpha": getattr(args, "alpha", None), "test_type": getattr(args, "test_type", None),
        "posthoc": getattr(args, "posthoc_test_type", None),
        "T_first": getattr(args, "t_first", None), "T_each": getattr(args, "t_each", None),
        "elimit": getattr(args, "elitist_limit", None),
        "refill": bool(getattr(args, "save_pop", getattr(args, "save_diversity", False))),
        "incumbent_rule": "both",
        "B": budget_cap, "llm_model": cached.client.model,
    })
    wandb_logger = make_wandb_logger(enabled=args.use_wandb, project="llm4ad",
                                     name=run_name, config=args_dict)

    _inject_hifo_hp(args)

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

    _racer = RacingHiFo(
        evaluation=evaluation, score_one=score_tsp_inst, instances=instances,
        label=args.label, log_dir=log_dir, llm=llm, pop_size=args.pop_size,
        max_generations=max_generations, selection_num=args.selection_num,
        budget_cap=budget_cap, t_first=args.t_first, t_each=args.t_each, alpha=args.alpha,
        seed=args.seed, opt_by_idx=opt_by_idx, mean_opt=mean_opt, problem_size=args.problem_size,
        num_threads=args.num_threads, num_cores=args.num_cores,
        eval_timeout=args.eval_timeout, timeout_cost=args.timeout_cost,
        test_type=args.test_type, posthoc_test_type=args.posthoc_test_type,
        elitist=args.elitist, elitist_new_instances=args.elitist_new_instances,
        elitist_limit=args.elitist_limit, save_pop=args.save_pop,
        early_stopping_non_elitist=args.early_stopping_non_elitist,
        deal_with_crashed=args.deal_with_crashed,
        deterministic=args.deterministic, fitness_mode=args.fitness_mode,
        wandb_logger=wandb_logger,
    )
    if args.fix_init_pop:
        from utils.fixed_init_pop import load_fixed_initial_population
        _fp = ROOT / "src" / "init_pop" / "hifo_tsp_gls.json"
        # HiFo's initial batch is 2*pop_size (n_init_batches=2); enter all of them
        # into the first F-race (they get pruned by racing).
        _racer._fixed_init_sources = [
            h["source"] for h in load_fixed_initial_population(_fp, 2 * _racer.pop_size)]
        print(f"  [fixed-init-pop] loaded {len(_racer._fixed_init_sources)} "
              f"heuristics from {_fp}", flush=True)
    _racer.run()

    print(f"logs -> {log_dir}")
    print(f"total runtime: {time.time() - main_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
