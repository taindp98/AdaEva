"""Structured logging for the racing runs.

Emits the JSONL/JSON artifacts below, so every paper figure is a pure
post-hoc reduction of logged facts (no figure needs a re-run):

    race_step_trace.jsonl        - per instance-step within a race (Dims 2,3,8)
    race_summary.jsonl           - per generation/race rollup     (Dims 1,2,3)
    candidate_log.jsonl          - candidate lifecycle/genealogy   (Dims 1,2,4)
    population_diversity_log.jsonl - per-generation health         (Dim  4)
    fitness_reliability_log.jsonl  - partial vs. full score        (Dim  6)  [end-of-run]

The LLM-context log (``llm_prompts.jsonl``) and the full-suite validation
(``valid_trajectory.json``) are written by the runners themselves; this module
only covers the artifacts that are serialisations of race/population state.

All writers are append-only, best-effort and never raise into the search loop:
a logging failure must not abort a run.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tokenize
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional


def _json_default(o):
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)                       # inf / -inf / nan -> JSON-safe string
    try:
        return float(o)
    except Exception:
        return str(o)


def _num(x):
    """JSON-safe number: finite float stays a float, non-finite becomes a string."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


def _sanitize(obj):
    """Recursively replace non-finite floats (inf/-inf/nan) with strings so the
    emitted JSON is strictly valid (no bare Infinity/NaN tokens)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Token-level code diversity (cheap; no AST)
# --------------------------------------------------------------------------- #

def _tokens(src: str) -> list[str]:
    """Python token strings, ignoring layout/comment tokens. Falls back to a
    whitespace split if the source does not tokenise (e.g. a partial fence)."""
    skip = {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER}
    try:
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(src or "").readline):
            if tok.type in skip:
                continue
            s = tok.string.strip()
            if s:
                out.append(s)
        return out or (src or "").split()
    except Exception:
        return (src or "").split()


def code_hash(src: str) -> str:
    """Stable 12-hex token-level hash (whitespace/comment-insensitive)."""
    return hashlib.sha1(" ".join(_tokens(src)).encode("utf-8", "replace")).hexdigest()[:12]


def mean_pairwise_token_distance(sources: Iterable[str]) -> float:
    """Mean pairwise Jaccard distance (1 - |A∩B|/|A∪B|) over the token SETS of
    ``sources``. 0.0 for < 2 usable sources. A cheap genotype-diversity proxy."""
    tsets = [set(_tokens(s)) for s in sources if s]
    tsets = [t for t in tsets if t]
    if len(tsets) < 2:
        return 0.0
    dists = []
    for a, b in combinations(tsets, 2):
        union = a | b
        dists.append(1.0 - (len(a & b) / len(union)) if union else 0.0)
    return float(sum(dists) / len(dists)) if dists else 0.0


# --------------------------------------------------------------------------- #
# run_meta.json — the per-run join / reproducibility record
# --------------------------------------------------------------------------- #

def perf_row_extras(cfg, task_idx, cost, order, big_penalty: float = 1e5) -> dict:
    """Shared instance_seed_perf extras: ``status`` (ok|crash|timeout),
    ``instance_idx_k`` (1-based prefix position in the race's instance order), plus
    eval-time ``race_step`` and ``wall_s`` from ``cfg.meta_by_inst``. ``order`` is the
    race's instance order (0-based task indices). Module-level so both RacingBase
    subclasses and the standalone LLaMEA runner can use it."""
    timed = getattr(cfg, "timed_out_insts", None) or set()
    c = float(cost)
    status = ("timeout" if task_idx in timed
              else "crash" if (not math.isfinite(c)) or c >= float(big_penalty)
              else "ok")
    try:
        instance_idx_k = list(order).index(task_idx - 1) + 1
    except (ValueError, AttributeError):
        instance_idx_k = None
    m = (getattr(cfg, "meta_by_inst", None) or {}).get(task_idx) or {}
    return {
        "instance_idx_k": instance_idx_k,
        "race_step": m.get("race_step"),
        "status": status,
        "wall_s": m.get("wall_s"),
    }


def write_run_meta(log_dir, meta: dict) -> None:
    """Write ``run_meta.json`` — the record every cross-cell join, filter, and repro
    claim keys on. ``meta`` carries the run's identity + settled hyperparameters
    (framework/domain/policy/seed/M/N/alpha/... /B/K/llm_model). ``git_sha`` and
    ``config_hash`` are computed here if absent. Best-effort; never raises into the run.
    """
    import subprocess
    out = dict(meta)
    d = Path(log_dir)
    if out.get("git_sha") is None:
        try:
            out["git_sha"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(d),
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            out["git_sha"] = None
    if out.get("config_hash") is None:
        try:
            payload = json.dumps({k: v for k, v in out.items()
                                  if k not in ("git_sha", "config_hash", "run_id")},
                                 sort_keys=True, default=str)
            out["config_hash"] = hashlib.sha1(
                payload.encode("utf-8", "replace")).hexdigest()[:12]
        except Exception:
            out["config_hash"] = None
    try:
        with open(d / "run_meta.json", "w") as f:
            json.dump(_sanitize(out), f, indent=2, default=_json_default)
    except Exception as e:
        print(f"  [manuscript-log] WARN: failed to write run_meta.json: {e}", flush=True)


# --------------------------------------------------------------------------- #
# ManuscriptLogger
# --------------------------------------------------------------------------- #

class ManuscriptLogger:
    """Append-only writer for the manuscript logs in ``log_dir``.

    ``n_full_instances`` is the size of the full evaluation grid for the domain
    (72 for BBOB; the instance-pool size for TSP/OBP/FSSP), used to compute the
    evaluations SAVED vs. a full-grid evaluation of the same candidates.
    """

    def __init__(self, log_dir, n_full_instances: int, enabled: bool = True):
        self.dir = Path(log_dir)
        self.n_full = int(n_full_instances) if n_full_instances else 0
        self.enabled = bool(enabled)

    def _append(self, name: str, rec: dict) -> None:
        if not self.enabled:
            return
        try:
            with open(self.dir / name, "a") as f:
                f.write(json.dumps(_sanitize(rec), default=_json_default) + "\n")
        except Exception as e:                          # never break the run
            print(f"  [manuscript-log] WARN: failed to write {name}: {e}", flush=True)

    # ---- per instance-step within a race (Dims 2,3,8) -------------------- #

    def log_race_steps(self, gen_id: int, steps: list[dict],
                       instance_meta=None, op: str | None = None) -> None:
        """Write the race's per-step trace, tagging each with ``gen_id`` and (if
        ``instance_meta`` is given) resolving ``inst_idx`` -> instance metadata.

        ``instance_meta`` maps a 1-based inst_idx to a dict, e.g.
        ``{"instance_id": "f8i3r0", "fid": 8, "iid": 3, "seed": 91234, "rep": 0}``.

        ``op`` names the sub-race the steps belong to. HiFo runs several sub-races
        (one per operator e1/e2/m1/m2/m3) inside a single generation, so ``gen_id``
        alone does not identify a race there: without ``op`` the traces of
        different sub-races are indistinguishable, and ranks from different races
        would be pooled into one comparison that never happened. Single-race
        frameworks (EoH, LLaMEA) leave it None and the field is omitted.
        """
        for st in steps or []:
            rec = {"gen_id": int(gen_id), **st}
            if op is not None:
                rec["op"] = str(op)
            if instance_meta is not None:
                rec.update(instance_meta(st.get("inst_idx")) or {})
            self._append("race_step_trace.jsonl", rec)

    # ---- per generation/race rollup (Dims 1,2,3) ------------------------ #

    def log_race_summary(self, rec: dict) -> None:
        self._append("race_summary.jsonl", rec)

    def race_summary_record(
        self, *, gen_id, n_candidates_init, n_survivors, instances_evaluated_max,
        total_evaluations_spent, topup_evaluations, stop_reason, race_wall_seconds,
        cpu_seconds, incumbent_cand_id, incumbent_mode, incumbent_partial_score,
        incumbent_coverage, crashed_count, timeout_count, eliminated_count,
        # Optional so that call sites which do not track these fields keep working.
        budget_consumed_cum=None, n_refilled=None, refilled_ids=None, refill_keys=None,
        llm_calls_cum=None, llm_prompt_tokens_cum=None, llm_completion_tokens_cum=None,
        incumbent_rank_id=None, incumbent_cost_id=None,
    ) -> dict:
        """Assemble a race-summary row, computing evaluations_saved_vs_full."""
        saved = None
        if self.n_full:
            saved = int(n_candidates_init) * self.n_full - int(total_evaluations_spent)
        llm_tokens_cum = None
        if llm_prompt_tokens_cum is not None or llm_completion_tokens_cum is not None:
            llm_tokens_cum = int(llm_prompt_tokens_cum or 0) + int(llm_completion_tokens_cum or 0)
        return {
            "gen_id": int(gen_id),
            "n_candidates_init": int(n_candidates_init),
            "n_survivors": int(n_survivors),
            # Refilled candidates (--save-pop).
            "n_refilled": (int(n_refilled) if n_refilled is not None else None),
            "refilled_ids": (list(refilled_ids) if refilled_ids is not None else None),
            "refill_keys": ({k: _num(v) for k, v in refill_keys.items()}
                            if refill_keys is not None else None),
            "instances_evaluated_max": int(instances_evaluated_max),
            "total_evaluations_spent": int(total_evaluations_spent),
            "evaluations_saved_vs_full": (int(saved) if saved is not None else None),
            # Global cumulative evaluation budget.
            "budget_consumed_cum": (int(budget_consumed_cum)
                                    if budget_consumed_cum is not None else None),
            "topup_evaluations": int(topup_evaluations),
            "stop_reason": str(stop_reason),
            "race_wall_seconds": round(float(race_wall_seconds), 3),
            "cpu_seconds": round(float(cpu_seconds), 3),
            # Cumulative LLM cost.
            "llm_calls_cum": (int(llm_calls_cum) if llm_calls_cum is not None else None),
            "llm_prompt_tokens_cum": (int(llm_prompt_tokens_cum)
                                      if llm_prompt_tokens_cum is not None else None),
            "llm_completion_tokens_cum": (int(llm_completion_tokens_cum)
                                          if llm_completion_tokens_cum is not None else None),
            "llm_tokens_cum": (int(llm_tokens_cum) if llm_tokens_cum is not None else None),
            "incumbent_cand_id": incumbent_cand_id,
            "incumbent_mode": incumbent_mode,
            # Both incumbent rules, every generation.
            "incumbent_rank_id": incumbent_rank_id,
            "incumbent_cost_id": incumbent_cost_id,
            "incumbent_partial_score": _num(incumbent_partial_score),
            "incumbent_coverage": int(incumbent_coverage),
            "crashed_count": int(crashed_count),
            "timeout_count": int(timeout_count),
            "eliminated_count": int(eliminated_count),
        }

    # ---- candidate lifecycle / genealogy (Dims 1,2,4) ------------------- #

    def log_candidate(self, rec: dict) -> None:
        self._append("candidate_log.jsonl", rec)

    # ---- per-generation diversity (Dim 4) ------------------------------- #

    def log_diversity(self, rec: dict) -> None:
        self._append("population_diversity_log.jsonl", rec)

    def diversity_record(self, *, gen_id, survivor_sources: list[str],
                         survivor_fitnesses: list[float], crashed_fraction: float,
                         eliminated_carried_count: int) -> dict:
        fits = [f for f in survivor_fitnesses if isinstance(f, (int, float)) and math.isfinite(f)]
        mean = float(sum(fits) / len(fits)) if fits else float("nan")
        std = (float(math.sqrt(sum((f - mean) ** 2 for f in fits) / len(fits)))
               if len(fits) >= 1 else float("nan"))
        hashes = {code_hash(s) for s in survivor_sources if s}
        return {
            "gen_id": int(gen_id),
            "survivor_count": len(survivor_sources),
            "fitness_mean_survivors": _num(mean),
            "fitness_std_survivors": _num(std),
            "unique_code_hashes": len(hashes),
            "mean_pairwise_code_distance": round(mean_pairwise_token_distance(survivor_sources), 4),
            "crashed_fraction": round(float(crashed_fraction), 4),
            "eliminated_carried_count": int(eliminated_carried_count),
        }

    # ---- end-of-run: partial vs. full reliability pairing (Dim 6) ------- #

    def write_reliability(self, rows: list[dict], rule: str = "") -> None:
        if not self.enabled:
            return
        name = f"fitness_reliability_log_{rule}.jsonl" if rule else "fitness_reliability_log.jsonl"
        try:
            with open(self.dir / name, "w") as f:
                for r in rows:
                    f.write(json.dumps(_sanitize(r), default=_json_default) + "\n")
        except Exception as e:
            print(f"  [manuscript-log] WARN: failed to write fitness_reliability_log: {e}", flush=True)

    def finalize_reliability(
        self, incumbents: list[dict], validation: Optional[dict],
        *, higher_is_better: bool = True, rule: str = "",
    ) -> None:
        """Pair each per-generation incumbent's PARTIAL (in-race) score with the
        end-of-run FULL-suite score of the same candidate, reusing
        ``valid_trajectory.json`` only (no extra validation pass).

        ``incumbents`` are this run's per-gen incumbent rows (need ``gen_id``,
        ``cand_id``, ``score`` = partial, ``n_instances`` = partial coverage).
        ``validation`` is the parsed valid_trajectory.json; the full score of a
        candidate is looked up by ``cand_id`` when the validation records carry it,
        else the per-generation validated score is aligned by ``gen_id``.
        """
        # Build cand_id -> full score, and gen_id -> full score, from whatever
        # shape valid_trajectory has (a list of incumbent rows, or a bare score
        # list aligned to the trajectory).
        full_by_cand: dict = {}
        full_by_gen: dict = {}
        full_by_gen_class: dict = {}
        vscore_list = None
        if isinstance(validation, dict):
            vinc = validation.get("incumbents")
            if isinstance(vinc, list):
                for r in vinc:
                    if not isinstance(r, dict):
                        continue
                    fs = r.get("full_score", r.get("score"))
                    if r.get("cand_id") is not None:
                        full_by_cand[r["cand_id"]] = fs
                    if r.get("gen_id") is not None:
                        full_by_gen[r["gen_id"]] = fs
                        if r.get("per_class"):
                            full_by_gen_class[r["gen_id"]] = r["per_class"]
            vscore = validation.get("score")
            if isinstance(vscore, list):                # bare per-incumbent scores
                vscore_list = vscore
            vperclass_list = validation.get("per_class")  # per-incumbent {class -> AOCC}, aligned to score
            if not isinstance(vperclass_list, list):
                vperclass_list = None

        rows = []
        for i, inc in enumerate(incumbents):
            gid = inc.get("gen_id")
            cid = inc.get("cand_id")
            full = full_by_cand.get(cid)
            if full is None:
                full = full_by_gen.get(gid)
            if full is None and vscore_list is not None and i < len(vscore_list):
                full = vscore_list[i]           # positional: valid_trajectory aligned to incumbents
            per_class = full_by_gen_class.get(gid)
            if per_class is None and vperclass_list is not None and i < len(vperclass_list):
                per_class = vperclass_list[i]
            rows.append({
                "gen_id": gid,
                "cand_id": cid,
                "partial_score": _num(inc.get("score")),
                "partial_coverage": inc.get("n_instances"),
                "full_score": _num(full) if full is not None else None,
                "full_coverage": self.n_full or None,
                "per_class": per_class,
                "fair_scored": inc.get("fair_scored", None),
            })
        self.write_reliability(rows, rule=rule)
