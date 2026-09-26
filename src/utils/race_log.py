"""Per-race candidate logging for the racing/* runners -> ``race_log.jsonl``.

Each elitist race (one per generation for racing/eoh & racing/llamea; one per
operator sub-race for racing/hifo & sh/hifo) appends ONE JSON object recording:

  * ``race_idx``  -- global monotonic race counter (``self._n_races`` at start)
  * ``gen_id``    -- the generation the race belongs to
  * ``operator``  -- hifo per-operator sub-race tag (``null`` for eoh/llamea)
  * ``seed``      -- the race's RNG seed
  * ``phase_before`` -- the roster ENTERING the race: per-candidate id, whether
    it carried over as an elite, the instances it already holds, and the
    per-instance score, plus ``used_budget`` before the race.
  * ``phase_after``  -- the SETTLED roster: same per-candidate fields after the
    race has evaluated/eliminated, plus ``survived`` / ``eliminated_at`` /
    ``mean_cost`` / ``sum_ranks``, ``used_budget`` after, ``experiments_used``
    and ``break_reason``.

This is deliberately a SEPARATE file from the ManuscriptLogger family so it is
independent of it. Writing goes through ``RacingBase._race``
(covering every eoh/hifo/sh runner, which delegate to it) and inline in
``RacingLLaMEA._race`` (which reimplements the race). Both snapshot the same
``ConfigAS`` fields, so the helpers operate on ``ConfigAS`` objects directly.

Appends are best-effort: a logging failure must never break a run.
"""

from __future__ import annotations

import json
import math
import pathlib
import threading
from typing import Iterable, Optional

_LOCK = threading.Lock()
_FILENAME = "race_log.jsonl"


def _json_num(v) -> Optional[float]:
    """JSON can't hold NaN/inf; map non-finite costs to None so the file stays
    strict-JSON-parseable (the distinction survives as None = no valid score)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def snapshot_candidates(cfgs: Iterable, seen: Optional[set] = None,
                        scores: Optional[dict] = None) -> list:
    """Build the per-candidate roster from ``ConfigAS`` objects.

    ``seen`` (optional, 1-based instance idxs) restricts the reported
    instances/per-instance scores to those actually used THIS race — used for
    the after-phase so an elite's carried-over instances from prior races don't
    leak in. When ``None`` (before-phase) every instance the config carries is
    reported.

    ``scores`` (optional, ``id(cfg) -> sum_ranks``) supplies the race's computed
    sum_ranks per config. Needed for the llamea path, whose ``ConfigAS`` objects
    never populate ``ranks_by_inst`` (so their ``sum_ranks`` property reads 0) —
    the real rank score lives only in the race's local ``scores`` dict there. When
    ``None`` (eoh/hifo path), the config's own ``sum_ranks`` is read (base._race
    writes the correct value onto each CandidateRecord before snapshotting).
    """
    roster = []
    for c in cfgs:
        cbi = getattr(c, "costs_by_inst", {}) or {}
        if seen is not None:
            keys = sorted(k for k in cbi if k in seen)
        else:
            keys = sorted(cbi)
        per_instance = {str(k): _json_num(cbi[k]) for k in keys}
        mc = getattr(c, "mean_cost", float("inf"))
        if scores is not None:
            sr = scores.get(id(c), float("inf"))
        else:
            sr = getattr(c, "sum_ranks", float("inf"))
        roster.append({
            "cand_id": getattr(c, "id", None),
            "is_elite": bool(getattr(c, "is_elite", False)),
            "survived": bool(getattr(c, "alive", True)),
            "n_instances": len(keys),
            "instances": keys,
            "per_instance": per_instance,
            "mean_cost": _json_num(mc),
            "sum_ranks": _json_num(sr),
        })
    return roster


def append_race_record(log_dir, record: dict) -> None:
    """Append one race record to ``<log_dir>/race_log.jsonl`` (thread-safe,
    best-effort — never raises into the caller)."""
    if log_dir is None:
        return
    try:
        path = pathlib.Path(log_dir) / _FILENAME
        line = json.dumps(record)
        with _LOCK:
            with open(path, "a") as f:
                f.write(line + "\n")
    except Exception as e:  # pragma: no cover - logging must not break a run
        print(f"  [race-log] WARN: failed to append race record: {e}", flush=True)
