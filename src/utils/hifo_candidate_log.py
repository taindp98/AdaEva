"""Full-offspring candidate logging for the HiFo runners.

By default HiFo only dumps its SURVIVING population (pop_size, after
``population_management``) each generation, so a heuristics.json reconstructed from
those dumps lists only distinct survivors — NOT the ``n_operators * pop_size``
offspring HiFo actually generates per generation (unlike EoH's heuristics.json, which
its profiler fills with every sample). That breaks 1:1 EoH-vs-HiFo comparability.

This module captures EVERY evaluated candidate:

  * :func:`wrap_adapter_logging` wraps the problem adapter's ``.evaluate`` so each
    candidate's ``(code, objective)`` is appended to ``all_candidates.jsonl`` in
    EVALUATION ORDER (thread-safe — HiFo evaluates offspring on several sampler
    threads).
  * :func:`build_heuristics_from_candidate_log` reconstructs the full per-generation
    heuristics list from that file.

The flat eval order maps to generations STRUCTURALLY: HiFo runs the whole run loop
generation-by-generation (all of generation N's ``n_operators * pop_size`` offspring —
plus the ``n_init_batches * pop_size`` initial/seed evaluations for generation 0 —
complete before generation N+1 begins, since ``population_management`` sits between
generations and every operator has weight 1 so none are skipped). So candidate index
``k`` belongs to::

    gen 0                          if k < n_init_batches * pop_size          (init/seeds)
    1 + (k - init) // per_gen      otherwise, per_gen = n_operators * pop_size

and its within-generation slot is the position in that generation's eval sequence
(0-based), giving ``gen00_cand00..`` for the init batch and ``gen<g>_cand00..{per_gen-1}``
for each later generation — the full offspring record, NOT deduplicated.
"""

from __future__ import annotations

import json
import threading


def wrap_adapter_logging(adapter, log_dir):
    """Wrap ``adapter.evaluate`` to append every ``(code, objective)`` to
    ``<log_dir>/all_candidates.jsonl`` in evaluation order (thread-safe). Returns the
    same adapter (mutated). ``adapter._cand_log_fh`` holds the file handle so the
    caller can close it on shutdown."""
    path = log_dir / "all_candidates.jsonl"
    fh = open(path, "a", buffering=1)
    lock = threading.Lock()
    _orig_evaluate = adapter.evaluate

    def _logged_evaluate(code_string):
        obj = _orig_evaluate(code_string)
        try:
            _val = float(obj) if isinstance(obj, (int, float)) else None
            with lock:
                fh.write(json.dumps({"objective": _val, "code": code_string}) + "\n")
        except Exception:
            pass  # logging must never break the search
        return obj

    adapter.evaluate = _logged_evaluate
    adapter._cand_log_fh = fh
    return adapter


def build_heuristics_from_candidate_log(log_dir, n_init_batches, n_operators, pop_size):
    """Reconstruct the FULL per-generation heuristics list from all_candidates.jsonl.

    Returns a list of ``{cand_id, gen_id, score, source}`` for EVERY evaluated
    candidate (all offspring, not deduplicated), or ``None`` if the log is absent (the
    caller then falls back to the survivor-dump reconstruction). ``score`` = the
    candidate's objective (mean cost, lower=better); a failed/None objective is stored
    as ``-inf`` for display, matching the dump-based path."""
    path = log_dir / "all_candidates.jsonl"
    if not path.exists():
        return None
    init = int(n_init_batches) * int(pop_size)
    per_gen = max(1, int(n_operators) * int(pop_size))
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    heuristics = []
    for k, r in enumerate(rows):
        if k < init:
            gid, within = 0, k
        else:
            gid = 1 + (k - init) // per_gen
            within = (k - init) % per_gen
        obj = r.get("objective")
        heuristics.append({
            "cand_id": "gen%02d_cand%02d" % (gid, within),
            "gen_id": gid,
            "score": (float(obj) if isinstance(obj, (int, float)) else float("-inf")),
            "source": r.get("code") or "",
        })
    return heuristics
