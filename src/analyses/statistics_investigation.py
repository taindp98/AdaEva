"""Friedman-Conover decision geometry for AdaEva-R racing runs.

Why this module exists
----------------------
The racing loop eliminates candidates with a Friedman omnibus test followed by a
Conover post-hoc test. When elimination under-performs in a scenario it is not
enough to report that "Conover failed": the interesting question is *where in the
decision chain the signal is lost*::

    global concordance -> omnibus detection -> pairwise rank separation
                       -> pairwise uncertainty -> decision margin -> elimination

Design principle: **extract event-level data first, aggregate later.** The
extractor preserves one row per statistical test (plus, optionally, one row per
candidate pair and the raw cost/rank matrices), so new questions can be answered
without re-running experiments or re-reading the logs.

Three levels of output
----------------------
Level 1  :func:`extract_test_events`  -- one row per statistical test.
Level 2  :func:`budget_bin_summary`   -- scenario x budget-bin summary.
Level 3  :func:`scenario_diagnostic`  -- the compact cross-scenario table.

:func:`investigate_conover` runs all three for one scenario;
:func:`compare_scenarios` does it across scenarios and builds the Level-3 table.

Artefacts consumed (per run directory)
--------------------------------------
``instance_seed_perf.jsonl``
    One row per (candidate, instance) evaluation -> the full cost matrix.
``race_step_trace.jsonl``
    One row per race step -> alive set, whether a test ran, eliminations, the
    logged Friedman omnibus p, and the logged ``kendall_w`` / ``q_var``.
``run_meta.json``
    Total budget ``B`` and ``alpha``, used for the normalised budget fraction.

Fidelity notes (verified, and load-bearing)
-------------------------------------------
1. **The instance key is ``task_idx``, not ``base_idx``.** The instance pool wraps
   around -- ``base_idx`` is reused with a fresh seed -- so keying on ``base_idx``
   silently overwrites elite costs and corrupts long races (drops the Friedman p
   match rate from 100% to ~21%).
2. **Eliminations recorded at a step are applied *after* that step's test.** The
   test at step t runs on the set alive going into t. With (1) and (2) the
   recomputed Friedman p matches the logged ``p_value_omnibus`` on **100%** of
   tests for both EoH/TSP (7110 tests) and EoH/OBP (2906 tests).
3. **The logged ``kendall_w`` / ``q_var`` describe a different population than the
   test.** The runner computes them on the *post*-elimination alive set, while the
   Conover test uses the *pre*-elimination set. Both are recorded here:
   ``kendall_W`` / ``Qvar`` are computed on the tested matrix (consistent with the
   test), and ``kendall_W_logged`` / ``Qvar_logged`` carry the runner's values.
   Reconstructing the runner's convention reproduces its ``kendall_w`` exactly.
4. **A non-finite Friedman p is serialised as the string ``"nan"``** (JSON has no
   NaN literal). This happens when every alive candidate is tied on every instance
   (Kendall W = 1, Q_var = 0), so the statistic is 0/0 -- frequent on OBP. Such
   tests are flagged ``all_tied_test`` and carry ``friedman_p = NaN``.
5. **A config that returns a non-finite cost is rejected outright** under
   ``deal_with_crashed: rejection`` (``race.py`` ``_reject_crashed``), so it never
   appears in a tested alive set. The replay drops it too; ranking ``inf`` as an
   ordinary worst value would invent a competitor the race never tested (it did,
   for 2 of 10,016 tests, before this was handled).
6. Recomputed *elimination sets* match the log on ~82% of tests; every difference
   is elite protection (``is_elite_credit``) force-keeping a candidate the test
   would drop, so the actual set is always a subset of the predicted one. The
   Conover geometry itself is exact.

With all of the above, the recomputed Friedman p matches the logged value on
**10,016 / 10,016** tests across the EoH/TSP and EoH/OBP racing runs.

Tie accounting -- exact definitions
-----------------------------------
The previous ``tie_rate`` in this module meant "fraction of *instances* on which at
least two candidates tie", which saturates near 1.0 and is easy to misread. It is
kept as ``frac_instances_with_ties``, and finer-grained counts are added:

``n_tied_observations``   candidate-instance cells whose rank is shared (>=2 tied).
``tie_rate``              ``n_tied_observations / (k*n)`` -- fraction of *observations*
                          that are tied. This is the one to quote.
``n_tied_pairs``          (candidate_i, candidate_j, instance) triples that tie.
``frac_tied_pairs``       ``n_tied_pairs / (C(k,2)*n)``.
``n_tie_groups``          tie groups summed over instances.
``largest_tie_group``     biggest group of mutually-tied candidates on any instance
                          (a max *across* instances -- it says nothing about whether
                          any single instance is fully tied).
``n_fully_tied_instances`` instances on which all k candidates share one rank.
``all_tied_test``         **every** candidate tied on **every** instance, i.e.
                          ``n_fully_tied_instances == n``. This is the degenerate
                          case where the Friedman statistic is 0/0.

                          Note the trap: ``largest_tie_group == k and
                          frac_instances_with_ties == 1`` does **not** imply this --
                          it only says *some* instance was fully tied and every
                          instance had *some* tie, which leaves Friedman perfectly
                          well defined. Testing it that way flagged 566 TSP tests
                          (of which 0 were genuinely all-tied) and 402 OBP tests
                          (of which 247 were).

Figure style
------------
Font sizes live in the :data:`FONTSIZES` dict at the top of this module (title,
axis_label, tick, legend, annotation), with floors in :data:`FONTSIZE_MIN` and a
default figure size in :data:`FIGSIZE`. Sizes are calibrated for a figure
:data:`FONTSIZE_REF_WIDTH` inches wide and scale down with narrower figures, so
bin labels stay legible without overlapping; set ``FONTSIZE_REF_WIDTH = None`` to
disable scaling. Restyle every figure at once by mutating the dict::

    from analyses import statistics_investigation as si
    si.FONTSIZES["axis_label"] = 20
    si.FIGSIZE = (6, 4)

or one figure at a time via ``fontsizes=`` / ``figsize=`` on any plot function or
on :func:`investigate_conover`.

Usage
-----
    from analyses.statistics_investigation import investigate_conover, compare_scenarios

    res = investigate_conover(settings["AdaEva-R"], root=ROOT, scenario="TSP")
    res.events    # Level 1
    res.binned    # Level 2

    table, results = compare_scenarios(
        {"TSP": tsp_runs, "OBP": obp_runs, "FSSP": fssp_runs}, root=ROOT)
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import re
import sys
import time
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import friedmanchisquare, rankdata, t as t_dist

__all__ = [
    "ALPHA",
    "FONTSIZES",
    "FONTSIZE_MIN",
    "FONTSIZE_REF_WIDTH",
    "FIGSIZE",
    "BOX_COLORS",
    "BOX_EDGE_COLOR",
    "ConoverResult",
    "investigate_conover",
    "compare_scenarios",
    "build_events_cache",
    "load_events_cache",
    "visualize_conover",
    "CACHE_DIR",
    "HEARTBEAT_SECONDS",
    "extract_test_events",
    "extract_run_events",
    "reconstruct_race_tests",
    "conover_geometry",
    "tie_profile",
    "budget_bin_summary",
    "scenario_diagnostic",
    "plot_racing_dynamics_boxplot",
    "plot_rank_separation_boxplot",
    "plot_population_dynamics",
    "load_population_dynamics",
    "plot_acceleration_margin",
    "plot_cost_dispersion_pairwise",
    "plot_budget_trajectories",
    "plot_depth_generation_tradeoff",
]

ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Figure style -- edit these to restyle every figure this module produces.
# --------------------------------------------------------------------------- #
#: Font sizes in points, at the reference figure width below. Every plotting
#: function reads these, so changing one entry restyles all figures. Override
#: globally by mutating the dict::
#:
#:     from analyses import statistics_investigation as si
#:     si.FONTSIZES["axis_label"] = 20
#:
#: or per call by passing ``fontsizes={"axis_label": 20}`` to a plot function
#: (merged over these defaults, so partial dicts are fine).
FONTSIZES = {
    "title": 20.0,        # axes title / figure suptitle
    "axis_label": 20.0,   # x and y axis labels
    "tick": 18.0,         # tick labels on both axes
    "legend": 18.0,       # legend entries
    "annotation": 16.0,   # in-axes annotations (the "n=" run counts)
}

#: Lower bounds (points) applied after width scaling, so shrinking a figure
#: never drives text below legibility. Keys match ``FONTSIZES``.
FONTSIZE_MIN = {
    "title": 11.0,
    "axis_label": 11.0,
    "tick": 8.0,
    "legend": 6.5,
    "annotation": 6.0,
}

#: Figure width (inches) the sizes above are calibrated for. Font sizes scale by
#: ``figure_width / FONTSIZE_REF_WIDTH``, clamped to ``[FONTSIZE_MIN, FONTSIZES]``,
#: so the five bin labels stay legible and non-overlapping at any figure size.
#: Set to ``None`` to disable scaling and use ``FONTSIZES`` verbatim.
FONTSIZE_REF_WIDTH = 7.5

#: Default figure size (inches) for the single-axes boxplot.
FIGSIZE = (5, 4)


def _resolve_fontsizes(fig, fontsizes: Optional[dict] = None) -> dict:
    """Resolve effective point sizes for one figure.

    Starts from :data:`FONTSIZES` (overridden by ``fontsizes``), then scales by
    the figure's width relative to :data:`FONTSIZE_REF_WIDTH` and clamps to
    :data:`FONTSIZE_MIN`. Scaling never enlarges past the configured size.
    """
    base = dict(FONTSIZES)
    if fontsizes:
        unknown = set(fontsizes) - set(base)
        if unknown:
            raise KeyError(f"unknown font-size key(s): {sorted(unknown)}; "
                           f"expected any of {sorted(base)}")
        base.update(fontsizes)
    if FONTSIZE_REF_WIDTH:
        scale = fig.get_size_inches()[0] / FONTSIZE_REF_WIDTH
        return {k: float(np.clip(v * scale, FONTSIZE_MIN.get(k, 0.0), v))
                for k, v in base.items()}
    return {k: float(v) for k, v in base.items()}


#: Series colours for the decision-chain boxplot, taken from matplotlib's default
#: ``tab10`` cycle (entry 0 = blue, 1 = orange) rather than hardcoded hex, so the
#: figures follow the active matplotlib style. Override by assigning to
#: :data:`BOX_COLORS`.
BOX_COLORS = list(plt.get_cmap("tab10").colors[:2])

#: Outline colour for boxes, whiskers, caps, medians and outlier markers.
BOX_EDGE_COLOR = "black"

#: Colour of the Conover rejection-threshold reference line.
_C_THRESHOLD = "#d62728"


def _progress(iterable, desc: str, total=None, enabled: bool = True, position: int = 0,
              leave: bool = True):
    """Wrap an iterable in a tqdm bar when tqdm is available and progress is on.

    Falls back to the bare iterable if tqdm is missing, so the module never hard-
    depends on it. Bars are written to stderr, which keeps them out of piped stdout
    and out of the SLURM .out log's data lines.
    """
    if not enabled:
        return iterable
    # Under SLURM stderr is a file, not a terminal. tqdm's carriage-return redraws
    # would pile up thousands of partial lines there, but suppressing the inner
    # bars leaves a long job looking stalled -- the top-level bar only moves when a
    # whole run finishes. So in a non-TTY emit a periodic one-line heartbeat
    # instead: it is append-friendly, greppable, and shows real within-run motion.
    if not sys.stderr.isatty():
        return _heartbeat(iterable, desc=desc, total=total,
                          every=HEARTBEAT_SECONDS)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _heartbeat(iterable, desc=desc, total=total,
                          every=HEARTBEAT_SECONDS)
    return tqdm(iterable, desc=desc, total=total, position=position, leave=leave,
                dynamic_ncols=True, mininterval=1.0, file=sys.stderr)


def _heartbeat(iterable, desc: str, total=None, every: float = 30.0):
    """Yield from ``iterable``, printing one progress line every ``every`` seconds.

    Written for log files: one timestamped line per interval, always terminated by
    a newline, plus a final summary line. Safe from several processes at once --
    each writes whole lines to its own buffer and includes ``desc`` to identify
    itself.
    """
    t0 = last = time.time()
    i = 0
    for i, item in enumerate(iterable, 1):
        yield item
        now = time.time()
        if now - last >= every:
            el = now - t0
            rate = i / el if el > 0 else 0.0
            if total:
                eta = (total - i) / rate if rate > 0 else float("nan")
                print(f"[{_dt.datetime.now():%H:%M:%S}] {desc}: {i}/{total} "
                      f"({100.0 * i / total:.1f}%) {rate:.0f}/s elapsed {el:.0f}s "
                      f"eta {eta:.0f}s", file=sys.stderr, flush=True)
            else:
                print(f"[{_dt.datetime.now():%H:%M:%S}] {desc}: {i} "
                      f"{rate:.0f}/s elapsed {el:.0f}s", file=sys.stderr, flush=True)
            last = now
    el = time.time() - t0
    print(f"[{_dt.datetime.now():%H:%M:%S}] {desc}: done {i}"
          f"{'/' + str(total) if total else ''} in {el:.0f}s", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# artefact loading
# --------------------------------------------------------------------------- #
def _num_or_nan(v) -> float:
    """Coerce a logged scalar to float; the string ``"nan"`` and ``None`` -> NaN."""
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _cost_or_nan(v) -> float:
    """Coerce a logged cost to float.

    JSON has no Infinity literal, so a crashed / rejected evaluation is written as
    the string ``"inf"`` (298 such rows in the EoH/OBP racing runs). Left as a
    string it poisons the cost matrix dtype, so it is parsed to ``+inf`` and kept
    as a marker -- ``reconstruct_race_tests`` then drops the owning candidate from
    the tested set, matching the runner's ``rejection`` policy.
    """
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _resolve(exp_dir, root: pathlib.Path, savpop: bool) -> pathlib.Path:
    """Locate a run directory, tolerating a present-or-absent ``_savpop`` suffix."""
    root = pathlib.Path(root)
    exp_dir = str(exp_dir)
    cands = []
    if savpop and not exp_dir.endswith("_savpop"):
        cands.append(root / f"{exp_dir}_savpop")
    cands.append(root / exp_dir)
    if not exp_dir.endswith("_savpop"):
        cands.append(root / f"{exp_dir}_savpop")
    seen, uniq = set(), []
    for c in cands:
        if str(c) not in seen:
            seen.add(str(c))
            uniq.append(c)
    for c in uniq:
        if (c / "race_step_trace.jsonl").exists():
            return c
    raise FileNotFoundError(
        f"no race_step_trace.jsonl for {exp_dir!r} under {root} (tried: "
        + ", ".join(str(c) for c in uniq) + ")"
    )


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _read_meta(d: pathlib.Path) -> dict:
    p = d / "run_meta.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


# --------------------------------------------------------------------------- #
# concordance / heterogeneity (mirrors utils.race, on the *tested* matrix)
# --------------------------------------------------------------------------- #
def _kendall_w(D: np.ndarray) -> tuple[float, float]:
    """Kendall's W and Spearman's rho. ``D`` is instances (rows) x configs (cols).

    Mirrors ``utils.race._concordance`` (itself irace's ``concordance``), including
    its tie correction and its all-ranks-equal special case.
    """
    n, k = D.shape
    if n <= 1 or k <= 1:
        return float("nan"), float("nan")
    r = np.apply_along_axis(rankdata, 1, D)          # average ties, per instance
    ties_sum = 0.0
    for row in r:
        _, counts = np.unique(row, return_counts=True)
        ties_sum += float(np.sum(counts ** 3 - counts))
    if np.allclose(r, r[:, :1]) and np.all(np.ptp(r, axis=1) == 0):
        w = 1.0
    else:
        col_sums = r.sum(axis=0)
        num = 12 * np.sum((col_sums - n * (k + 1) / 2) ** 2)
        den = (n ** 2 * (k ** 3 - k)) - (n * ties_sum)
        w = num / den if den != 0 else 1.0
    rho = (n * w - 1) / (n - 1)
    return float(w), float(rho)


def _q_var(D: np.ndarray) -> float:
    """Instance-set heterogeneity in [0, 1]; mirrors ``utils.race._data_variance``.

    Returns NaN when any cost is non-finite (a crashed evaluation logged as
    ``"inf"``): the z-scores it is built from are undefined there. Ranks, and so
    every Friedman/Conover quantity, remain well defined.
    """
    n, k = D.shape
    if n <= 1 or k <= 1 or not np.all(np.isfinite(D)):
        return float("nan")
    mean = D.mean(axis=1, keepdims=True)
    std = D.std(axis=1, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    z = (D - mean) / std
    return float(np.mean(z.var(axis=0, ddof=1)))


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
def _legacy_generation_map(d: pathlib.Path, rows: list[dict], steps: list[dict]
                           ) -> tuple[dict, dict]:
    """Recover per-generation instance order and candidate pool without a ``gen`` field.

    LLaMEA-BBOB runs logged before ``src/racing/llamea_bbob.py`` gained the ``gen``
    column omit it from ``instance_seed_perf.jsonl``. It cannot be read off the perf
    rows themselves -- ``task_idx`` is reused across generations, and ``cand_id``
    encodes only a candidate's *birth* generation, not the ones it survives into --
    but it can be rebuilt from three other artefacts:

    instance order
        ``race_step_trace.jsonl`` gives ``(gen_id, task_step) -> seed``, and ``seed``
        maps 1:1 onto ``task_idx`` in the perf rows, so each generation's instance
        sequence is recoverable exactly.

    candidate pool
        A generation's racing pool is its offspring plus the elites carried into it.
        Offspring come from ``candidate_log.jsonl`` (``gen_born``); the carried elite
        pool is printed verbatim by the runner into ``terminal.txt`` as
        ``[save-diversity] elite pool (N candidates) carried to gen NN: [ids]``.
        Generation 1's elites are the initial ``g0_*`` parents.

    Verified against the logged ``p_value_omnibus``: 16185 / 16185 tests match
    exactly across the five 2026-09-01 AdaEva-R BBOB runs.

    Raises:
        ValueError: if the supporting artefacts are missing or inconsistent, rather
            than silently reconstructing a wrong alive set.
    """
    import ast

    seed_to_task: dict = {}
    for r in rows:
        if r.get("seed") is not None:
            seed_to_task.setdefault(r["seed"], r["task_idx"])

    seen: dict[int, dict[int, int]] = collections.defaultdict(dict)
    missing = 0
    for st in steps:
        t = seed_to_task.get(st.get("seed"))
        if t is None:
            missing += 1
            continue
        seen[st["gen_id"]].setdefault(st["task_step"], t)
    if missing:
        raise ValueError(
            f"{d.name}: cannot rebuild the generation map -- {missing} race steps "
            "have a seed absent from instance_seed_perf.jsonl."
        )

    cl_path = d / "candidate_log.jsonl"
    term_path = d / "terminal.txt"
    if not cl_path.exists() or not term_path.exists():
        raise ValueError(
            f"{d.name}: instance_seed_perf.jsonl has no 'gen' field and "
            f"{'candidate_log.jsonl' if not cl_path.exists() else 'terminal.txt'} "
            "is missing, so the racing pool cannot be reconstructed."
        )

    born: dict[int, set] = collections.defaultdict(set)
    for c in _read_jsonl(cl_path):
        born[c["gen_born"]].add(c["cand_id"])

    elite_pool: dict[int, list] = {}
    pat = re.compile(r"elite pool \(\d+ candidates\) carried to gen (\d+): (\[.*\])")
    with open(term_path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                elite_pool[int(m.group(1))] = ast.literal_eval(m.group(2))

    gen0 = sorted({r["cand_id"] for r in rows if str(r["cand_id"]).startswith("g0_")})

    gen_cands: dict[int, set] = collections.defaultdict(set)
    for g in seen:
        carried = elite_pool.get(g, gen0 if g == 1 else None)
        if carried is None:
            raise ValueError(
                f"{d.name}: no '[save-diversity] elite pool ... carried to gen {g:02d}' "
                "line in terminal.txt, so that generation's elite pool is unknown."
            )
        gen_cands[g] = set(born.get(g, set())) | set(carried)

    return seen, gen_cands


def reconstruct_race_tests(exp_dir, root=pathlib.Path("."), savpop: bool = True,
                           progress: bool = False, progress_desc: str = "") -> list[dict]:
    """Replay every elimination test of one run.

    Returns one dict per step where the race actually ran a test, carrying the
    alive candidate ids, the balanced instance set, the (k x n) cost and rank
    matrices, the logged omnibus p / W / Q_var, the cumulative evaluations and the
    eliminations recorded at that step.
    """
    d = _resolve(exp_dir, root, savpop)
    # This phase is the long silent one on big runs (tens of MB of JSONL), so it
    # reports too -- otherwise a worker looks hung until its per-test loop starts.
    _tag = progress_desc or d.name[:28]
    _t0 = time.time()
    _say = (lambda m: print(f"[{_dt.datetime.now():%H:%M:%S}] {_tag}: {m}",
                            file=sys.stderr, flush=True)) if progress else (lambda m: None)
    _say("reading instance_seed_perf.jsonl ...")
    rows = _read_jsonl(d / "instance_seed_perf.jsonl")
    _say(f"read {len(rows)} perf rows in {time.time() - _t0:.0f}s; "
         "reading race_step_trace.jsonl ...")
    steps = _read_jsonl(d / "race_step_trace.jsonl")
    _say(f"read {len(steps)} race steps in {time.time() - _t0:.0f}s; replaying ...")
    meta = _read_meta(d)

    # Run-cumulative budget. NOTE: `cum_evaluations` in the step trace resets at
    # every generation -- it counts evaluations *within the current race*. The
    # run-level total lives in race_summary.jsonl as `budget_consumed_cum` (value
    # at the END of each generation), so the budget at a step is the previous
    # generation's cumulative total plus the within-race counter.
    summ = _read_jsonl(d / "race_summary.jsonl") if (d / "race_summary.jsonl").exists() else []
    cum_before = {}
    prev = 0
    for r in sorted(summ, key=lambda x: x["gen_id"]):
        cum_before[r["gen_id"]] = prev
        prev = r.get("budget_consumed_cum", prev)

    legacy = bool(rows) and "gen" not in rows[0]

    cost: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for r in rows:
        cost[r["cand_id"]][r["task_idx"]] = _cost_or_nan(r["cost"])

    seen: dict[int, dict[int, int]] = collections.defaultdict(dict)
    gen_cands: dict[int, set] = collections.defaultdict(set)
    if legacy:
        seen, gen_cands = _legacy_generation_map(d, rows, steps)
    else:
        for r in rows:
            seen[r["gen"]].setdefault(r["race_step"], r["task_idx"])
            gen_cands[r["gen"]].add(r["cand_id"])

    max_gen = max(seen) if seen else 1
    _gen_iter = _progress(sorted(seen), desc=f"{_tag} replay", total=len(seen),
                          enabled=progress, position=0, leave=False)
    budget_total = meta.get("B") or prev or 1

    out: list[dict] = []
    for g in _gen_iter:
        elim: set[str] = set()
        for s in (x for x in steps if x["gen_id"] == g):
            st = s["task_step"]
            insts = [seen[g][i] for i in sorted(seen[g]) if i <= st]
            alive = [c for c in sorted(gen_cands[g]) if c not in elim]
            # Under `deal_with_crashed: rejection` the runner eliminates a config
            # the moment it returns a non-finite cost (race.py `_reject_crashed`),
            # so it is never part of a tested alive set. Drop those here too --
            # ranking `inf` as an ordinary worst value would invent a competitor
            # the race never tested.
            alive = [c for c in alive
                     if all(np.isfinite(cost[c][k]) for k in insts if k in cost[c])]
            # irace keeps only instances every alive config has been run on
            bal = [k for k in insts if all(k in cost[c] for c in alive)]
            if s["test_ran"] and len(alive) >= 2 and len(bal) >= 2:
                M = np.array([[cost[c][k] for k in bal] for c in alive])
                R = np.apply_along_axis(rankdata, 0, M)  # per-instance ranks
                within = s.get("cum_evaluations") or 0
                cum = cum_before.get(g, 0) + within
                out.append(dict(
                    gen=g, gen_pct=g / max_gen * 100.0, race_step=st,
                    cands=alive, R=R, M=M, insts=bal,
                    logged_p=_num_or_nan(s["p_value_omnibus"]),
                    kendall_W_logged=_num_or_nan(s.get("kendall_w")),
                    Qvar_logged=_num_or_nan(s.get("q_var")),
                    budget_used=cum,
                    budget_within_race=within,
                    budget_total=budget_total,
                    budget_fraction=cum / budget_total,
                    protection_active=bool(s.get("protection_active", False)),
                    eliminated=list(s["eliminated_cand_ids"]),
                    meta=meta,
                ))
            # eliminations recorded at this step take effect for the *next* test
            elim |= set(s["eliminated_cand_ids"])
    return out


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def conover_geometry(R: np.ndarray, alpha: float = ALPHA) -> Optional[dict]:
    """Decompose the irace/Conover best-vs-rest decision (Conover 1999, pp. 369-371).

    Mirrors ``utils.tests.friedman_eliminate(posthoc_test_type="conover")``: each
    config's rank *sum* is compared against the best config's against a single
    critical difference built from a pooled rank variance. There is **no**
    familywise (Tukey/Nemenyi) penalty and no per-pair p-value -- the decision is
    a margin, so that is what gets reported::

        A   = sum of squared ranks,  df = (n-1)(k-1)
        SE  = sqrt( 2 (n*A - sum_i R_i^2) / df )
        T_j = |R_j - R_best| / SE      -> eliminate when T_j > t_{1-alpha/2, df}

    Returns ``None`` when ``df <= 0`` (too few configs or instances to test).
    """
    k, n = R.shape
    Rsum = R.sum(axis=1)
    best = int(np.argmin(Rsum))
    df = (n - 1) * (k - 1)
    if df <= 0:
        return None
    A = float(np.sum(R ** 2))
    R_sq_sum = float(np.sum(Rsum ** 2))
    se = float(np.sqrt(max(0.0, 2 * (n * A - R_sq_sum) / df)))
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=df))
    delta = np.abs(Rsum - Rsum[best])
    T = delta / se if se > 0 else np.where(delta > 0, np.inf, 0.0)
    return dict(best=best, Rsum=Rsum, mean_rank=R.mean(axis=1), delta=delta,
                se=se, t_crit=t_crit, thr=t_crit * se, T=T, margin=T - t_crit,
                k=k, n=n, df=df)


def tie_profile(R: np.ndarray) -> dict:
    """Exact tie accounting for one rank matrix (k configs x n instances).

    See the module docstring for what each field means; ``tie_rate`` is the
    fraction of *observations* (candidate-instance cells) that are tied.
    """
    k, n = R.shape
    n_tied_obs = n_tied_pairs = n_groups = 0
    largest = 1
    inst_with_ties = 0
    n_fully_tied_inst = 0          # instances where ALL k candidates share one rank
    for j in range(n):
        _, counts = np.unique(R[:, j], return_counts=True)
        if counts.max() == k:
            n_fully_tied_inst += 1
        tied = counts[counts > 1]
        if tied.size:
            inst_with_ties += 1
            n_tied_obs += int(tied.sum())
            n_tied_pairs += int(np.sum(tied * (tied - 1) // 2))
            n_groups += int(tied.size)
            largest = max(largest, int(tied.max()))
    total_obs = k * n
    total_pairs = (k * (k - 1) // 2) * n
    return dict(
        n_tied_observations=n_tied_obs,
        tie_rate=n_tied_obs / total_obs if total_obs else np.nan,
        n_tied_pairs=n_tied_pairs,
        frac_tied_pairs=n_tied_pairs / total_pairs if total_pairs else np.nan,
        n_tie_groups=n_groups,
        largest_tie_group=largest,
        frac_instances_with_ties=inst_with_ties / n if n else np.nan,
        n_fully_tied_instances=n_fully_tied_inst,
        frac_fully_tied_instances=n_fully_tied_inst / n if n else np.nan,
        all_tied_test=bool(n and n_fully_tied_inst == n),
    )


# --------------------------------------------------------------------------- #
# Level 1: event-level extraction
# --------------------------------------------------------------------------- #
def extract_run_events(exp_dir, root=pathlib.Path("."), alpha: float = ALPHA,
                       savpop: bool = True, scenario: Optional[str] = None,
                       run_id: Optional[str] = None, run_index: int = 0,
                       keep_pairs: bool = False, keep_matrices: bool = False,
                       progress: bool = False, progress_position: int = 0
                       ) -> tuple[list[dict], list[dict]]:
    """Event-level extraction for one run. Returns ``(test_events, pair_events)``.

    ``progress`` shows a per-test tqdm bar (on stderr); ``progress_position`` sets
    its row, so parallel workers can each keep their own line.
    """
    label0 = pathlib.Path(str(exp_dir)).name
    tests = reconstruct_race_tests(exp_dir, root=root, savpop=savpop,
                                   progress=progress,
                                   progress_desc=f"run {run_index} {label0[:28]}")
    events: list[dict] = []
    pairs: list[dict] = []

    label = pathlib.Path(str(exp_dir)).name
    for t in _progress(tests, desc=f"run {run_index} {label[:28]}", total=len(tests),
                       enabled=progress, position=progress_position, leave=False):
        R, M = t["R"], t["M"]
        gm = conover_geometry(R, alpha=alpha)
        if gm is None:
            continue
        meta = t["meta"]
        rid = run_id or meta.get("run_id") or str(exp_dir)

        # omnibus, recomputed (matches the log exactly; see fidelity note 2)
        try:
            f_stat, f_p = friedmanchisquare(*[R[i] for i in range(R.shape[0])])
            f_stat, f_p = float(f_stat), float(f_p)
        except Exception:
            f_stat = f_p = float("nan")

        ties = tie_profile(R)
        D = M.T                                   # instances x configs
        W, rho = _kendall_w(D)
        qv = _q_var(D)

        others = np.array([i for i in range(gm["k"]) if i != gm["best"]])
        m_other = gm["margin"][others]
        n_sep = int(np.sum(m_other > 0))
        k, n = gm["k"], gm["n"]
        # rank-sum gap normalised by its theoretical maximum, n*(k-1)
        dr_max = float(gm["delta"].max())
        dr_norm = dr_max / (n * (k - 1)) if n * (k - 1) else np.nan

        ev = dict(
            scenario=scenario, run=run_index, run_id=rid,
            domain=meta.get("domain"), framework=meta.get("framework"),
            generation=t["gen"], gen_pct=t["gen_pct"], race_step=t["race_step"],
            budget_used=t["budget_used"], budget_total=t["budget_total"],
            budget_fraction=t["budget_fraction"],
            budget_within_race=t["budget_within_race"],
            k=k, n=n, df=gm["df"],
            friedman_stat=f_stat, friedman_p=f_p,
            friedman_p_logged=t["logged_p"],
            friedman_significant=bool(np.isfinite(f_p) and f_p < alpha),
            kendall_W=W, spearman_rho=rho, Qvar=qv,
            kendall_W_logged=t["kendall_W_logged"], Qvar_logged=t["Qvar_logged"],
            conover_se=gm["se"], conover_t_critical=gm["t_crit"],
            conover_threshold=gm["thr"],
            best_conover_statistic=float(np.max(gm["T"][others])),
            Mmax=float(np.max(m_other)),
            max_rank_sum_gap=dr_max,
            rank_sum_gap_normalised=dr_norm,
            median_rank_sum_gap=float(np.median(gm["delta"][others])),
            delta_over_se=dr_max / gm["se"] if gm["se"] > 0 else np.inf,
            num_separable_candidates=n_sep,
            fraction_separable_candidates=float(n_sep / len(others)) if len(others) else np.nan,
            num_candidates_eliminated=len(t["eliminated"]),
            elimination_occurred=bool(t["eliminated"]),
            protection_active=t["protection_active"],
            best_cand=t["cands"][gm["best"]],
            **ties,
        )
        events.append(ev)

        if keep_matrices:
            ev["rank_matrix"] = R
            ev["cost_matrix"] = M
            ev["candidates"] = list(t["cands"])
            ev["instances"] = list(t["insts"])

        if keep_pairs:
            # All-pairs geometry. NOTE: the *implemented* rule is best-vs-rest with
            # no familywise adjustment, so `significant` here is what the same
            # critical difference would say about an arbitrary pair -- reported for
            # diagnosis, not because the race tests every pair.
            Rsum, mr = gm["Rsum"], gm["mean_rank"]
            for a in range(k):
                for b in range(a + 1, k):
                    d_ab = abs(float(Rsum[a] - Rsum[b]))
                    T_ab = d_ab / gm["se"] if gm["se"] > 0 else (np.inf if d_ab else 0.0)
                    pairs.append(dict(
                        scenario=scenario, run=run_index, run_id=rid,
                        generation=t["gen"], race_step=t["race_step"],
                        budget_fraction=t["budget_fraction"], k=k, n=n,
                        candidate_i=t["cands"][a], candidate_j=t["cands"][b],
                        mean_rank_i=float(mr[a]), mean_rank_j=float(mr[b]),
                        rank_sum_i=float(Rsum[a]), rank_sum_j=float(Rsum[b]),
                        rank_difference=d_ab,
                        conover_statistic=float(T_ab),
                        standard_error=gm["se"],
                        t_critical=gm["t_crit"],
                        margin=float(T_ab - gm["t_crit"]),
                        p_value=float(2 * t_dist.sf(T_ab, df=gm["df"]))
                        if np.isfinite(T_ab) else 0.0,
                        significant=bool(T_ab > gm["t_crit"]),
                        involves_best=bool(a == gm["best"] or b == gm["best"]),
                    ))
    return events, pairs


def extract_test_events(run_dirs: Iterable, root=pathlib.Path("."), alpha: float = ALPHA,
                        savpop: bool = True, scenario: Optional[str] = None,
                        keep_pairs: bool = False, keep_matrices: bool = False,
                        n_jobs: int = 1, progress: bool = False
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**Level 1.** One row per statistical test across all runs of a scenario.

    Returns ``(events, pairs)``; ``pairs`` is empty unless ``keep_pairs=True``.
    Set ``keep_matrices=True`` to also carry the raw cost/rank matrices per event
    (object columns -- heavy, but lets any later metric be recomputed from the
    same evidence).

    ``n_jobs`` > 1 extracts runs in parallel processes (``n_jobs=-1`` uses every
    core). Runs are independent, so this scales close to linearly in the number of
    runs; it is ignored when ``keep_matrices=True``.
    """
    run_dirs = list(run_dirs)
    if n_jobs and n_jobs != 1 and len(run_dirs) > 1 and not keep_matrices:
        return _extract_parallel(run_dirs, root=root, alpha=alpha, savpop=savpop,
                                 scenario=scenario, keep_pairs=keep_pairs,
                                 n_jobs=n_jobs, progress=progress)
    all_ev, all_pr = [], []
    outer = _progress(list(enumerate(run_dirs)), desc=f"{scenario or 'runs'}: runs",
                      total=len(run_dirs), enabled=progress, position=0)
    for i, d in outer:
        ev, pr = extract_run_events(d, root=root, alpha=alpha, savpop=savpop,
                                    scenario=scenario, run_index=i,
                                    keep_pairs=keep_pairs, keep_matrices=keep_matrices,
                                    progress=progress, progress_position=1)
        all_ev.extend(ev)
        all_pr.extend(pr)
    return pd.DataFrame(all_ev), pd.DataFrame(all_pr)


def _extract_one(args: tuple) -> tuple[list, list]:
    """Worker entry point: one run's events. Top-level so it is picklable."""
    d, root, alpha, savpop, scenario, run_index, keep_pairs, progress = args
    return extract_run_events(d, root=root, alpha=alpha, savpop=savpop,
                              scenario=scenario, run_index=run_index,
                              keep_pairs=keep_pairs, progress=progress,
                              progress_position=run_index + 1)


def _extract_parallel(run_dirs: list, root, alpha: float, savpop: bool,
                      scenario: Optional[str], keep_pairs: bool,
                      n_jobs: int, progress: bool = False
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract several runs concurrently.

    Runs are fully independent -- each reads only its own log directory -- so this
    is a plain process fan-out. ``run_index`` is assigned before dispatch, so the
    ``run`` column is deterministic regardless of completion order.

    ``keep_matrices`` is not supported here: the raw matrices would have to be
    pickled back from every worker, which costs more than it saves.
    """
    import concurrent.futures as _cf

    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, len(run_dirs))
    tasks = [(d, root, alpha, savpop, scenario, i, keep_pairs, progress)
             for i, d in enumerate(run_dirs)]
    ev_by_run: dict[int, list] = {}
    pr_by_run: dict[int, list] = {}
    with _cf.ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futs = {pool.submit(_extract_one, t): t[5] for t in tasks}
        done = _progress(_cf.as_completed(futs), desc=f"{scenario or 'runs'}: runs",
                         total=len(futs), enabled=progress, position=0)
        for fut in done:
            i = futs[fut]
            ev_by_run[i], pr_by_run[i] = fut.result()
    all_ev = [e for i in sorted(ev_by_run) for e in ev_by_run[i]]
    all_pr = [p for i in sorted(pr_by_run) for p in pr_by_run[i]]
    return pd.DataFrame(all_ev), pd.DataFrame(all_pr)


# --------------------------------------------------------------------------- #
# Cached extraction: run the heavy Level-1 pass offline, plot from the cache
# --------------------------------------------------------------------------- #
#: Seconds between heartbeat progress lines when stderr is not a terminal
#: (i.e. under SLURM). Lower it for chattier logs.
HEARTBEAT_SECONDS = 30.0

#: Default directory for cached event tables.
CACHE_DIR = pathlib.Path("figures/homo_hetero/conover_cache")


def _cache_path(scenario: str, cache_dir=None, kind: str = "events") -> pathlib.Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario)).strip("_") or "scenario"
    return pathlib.Path(cache_dir or CACHE_DIR) / f"{slug}_{kind}.parquet"


def build_events_cache(scenarios: dict, root=pathlib.Path("."), cache_dir=None,
                       alpha: float = ALPHA, savpop: bool = True,
                       keep_pairs: bool = False, n_jobs: int = -1,
                       overwrite: bool = True, progress: bool = True,
                       verbose: bool = True) -> dict:
    """**Step 1 (offline).** Extract event tables once and cache them to disk.

    This is the expensive half: it replays every elimination test in every run.
    Do it once outside the notebook -- optionally across many cores -- then let the
    notebook plot from the cache with :func:`load_events_cache`.

    Arguments:
        scenarios: ``{scenario_label: run_dirs}``.
        root: Repo root the run paths are relative to.
        cache_dir: Where to write (default :data:`CACHE_DIR`, relative to ``root``).
        n_jobs: Worker processes per scenario (``-1`` = all cores). Runs within a
            scenario are extracted in parallel.
        overwrite: When False, scenarios whose cache already exists are skipped.
        progress: Show tqdm bars (on stderr) -- one for run completion, one per
            run for its elimination tests. Useful for the long OBP-W128 build.

    Returns:
        ``{scenario: path}`` for the event tables written or already present.
    """
    out = {}
    cdir = pathlib.Path(root) / (cache_dir or CACHE_DIR) if not pathlib.Path(
        cache_dir or CACHE_DIR).is_absolute() else pathlib.Path(cache_dir or CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    for label, dirs in scenarios.items():
        ev_path = _cache_path(label, cdir, "events")
        if ev_path.exists() and not overwrite:
            if verbose:
                print(f"[{label}] cache exists, skipping -> {ev_path}")
            out[label] = ev_path
            continue
        t0 = time.time()
        events, pairs = extract_test_events(
            dirs, root=root, alpha=alpha, savpop=savpop, scenario=label,
            keep_pairs=keep_pairs, n_jobs=n_jobs, progress=progress)
        if events.empty:
            raise ValueError(f"no elimination tests reconstructed for {label!r}")
        events.to_parquet(ev_path, index=False)
        out[label] = ev_path
        if keep_pairs and not pairs.empty:
            pairs.to_parquet(_cache_path(label, cdir, "pairs"), index=False)
        if verbose:
            print(f"[{label}] {len(events)} tests from {events.run.nunique()} runs "
                  f"in {time.time() - t0:.1f}s -> {ev_path}")
    return out


def load_events_cache(scenario, root=pathlib.Path("."), cache_dir=None,
                      kind: str = "events") -> pd.DataFrame:
    """**Step 2 (notebook).** Load one cached event table written by
    :func:`build_events_cache`. Accepts a scenario label or a direct path."""
    p = pathlib.Path(scenario)
    if not p.suffix:
        base = pathlib.Path(cache_dir or CACHE_DIR)
        if not base.is_absolute():
            base = pathlib.Path(root) / base
        p = _cache_path(scenario, base, kind)
    if not p.exists():
        raise FileNotFoundError(
            f"no cached events at {p}. Build it first, e.g.:\n"
            f"    python -m analyses.statistics_investigation --scenario ... "
            f"--runs ... --n-jobs 20"
        )
    return pd.read_parquet(p)


def visualize_conover(events: pd.DataFrame, n_bins: int = 5,
                      by: str = "budget_fraction", scenario: Optional[str] = None,
                      title: Optional[str] = None, output_path=None,
                      figsize=None, fontsizes: Optional[dict] = None,
                      legend_loc: str = "best", legend_ncol: int = 1,
                      plot: bool = True, ratio_output_path=None,
                      bootstrap: bool = False,
                      verbose: bool = True) -> ConoverResult:
    """**Step 2 (notebook).** Summaries + figure from an already-extracted table.

    The cheap half of :func:`investigate_conover`: it does no log reading, so it is
    fast enough to re-run while tuning a figure. ``events`` is what
    :func:`load_events_cache` returns.
    """
    if events.empty:
        raise ValueError("empty events table")
    scenario = scenario or (events["scenario"].iloc[0]
                            if "scenario" in events.columns else None)
    binned = budget_bin_summary(events, n_bins=n_bins, by=by, bootstrap=bootstrap)
    diag = scenario_diagnostic(events)

    if verbose:
        head = f"[{scenario}] " if scenario else ""
        print(f"{head}{len(events)} tests from {events.run.nunique()} runs")
        print(f"\n--- Level 2: by {by} ---")
        cols = ["num_tests", "median_k", "median_n", "friedman_sig_rate",
                "all_tied_rate", "median_W", "median_Qvar", "P(Mmax>0)",
                "median_Mmax", "median_deltaR", "median_SE", "median_deltaR/SE",
                "eliminations_per_test"]
        print(binned[cols].to_string(float_format=lambda v: f"{v:.3f}"))
        print("\n--- Level 3: diagnostic ---")
        print(diag.to_string())

    fig = ax = None
    if plot:
        fig, ax = plot_racing_dynamics_boxplot(
            events, n_bins=n_bins, by=by, title=title or scenario,
            output_path=output_path, figsize=figsize, fontsizes=fontsizes,
            legend_loc=legend_loc, legend_ncol=legend_ncol, verbose=False)
    res = ConoverResult(events=events, binned=binned, diagnostic=diag,
                        pairs=pd.DataFrame(), scenario=scenario, fig=fig, axes=ax)
    if plot and ratio_output_path is not None:
        # dR/SE is on a t-like scale, so it gets its own figure rather than a
        # third box on the [0, 1] rate axis.
        res.ratio_fig, res.ratio_axes = plot_rank_separation_boxplot(
            events, n_bins=n_bins, by=by, title=title or scenario,
            output_path=ratio_output_path, figsize=figsize, fontsizes=fontsizes,
            verbose=verbose)
    return res


# --------------------------------------------------------------------------- #
# Level 2: scenario x budget-bin summary
# --------------------------------------------------------------------------- #
def _iqr(s: pd.Series) -> tuple[float, float]:
    s = s.dropna()
    if s.empty:
        return (np.nan, np.nan)
    return (float(s.quantile(0.25)), float(s.quantile(0.75)))


def _boot_ci(s: pd.Series, stat=np.median, n_boot: int = 1000,
             seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of one column."""
    v = s.dropna().values
    if v.size == 0:
        return (np.nan, np.nan)
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    bs = stat(v[idx], axis=1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def budget_bin_summary(events: pd.DataFrame, n_bins: int = 5,
                       by: str = "budget_fraction", bootstrap: bool = False
                       ) -> pd.DataFrame:
    """**Level 2.** Scenario x budget-bin summary of the decision chain.

    ``by`` selects the progress axis: ``"budget_fraction"`` (the recommended common
    x-axis, beta = B_used / B_total) or ``"gen_pct"``.
    """
    if events.empty:
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0 if by == "budget_fraction" else 100.0, n_bins + 1)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" if by == "budget_fraction"
              else f"{int(edges[i])}–{int(edges[i+1])}%" for i in range(n_bins)]
    idx = np.clip(np.searchsorted(edges, events[by].values, side="left") - 1,
                  0, n_bins - 1)
    d = events.assign(_bin=idx)

    group_cols = (["scenario"] if "scenario" in d.columns and d.scenario.notna().any()
                  else [])
    rows = []
    for keys, s in d.groupby(group_cols + ["_bin"], dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        b = keys[-1]
        sig = s[s.friedman_significant]
        row = {}
        if group_cols:
            row["scenario"] = keys[0]
        row.update({
            "bin": labels[int(b)],
            "num_tests": len(s),
            "median_k": s.k.median(),
            "median_n": s.n.median(),
            "friedman_sig_rate": s.friedman_significant.mean(),
            "all_tied_rate": s.all_tied_test.mean(),
            "median_W": s.kendall_W.median(),
            "median_Qvar": s.Qvar.median(),
            "P(Mmax>0)": (s.Mmax > 0).mean(),
            "P(Mmax>0|F)": (sig.Mmax > 0).mean() if len(sig) else np.nan,
            "median_Mmax": s.Mmax.median(),
            "median_Mmax|F": sig.Mmax.median() if len(sig) else np.nan,
            "median_deltaR": s.max_rank_sum_gap.median(),
            "median_deltaR_norm": s.rank_sum_gap_normalised.median(),
            "median_SE": s.conover_se.median(),
            "median_deltaR/SE": s.delta_over_se.median(),
            "median_eliminations": s.num_candidates_eliminated.median(),
            "eliminations_per_test": s.num_candidates_eliminated.mean(),
            "P(elim)": s.elimination_occurred.mean(),
            "P(elim|F)": sig.elimination_occurred.mean() if len(sig) else np.nan,
            "tie_rate": s.tie_rate.mean(),
        })
        lo, hi = _iqr(s.Mmax)
        row["Mmax_IQR"] = f"[{lo:.2f}, {hi:.2f}]"
        if bootstrap:
            lo, hi = _boot_ci(s.Mmax)
            row["Mmax_CI95"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = (["scenario"] if group_cols else []) + ["bin"]
    return out.sort_values(sort_cols).set_index(sort_cols)


# --------------------------------------------------------------------------- #
# Level 3: cross-scenario diagnostic
# --------------------------------------------------------------------------- #
def scenario_diagnostic(events: pd.DataFrame) -> pd.Series:
    """**Level 3.** The compact per-scenario diagnostic."""
    sig = events[events.friedman_significant]
    return pd.Series({
        "tests": len(events),
        "runs": events.run.nunique(),
        "median k": events.k.median(),
        "median n": events.n.median(),
        "P(Friedman sig)": events.friedman_significant.mean(),
        "P(all-tied test)": events.all_tied_test.mean(),
        "median W": events.kendall_W.median(),
        "median Qvar": events.Qvar.median(),
        "P(Mmax>0)": (events.Mmax > 0).mean(),
        "P(Mmax>0 | F sig)": (sig.Mmax > 0).mean() if len(sig) else np.nan,
        "median Mmax | F sig": sig.Mmax.median() if len(sig) else np.nan,
        "median dR_max": events.max_rank_sum_gap.median(),
        "median dR_norm": events.rank_sum_gap_normalised.median(),
        "median SE": events.conover_se.median(),
        "median dR/SE": events.delta_over_se.median(),
        "P(elimination)": events.elimination_occurred.mean(),
        "P(elim | F sig)": sig.elimination_occurred.mean() if len(sig) else np.nan,
        "eliminations / test": events.num_candidates_eliminated.mean(),
        "tie rate (obs)": events.tie_rate.mean(),
        "frac instances w/ ties": events.frac_instances_with_ties.mean(),
    })


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_racing_dynamics_boxplot(events: pd.DataFrame, n_bins: int = 5,
                                 by: str = "budget_fraction", title: Optional[str] = None,
                                 ax=None, output_path=None, ylim=(0.0, 1.0),
                                 figsize=None, fontsizes: Optional[dict] = None,
                                 legend_loc: str = "best", legend_ncol: int = 1,
                                 verbose: bool = False):
    """Boxplot of the two decision-chain rates against normalised budget.

    Series, per progress bin:

    ``P(Friedman sig)``   ``mean(friedman_significant)``
    ``P(elim | F sig)``   ``mean(elimination_occurred)`` over the rows where
                          ``friedman_significant`` -- i.e. how often a detected
                          difference actually converts into an elimination.

    The standardised separation ``dR/SE`` lives on a different scale and has its
    own figure -- see :func:`plot_rank_separation_boxplot`.

    Every rate is computed **per run first**, and the boxes show the distribution
    across runs. Pooling all tests into one rate per bin would discard
    between-run variability and leave nothing for a boxplot to show.

    ``P(elim | F sig)`` is undefined for a run that has no significant test in a
    bin. Such runs are dropped (never imputed or forward-filled) and the number of
    contributing runs is annotated under the box whenever it is below the full
    count.

    ``legend_loc`` / ``legend_ncol`` control the in-axes legend box; the default
    ``"best"`` places it where it overlaps the data least.
    """
    hi = 1.0 if by == "budget_fraction" else 100.0
    edges = np.linspace(0.0, hi, n_bins + 1)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" if by == "budget_fraction"
              else f"{int(edges[i])}–{int(edges[i+1])}%" for i in range(n_bins)]
    idx = np.clip(np.searchsorted(edges, events[by].values, side="left") - 1,
                  0, n_bins - 1)
    d = events.assign(_bin=idx)

    n_runs_total = d["run"].nunique()

    def _per_run(sub: pd.DataFrame) -> tuple[float, float]:
        """(P(F sig), P(elim | F sig)) for one run within one bin."""
        sig = sub[sub.friedman_significant]
        return (
            float(sub.friedman_significant.mean()),
            float(sig.elimination_occurred.mean()) if len(sig) else np.nan,
        )

    f_data, e_data, e_counts = [], [], []
    for b in range(n_bins):
        sl = d[d._bin == b]
        fs, es = [], []
        for _, sub in sl.groupby("run"):
            f, e = _per_run(sub)
            if np.isfinite(f):
                fs.append(f)
            if np.isfinite(e):        # undefined -> dropped, never imputed
                es.append(e)
        f_data.append(fs); e_data.append(es)
        e_counts.append(len(es))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
    else:
        fig = ax.figure

    fs = _resolve_fontsizes(fig, fontsizes)

    x = np.arange(n_bins)
    width = 0.28
    common = dict(patch_artist=True, widths=width * 0.85, manage_ticks=False,
                  medianprops=dict(color=BOX_EDGE_COLOR, linewidth=1.4))
    series = [
        (f_data, -width / 2, BOX_COLORS[0], "P(Friedman sig)"),
        (e_data, width / 2, BOX_COLORS[1], "P(elim | F sig)"),
    ]
    for data, off, color, _ in series:
        ax.boxplot([v or [np.nan] for v in data], positions=x + off,
                   boxprops=dict(facecolor=color, edgecolor=BOX_EDGE_COLOR,
                                 linewidth=1.0),
                   whiskerprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                   capprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                   flierprops=dict(marker="o", markersize=3.5,
                                   markerfacecolor="none",
                                   markeredgecolor=BOX_EDGE_COLOR,
                                   markeredgewidth=0.8),
                   **common)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs["tick"])
    ax.yaxis.set_tick_params(labelsize=fs["tick"])
    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.set_xlabel("Normalized budget", fontsize=fs["axis_label"])
    ax.set_ylabel("Rate", fontsize=fs["axis_label"])
    if ylim:
        ax.set_ylim(*ylim)

    # y-only dotted grid, behind the boxes
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    # Annotate the contributing-run count wherever P(elim | F sig) lost a run.
    for b, cnt in enumerate(e_counts):
        if cnt < n_runs_total:
            ax.annotate(f"n={cnt}", xy=(b, 0.0), xytext=(0, -2),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=fs["annotation"], color="0.35",
                        annotation_clip=False)

    # Legend inside the axes, as a conventional framed box. Patch handles match
    # the filled boxes (a line proxy would misrepresent them). `loc` is passed
    # through so callers can move it off whichever corner the data occupies;
    # "best" lets matplotlib avoid overlapping the boxes.
    handles = [Patch(facecolor=color, edgecolor=BOX_EDGE_COLOR, linewidth=1.0,
                     label=label)
               for _, _, color, label in series]
    ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol,
              fontsize=fs["legend"], frameon=True, framealpha=0.9,
              edgecolor="0.7", fancybox=False, borderaxespad=0.6)
    if title:
        ax.set_title(title, fontsize=fs["title"], pad=8)

    fig.tight_layout()
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax



def plot_rank_separation_boxplot(events, n_bins: int = 5,
                                 by: str = "budget_fraction", title: Optional[str] = None,
                                 ax=None, output_path=None, ylim=None,
                                 figsize=None, fontsizes: Optional[dict] = None,
                                 legend_loc: str = "best", legend_ncol: int = 1,
                                 colors=None, log_y: bool = False,
                                 show_threshold: bool = True,
                                 threshold_label: bool = True,
                                 verbose: bool = False):
    """Boxplot of the standardised rank separation dR/SE against normalised budget.

    ``dR/SE`` = ``max_rank_sum_gap / conover_se`` is the quantity the Conover test
    thresholds: the largest rank-sum gap in units of its own pooled standard error.
    It sits on a t-like scale (per-run medians run ~1.5 on BBOB to ~26 on OBP), so
    it gets its own figure rather than sharing the [0, 1] rate axis of
    :func:`plot_racing_dynamics_boxplot`.

    As there, the median is taken **per run** within each bin and the boxes span
    runs, so they show between-run variability.

    Non-finite ratios (``conover_se == 0``) are dropped, never imputed or clipped.
    They are effectively OBP-only -- 280 of 2906 OBP tests versus 0 for TSP, 0 for
    LLaMEA-BBOB and 1 for FSSP -- and come in two kinds:

    * ``dR == 0`` too (247 of them): every candidate tied on every instance, so the
      ratio is 0/0 and genuinely undefined.
    * ``dR > 0`` (33): every instance produced the *same* split, so the
      between-instance variance is zero while the gap is not. The separation is
      real and maximal (Friedman p ~1e-34..1e-16), just unplottable.

    Dropping the second kind biases the OBP boxes slightly *downward*, since it
    removes the strongest separations; ``verbose`` reports how many were lost.

    ``events`` may be one DataFrame, or a ``{label: DataFrame}`` mapping to draw
    several scenarios side by side within each budget bin (one box per scenario,
    with a legend). Scenarios are drawn in the mapping's order. Set ``log_y=True``
    when the scenarios' scales differ by an order of magnitude or more.
    """
    import matplotlib.pyplot as plt

    # Normalise to {label: frame}; a bare frame is a single unlabelled series.
    if isinstance(events, pd.DataFrame):
        frames = {None: events}
    else:
        frames = dict(events)
    if not frames:
        raise ValueError("no events given")

    hi = 1.0 if by == "budget_fraction" else 100.0
    edges = np.linspace(0.0, hi, n_bins + 1)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" if by == "budget_fraction"
              else f"{int(edges[i])}–{int(edges[i+1])}%" for i in range(n_bins)]
    series = {}          # label -> (per-bin lists, per-bin run counts, per-bin drops)
    for lab, ev in frames.items():
        idx = np.clip(np.searchsorted(edges, ev[by].values, side="left") - 1,
                      0, n_bins - 1)
        d = ev.assign(_bin=idx)
        r_data, r_counts, n_dropped = [], [], []
        for b in range(n_bins):
            sl = d[d._bin == b]
            rs, drop = [], 0
            for _, sub in sl.groupby("run"):
                ratio = sub["delta_over_se"]
                finite = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                drop += len(ratio) - len(finite)
                if len(finite):
                    rs.append(float(finite.median()))
            r_data.append(rs); r_counts.append(len(rs)); n_dropped.append(drop)
        series[lab] = (r_data, r_counts, n_dropped, d["run"].nunique())

    if verbose:
        for lab, (r_data, _c, n_dropped, _n) in series.items():
            if lab is not None:
                print(f"[{lab}]")
            for blab, rs, nd in zip(labels, r_data, n_dropped):
                med = np.median(rs) if rs else np.nan
                print(f"{blab:>10}  n_runs={len(rs):2d}  median dR/SE={med:8.3f}"
                      f"  dropped(SE=0)={nd}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
    else:
        fig = ax.figure
    fs = _resolve_fontsizes(fig, fontsizes)

    x = np.arange(n_bins)
    n_ser = len(series)
    # One box per scenario within each bin, centred on the tick.
    group_w = 0.8
    box_w = group_w / max(n_ser, 1)
    offsets = [(-group_w / 2) + box_w * (i + 0.5) for i in range(n_ser)]
    if colors is None:
        cmap = list(plt.get_cmap("tab10").colors)
        colors = [cmap[i % len(cmap)] for i in range(n_ser)]

    for i, (lab, (r_data, _c, _nd, _n)) in enumerate(series.items()):
        ax.boxplot([v or [np.nan] for v in r_data], positions=x + offsets[i],
                   patch_artist=True, widths=box_w * 0.85, manage_ticks=False,
                   boxprops=dict(facecolor=colors[i], edgecolor=BOX_EDGE_COLOR,
                                 linewidth=1.0),
                   medianprops=dict(color=BOX_EDGE_COLOR, linewidth=1.4),
                   whiskerprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                   capprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                   flierprops=dict(marker="o", markersize=3.5, markerfacecolor="none",
                                   markeredgecolor=BOX_EDGE_COLOR, markeredgewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs["tick"])
    ax.yaxis.set_tick_params(labelsize=fs["tick"])
    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.set_xlabel("Normalized budget", fontsize=fs["axis_label"])
    ax.set_ylabel(r"$\Delta R_{\max}\,/\,SE$", fontsize=fs["axis_label"])
    if log_y:
        # Scenario medians span ~2 (TSP/FSSP) to ~40 (OBP); on a linear axis the
        # smaller two collapse onto the baseline.
        ax.set_yscale("log")

    # Conover's rejection threshold, the reference that makes the axis readable:
    # a box above it is separable, below it is not. Strictly t_{1-alpha/2, df}
    # varies per test with df=(n-1)(k-1), but df is large throughout (44..1520
    # here) so the value is 1.961..2.015 across every scenario -- close enough to
    # a single line. The median over the plotted tests is used, and the actual
    # spread is reported when `verbose`.
    t_all = pd.concat([ev["conover_t_critical"] for ev in frames.values()])
    t_crit = float(t_all.median())
    if verbose:
        print(f"t_crit: median={t_crit:.4f} range=[{t_all.min():.4f}, "
              f"{t_all.max():.4f}] over {len(t_all)} tests")
    if show_threshold:
        ax.axhline(t_crit, color=_C_THRESHOLD, linestyle="--", linewidth=1.4,
                   zorder=1.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    # Annotate only where a run was lost (single-series plots stay uncluttered).
    if n_ser == 1:
        (_lab, (_rd, r_counts, _nd, n_runs_total)), = series.items()
        for b, cnt in enumerate(r_counts):
            if cnt < n_runs_total:
                ax.annotate(f"n={cnt}", xy=(b, ax.get_ylim()[0]), xytext=(0, -2),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=fs["annotation"], color="0.35",
                            annotation_clip=False)

    if n_ser > 1:
        handles = [Patch(facecolor=colors[i], edgecolor=BOX_EDGE_COLOR,
                         linewidth=1.0, label=str(lab))
                   for i, lab in enumerate(series)]
        ax.legend(handles=handles, loc=legend_loc, ncol=legend_ncol,
                  fontsize=fs["legend"], frameon=True, framealpha=0.9,
                  edgecolor="0.7", fancybox=False, borderaxespad=0.6)

    if show_threshold and threshold_label:
        # Sit the label just BELOW the line at the left edge: the boxes crowd the
        # threshold from above (that is the point of the figure), so anything
        # placed above it collides with the lowest scenario's whiskers.
        lo, _hi = ax.get_ylim()
        ax.set_ylim(min(lo, t_crit / 1.6) if log_y else min(lo, t_crit - 0.6), _hi)
        ax.annotate(rf"$t_{{1-\alpha/2,\nu}}={t_crit:.2f}$",
                    xy=(0.0, t_crit), xycoords=("axes fraction", "data"),
                    xytext=(4, -3), textcoords="offset points",
                    ha="left", va="top", color=_C_THRESHOLD,
                    fontsize=fs["annotation"])

    if title:
        ax.set_title(title, fontsize=fs["title"], pad=8)
    fig.tight_layout()
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax



# --------------------------------------------------------------------------- #
# Population dynamics
# --------------------------------------------------------------------------- #
def load_population_dynamics(run_dirs, root=pathlib.Path("."), savpop: bool = True,
                             benchmark: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
    """Per-generation population metrics for one (framework, benchmark), all seeds.

    Reads ``population_diversity_log.jsonl`` (one row per generation, no budget
    field) and joins it to ``race_summary.jsonl`` on ``gen_id`` for
    ``budget_consumed_cum``, then normalises by the run's total budget ``B`` from
    ``run_meta.json``. ``M`` (the configured population size) is read from the same
    file -- never inferred from the observed ``survivor_count``, which ramps up
    over the first generations.

    Generations present in one log but not the other are reported, never dropped
    silently. Returns ``(frame, info)``; ``info`` carries the console-report
    fields (seeds, generation span, M, collapse count, zero-dispersion count).
    """
    frames, info = [], {"benchmark": benchmark, "seeds": [], "M": None,
                        "B": None, "gen_min": None, "gen_max": None,
                        "n_collapse": 0, "n_zero_std_multi": 0,
                        "orphans": [], "n_rows": 0}
    for seed_i, rd in enumerate(run_dirs):
        d = _resolve(rd, root, savpop)
        pdl_p, rs_p = d / "population_diversity_log.jsonl", d / "race_summary.jsonl"
        if not pdl_p.exists() or not rs_p.exists():
            raise FileNotFoundError(f"{d.name}: need both population_diversity_log.jsonl "
                                    f"and race_summary.jsonl")
        pdl = _read_jsonl(pdl_p)
        rs = {r["gen_id"]: r for r in _read_jsonl(rs_p)}
        meta = _read_meta(d)
        M, B = meta.get("M"), meta.get("B")
        if not M or not B:
            raise ValueError(f"{d.name}: run_meta.json lacks M or B (M={M}, B={B}); "
                             "population size must not be inferred from the log")
        info["M"], info["B"] = M, B

        only_pdl = sorted({r["gen_id"] for r in pdl} - set(rs))
        only_rs = sorted(set(rs) - {r["gen_id"] for r in pdl})
        if only_pdl or only_rs:
            info["orphans"].append((d.name, only_pdl, only_rs))

        for r in pdl:
            g = r["gen_id"]
            summ = rs.get(g)
            if summ is None:            # reported above, not silently dropped
                continue
            mean = r.get("fitness_mean_survivors")
            std = r.get("fitness_std_survivors")
            n = r.get("survivor_count")
            cum = summ.get("budget_consumed_cum")
            # CV as a percentage. |.| because fitness_mean_survivors is negative
            # in every row of every benchmark checked (890/890); asserted below.
            cv = (abs(std / mean) * 100.0
                  if (mean not in (None, 0) and std is not None) else np.nan)
            if mean is not None and mean > 0:
                info.setdefault("positive_mean_rows", 0)
                info["positive_mean_rows"] += 1
            if std == 0.0 and n and n > 1:
                info["n_zero_std_multi"] += 1
            if n == 1:
                info["n_collapse"] += 1
            frames.append(dict(
                benchmark=benchmark, seed=seed_i, run_dir=d.name, gen_id=g,
                budget_used=cum, budget_total=B,
                budget_fraction=(cum / B) if cum is not None else np.nan,
                M=M, survivor_count=n, frac_survivors=(n / M) if n is not None else np.nan,
                cv_pct=cv,
                fitness_mean_survivors=mean, fitness_std_survivors=std,
                mean_pairwise_code_distance=r.get("mean_pairwise_code_distance"),
                unique_code_hashes=r.get("unique_code_hashes"),
                crashed_fraction=r.get("crashed_fraction"),
                eliminated_carried_count=r.get("eliminated_carried_count"),
            ))
        info["seeds"].append(seed_i)
    df = pd.DataFrame(frames)
    if not df.empty:
        info["gen_min"], info["gen_max"] = int(df.gen_id.min()), int(df.gen_id.max())
        info["n_rows"] = len(df)
    return df, info


_POP_COLUMNS = [
    ("cv_pct", r"Cost dispersion, CV (\%)", "Cost dispersion"),
    ("crashed_fraction", "Crashed fraction", "Crash rate"),
    ("mean_pairwise_code_distance", "Mean pairwise code distance", "Structural diversity"),
]


def plot_population_dynamics(framework: str, logs_by_benchmark: dict,
                             root=pathlib.Path("."), savpop: bool = True,
                             n_bins: int = 5, output_path=None, figsize=(14, 4.2),
                             fontsizes: Optional[dict] = None, colors=None,
                             ylim_crashed=(0.0, None), log_cv: bool = False,
                             verbose: bool = True):
    """1x3 population-dynamics panel for one framework, three benchmarks per column.

    Columns: cost dispersion (CV %), crashed fraction, structural diversity. Each
    column draws one box per benchmark within each normalised-budget interval.

    Aggregation matches the rest of the paper: the metric is reduced to a **median
    per seed** within each budget interval, and the box shows the distribution of
    those per-seed values across seeds. Generations are not pooled with seeds --
    generations within a run are not independent replicates of seed variation.

    Nothing is imputed, forward-filled or excluded. In particular:

    * ``fitness_std_survivors == 0`` rows are kept and enter the CV as 0. They are
      not a logging gap: they are almost entirely OBP (44 of 45 in the EoH runs,
      1 in TSP, 0 in FSSP) and appear scattered rather than as a leading prefix in
      two of five OBP seeds, which is consistent with genuinely tied survivors in
      a benchmark whose tie rate is ~0.81.
    * ``mean_pairwise_code_distance == 0`` with ``survivor_count == 1`` is
      legitimate (a lone program has no pairwise distances) and is retained; these
      are the collapse events that corroborate column 2.

    ``crashed_fraction`` is the share of a generation's candidates that failed to
    evaluate. It is already a fraction in [0, 1], so unlike ``survivor_count`` it
    needs no normalisation by ``M`` and is comparable across frameworks as logged.

    ``load_population_dynamics`` still returns ``frac_survivors``
    (= ``survivor_count / M``, the **pre-refill** count) should the population-size
    column be wanted again. Note that its per-seed *median* sits at 1.0 almost
    everywhere -- contraction is a tail event, which a median hides.
    """
    import matplotlib.pyplot as plt

    frames, infos = {}, {}
    for bm, dirs in logs_by_benchmark.items():
        frames[bm], infos[bm] = load_population_dynamics(
            dirs, root=root, savpop=savpop, benchmark=bm)

    if verbose:
        for bm, nfo in infos.items():
            print(f"[{framework}/{bm}] seeds={len(nfo['seeds'])} "
                  f"gens={nfo['gen_min']}..{nfo['gen_max']} rows={nfo['n_rows']} "
                  f"M={nfo['M']} B={nfo['B']}")
            print(f"    collapse generations (survivor_count==1): {nfo['n_collapse']}")
            print(f"    zero-dispersion generations (std==0 with >1 survivor): "
                  f"{nfo['n_zero_std_multi']}  [kept, reported as 0]")
            if nfo.get("positive_mean_rows"):
                print(f"    WARNING: {nfo['positive_mean_rows']} rows have a POSITIVE "
                      "fitness mean; the |.| in CV assumes the usual negative convention")
            for name, only_pdl, only_rs in nfo["orphans"]:
                print(f"    WARNING {name}: gen_ids only in diversity log={only_pdl}, "
                      f"only in race_summary={only_rs}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Three panels on one row leave little width per bin: the full "0.0-0.2" form
    # overlaps its neighbour, so label each equal-width bin by its upper edge.
    labels = [f"{edges[i+1]:.1f}" for i in range(n_bins)]
    bms = list(frames)
    if colors is None:
        cmap = list(plt.get_cmap("tab10").colors)
        colors = [cmap[i % len(cmap)] for i in range(len(bms))]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    fs = _resolve_fontsizes(fig, fontsizes)
    x = np.arange(n_bins)
    group_w = 0.8
    box_w = group_w / max(len(bms), 1)
    offsets = [(-group_w / 2) + box_w * (i + 0.5) for i in range(len(bms))]

    for ax, (col, ylab, ptitle) in zip(axes, _POP_COLUMNS):
        for i, bm in enumerate(bms):
            df = frames[bm]
            idx = np.clip(np.searchsorted(edges, df["budget_fraction"].values,
                                          side="left") - 1, 0, n_bins - 1)
            d = df.assign(_bin=idx)
            # per seed -> median within the interval; box spans seeds
            per_seed = (d.groupby(["_bin", "seed"])[col].median()
                        .replace([np.inf, -np.inf], np.nan))
            data = []
            for b in range(n_bins):
                v = (per_seed.loc[b].dropna().tolist()
                     if b in per_seed.index.get_level_values(0) else [])
                data.append(v)
            ax.boxplot([v or [np.nan] for v in data], positions=x + offsets[i],
                       patch_artist=True, widths=box_w * 0.85, manage_ticks=False,
                       boxprops=dict(facecolor=colors[i], edgecolor=BOX_EDGE_COLOR,
                                     linewidth=1.0),
                       medianprops=dict(color=BOX_EDGE_COLOR, linewidth=1.4),
                       whiskerprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                       capprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                       flierprops=dict(marker="o", markersize=3.5,
                                       markerfacecolor="none",
                                       markeredgecolor=BOX_EDGE_COLOR,
                                       markeredgewidth=0.8))
        fs_xtick = min(fs["tick"], 11.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs_xtick)
        ax.tick_params(axis="y", labelsize=fs["tick"])
        ax.set_xlim(-0.6, n_bins - 0.4)
        ax.set_xlabel("Normalized budget", fontsize=fs["axis_label"])
        ax.set_ylabel(ylab, fontsize=fs["axis_label"])
        ax.set_title(ptitle, fontsize=fs["title"])
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        if col == "crashed_fraction" and ylim_crashed:
            ax.set_ylim(*ylim_crashed)
        if col == "cv_pct" and log_cv:
            ax.set_yscale("log")

    handles = [Patch(facecolor=colors[i], edgecolor=BOX_EDGE_COLOR, linewidth=1.0,
                     label=str(bm)) for i, bm in enumerate(bms)]
    # Lay the axes out first, then place the legend above them, so the shared
    # legend never lands on top of the panel titles.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles=handles, loc="upper center", ncol=len(bms),
               fontsize=min(fs["legend"], 13.0), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(framework, fontsize=fs["title"], y=1.06)
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, axes


def _resolve_valid_traj_dir(exp_dir, root: pathlib.Path = pathlib.Path("."),
                            savpop: bool = True) -> pathlib.Path:
    """Locate a run directory containing valid_trajectory_mean_cost.json."""
    root = pathlib.Path(root)
    s = str(exp_dir)
    cands = []
    if savpop and not s.endswith("_savpop"):
        cands.append(root / f"{s}_savpop")
    cands.append(root / s)
    if not s.endswith("_savpop"):
        cands.append(root / f"{s}_savpop")
    for c in cands:
        if c.is_dir() and (c / "valid_trajectory_mean_cost.json").exists():
            return c
    for c in cands:
        if c.is_dir():
            return c
    return root / s


def plot_acceleration_margin(framework: str, curves_by_benchmark: dict,
                             diversity_by_benchmark: Optional[dict] = None,
                             root=pathlib.Path("."), savpop: bool = True,
                             n_bins: int = 5, output_path=None, figsize=(14, 4.2),
                             fontsizes: Optional[dict] = None, verbose: bool = True):
    """1x3 panel showing acceleration margin (AdaEva-R vs E-SH) and generator crash rates.

    One column per combinatorial benchmark (OBP, TSP, FSSP).

    Left y-axis -- acceleration margin:
        At each normalized budget interval, the quality of AdaEva-R's incumbent
        minus that of E-SH's, signed so that positive means AdaEva-R is ahead:
            margin = quality(E-SH) - quality(AdaEva-R) (for minimization metrics)
        Quality is read from valid_trajectory_mean_cost.json (gap_pct for OBP/TSP,
        score for FSSP). For each seed, quality within an interval is reduced to its
        median across evaluations falling in that interval.
        Normalized by dividing by the terminal range (max - min) across all methods
        in that benchmark.
        Plotted as solid median line across paired seeds with an interquartile band (IQR).
        Dashed horizontal line at zero marks parity.

    Right y-axis -- crash rate:
        crashed_fraction for both AdaEva-R and AdaEva-S from population_diversity_log.jsonl,
        joined to race_summary.jsonl on gen_id to obtain budget_consumed_cum / B.
        Aggregated identically: per-seed median in each interval, then median across seeds.
        Plotted as subordinate thin dotted lines without markers in range [0, 0.5].

    Console reports per panel:
        - Margin at each checkpoint interval (median and IQR)
        - Median crash rate per method
        - Budget interval in which margin is greatest
        - Maximum absolute divergence between AdaEva-R and AdaEva-S crash rates
    """
    root = pathlib.Path(root)
    if diversity_by_benchmark is None:
        diversity_by_benchmark = curves_by_benchmark

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    def _format_edge(val: float) -> str:
        return "0" if abs(val) < 1e-6 else f"{val:.1f}"
    labels = [f"{_format_edge(edges[i])}-{_format_edge(edges[i+1])}" for i in range(n_bins)]
    bms = list(curves_by_benchmark.keys())

    fig, axes = plt.subplots(1, len(bms), figsize=figsize, sharex=True)
    if len(bms) == 1:
        axes = [axes]
    fs = _resolve_fontsizes(fig, fontsizes)
    x = np.arange(n_bins)

    panel_stats = {}
    y_mins, y_maxs = [], []

    # First pass: load data, compute normalized margins, calculate shared y limits
    for bm in bms:
        curves_cfg = curves_by_benchmark[bm]
        div_cfg = diversity_by_benchmark.get(bm, curves_cfg)
        metric = "score" if "FSSP" in bm.upper() else "gap_pct"

        # 1. Terminal range across methods for normalization (max - min)
        term_all = []
        for m in ["AdaEva-R", "E-SH"]:
            if m in curves_cfg:
                for d in curves_cfg[m]:
                    rd = _resolve_valid_traj_dir(d, root, savpop)
                    with open(rd / "valid_trajectory_mean_cost.json") as fh:
                        vt = json.load(fh)
                    term_all.append(vt[metric][-1])
        denom = max(term_all) - min(term_all) if term_all else 1.0
        if denom == 0:
            denom = 1.0

        # 2. Quality margin per seed per interval
        n_seeds = len(curves_cfg.get("AdaEva-R", []))
        margins_per_seed = np.zeros((n_seeds, n_bins))

        for seed_i in range(n_seeds):
            d_ada = _resolve_valid_traj_dir(curves_cfg["AdaEva-R"][seed_i], root, savpop)
            d_esh = _resolve_valid_traj_dir(curves_cfg["E-SH"][seed_i], root, savpop)

            with open(d_ada / "valid_trajectory_mean_cost.json") as fh:
                vt_ada = json.load(fh)
            meta_ada = _read_meta(d_ada)
            B_ada = meta_ada.get("B") or 1.0
            ubs_ada = np.array(vt_ada["used_budget"])
            vals_ada = np.array(vt_ada[metric])
            bins_ada = np.clip(np.searchsorted(edges, ubs_ada / B_ada, side="left") - 1, 0, n_bins - 1)

            with open(d_esh / "valid_trajectory_mean_cost.json") as fh:
                vt_esh = json.load(fh)
            meta_esh = _read_meta(d_esh)
            B_esh = meta_esh.get("B") or 1.0
            ubs_esh = np.array(vt_esh["used_budget"])
            vals_esh = np.array(vt_esh[metric])
            bins_esh = np.clip(np.searchsorted(edges, ubs_esh / B_esh, side="left") - 1, 0, n_bins - 1)

            for b in range(n_bins):
                v_ada = vals_ada[bins_ada == b]
                v_esh = vals_esh[bins_esh == b]
                q_ada = np.median(v_ada) if len(v_ada) else (vals_ada[-1] if len(vals_ada) else 0.0)
                q_esh = np.median(v_esh) if len(v_esh) else (vals_esh[-1] if len(vals_esh) else 0.0)
                # Lower is better: margin = AdaEva-S - AdaEva-R (positive means AdaEva-R leads)
                margins_per_seed[seed_i, b] = (q_esh - q_ada) / denom

        med_margin = np.median(margins_per_seed, axis=0)
        q25 = np.percentile(margins_per_seed, 25, axis=0)
        q75 = np.percentile(margins_per_seed, 75, axis=0)
        y_mins.append(float(np.min(q25)))
        y_maxs.append(float(np.max(q75)))

        # 3. Crash rate per method per interval
        crash_rates = {}
        for m in ["AdaEva-R", "E-SH"]:
            if m not in div_cfg:
                continue
            per_seed_crashes = np.full((n_seeds, n_bins), np.nan)
            for seed_i in range(n_seeds):
                rd = _resolve_valid_traj_dir(div_cfg[m][seed_i], root, savpop)
                pdl_p = rd / "population_diversity_log.jsonl"
                rs_p = rd / "race_summary.jsonl"
                meta = _read_meta(rd)
                B = meta.get("B") or 1.0
                if pdl_p.exists() and rs_p.exists():
                    pdl = _read_jsonl(pdl_p)
                    rs_map = {r["gen_id"]: r for r in _read_jsonl(rs_p)}
                    bin_crashes = {b: [] for b in range(n_bins)}
                    for r in pdl:
                        g = r.get("gen_id")
                        if g in rs_map:
                            frac = rs_map[g].get("budget_consumed_cum", 0) / B
                            bin_idx = np.clip(np.searchsorted(edges, frac, side="left") - 1, 0, n_bins - 1)
                            cf = r.get("crashed_fraction")
                            if cf is not None:
                                bin_crashes[bin_idx].append(cf)
                    for b in range(n_bins):
                        if bin_crashes[b]:
                            per_seed_crashes[seed_i, b] = np.median(bin_crashes[b])
            crash_rates[m] = np.nanmedian(per_seed_crashes, axis=0)

        panel_stats[bm] = {
            "denom": denom,
            "med_margin": med_margin,
            "q25": q25,
            "q75": q75,
            "crash_rates": crash_rates,
            "margins_per_seed": margins_per_seed,
        }

    # Determine global y limits for left axis with padding
    global_min = min(y_mins)
    global_max = max(y_maxs)
    pad = 0.1 * (global_max - global_min) if global_max > global_min else 0.5
    ylim_left = (global_min - pad, global_max + pad)

    # Styling
    color_margin = "#1f77b4"
    color_ada_crash = "#d62728"
    color_esh_crash = "#555555"

    twin_axes = []
    # Second pass: render plots
    for i, (ax, bm) in enumerate(zip(axes, bms)):
        st = panel_stats[bm]
        med_margin = st["med_margin"]
        q25, q75 = st["q25"], st["q75"]
        crash_rates = st["crash_rates"]

        # Left axis: Margin
        ax.plot(x, med_margin, color=color_margin, linewidth=2.0, zorder=4)
        ax.fill_between(x, q25, q75, color=color_margin, alpha=0.25, zorder=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
        ax.set_ylim(ylim_left)
        fs_xtick = min(fs["tick"], 11.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs_xtick)
        ax.tick_params(axis="y", labelsize=fs["tick"])
        ax.set_xlabel("Normalized budget", fontsize=fs["axis_label"])
        ax.set_title(bm, fontsize=fs["title"])
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

        if i == 0:
            ax.set_ylabel("Acceleration margin", fontsize=fs["axis_label"])

        # Right axis: Crash rate
        ax_r = ax.twinx()
        twin_axes.append(ax_r)
        ax_r.set_ylim(0.0, 0.5)
        ax_r.tick_params(axis="y", labelsize=fs["tick"])
        ax_r.grid(False)

        ax_r.plot(x, crash_rates.get("AdaEva-R", []), color=color_ada_crash,
                  linestyle=":", linewidth=1.5, alpha=0.85, zorder=5)
        ax_r.plot(x, crash_rates.get("E-SH", []), color=color_esh_crash,
                  linestyle=":", linewidth=1.5, alpha=0.85, zorder=5)

        if i == len(bms) - 1:
            ax_r.set_ylabel("Crash rate", fontsize=fs["axis_label"])
        else:
            ax_r.set_yticklabels([])

    # Construct shared legend
    handles = [
        Line2D([0], [0], color=color_margin, lw=2.0, label="Margin (median, IQR)"),
        Line2D([0], [0], color=color_ada_crash, lw=1.5, ls=":", alpha=0.85, label="AdaEva-R crash rate"),
        Line2D([0], [0], color=color_esh_crash, lw=1.5, ls=":", alpha=0.85, label="AdaEva-S crash rate"),
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=min(fs["legend"], 13.0), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(framework, fontsize=fs["title"], y=1.06)

    # Console reporting
    if verbose:
        print(f"=== [{framework}] Acceleration Margin vs Crash Rate ===")
        for bm in bms:
            st = panel_stats[bm]
            med = st["med_margin"]
            q25, q75 = st["q25"], st["q75"]
            cr = st["crash_rates"]
            best_interval_idx = int(np.argmax(med))
            best_interval = f"{edges[best_interval_idx]:.1f}–{edges[best_interval_idx+1]:.1f}"

            div = np.abs(cr["AdaEva-R"] - cr["E-SH"]) if ("AdaEva-R" in cr and "E-SH" in cr) else [np.nan]
            max_div = float(np.nanmax(div))

            print(f"[{bm}] (terminal range={st['denom']:.4f})")
            for b in range(n_bins):
                b_name = f"{edges[b]:.1f}–{edges[b+1]:.1f}"
                print(f"  bin {b_name} (checkpoint {labels[b]}): "
                      f"median margin = {med[b]:+.4f} (IQR: [{q25[b]:+.4f}, {q75[b]:+.4f}])")
            print(f"  Median crash rate across run: "
                  f"AdaEva-R={np.nanmedian(cr['AdaEva-R']):.4f}, E-SH={np.nanmedian(cr['E-SH']):.4f}")
            print(f"  Greatest margin interval: {best_interval} (margin={med[best_interval_idx]:+.4f})")
            print(f"  Max crash rate divergence between methods: {max_div:.4f}")

    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, axes


def plot_cost_dispersion_pairwise(framework: str, logs_by_benchmark: dict,
                                  root=pathlib.Path("."), savpop: bool = True,
                                  n_bins: int = 5, colors: Optional[dict] = None,
                                  drop_outliers: bool = True, outlier_cutoff: float = 5.0,
                                  share_y: bool = False, output_path=None,
                                  figsize=(14, 4.2), fontsizes: Optional[dict] = None,
                                  verbose: bool = True):
    """1x3 pairwise boxplot of Cost Dispersion CV (%) between AdaEva-R and E-SH.

    One column per combinatorial benchmark (OBP, TSP, FSSP).
    Within each column, at each of the 5 normalized budget intervals,
    draws pairwise side-by-side boxplots (AdaEva-R vs E-SH).

    CV(%) = |fitness_std_survivors / fitness_mean_survivors| * 100.
    Aggregated as median per seed within each budget interval, with boxes showing
    the distribution across seeds. Outliers with CV > outlier_cutoff% can be dropped
    (default: True with cutoff 5.0%).
    """
    root = pathlib.Path(root)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    def _format_edge(val: float) -> str:
        return "0" if abs(val) < 1e-6 else f"{val:.1f}"
    labels = [f"{_format_edge(edges[i])}-{_format_edge(edges[i+1])}" for i in range(n_bins)]
    bms = list(logs_by_benchmark.keys())

    if colors is None:
        colors = {"AdaEva-R": "#d62728", "E-SH": "#1f77b4"}

    methods = ["AdaEva-R", "E-SH"]

    fig, axes = plt.subplots(1, len(bms), figsize=figsize, sharex=True, sharey=share_y)
    if len(bms) == 1:
        axes = [axes]
    fs = _resolve_fontsizes(fig, fontsizes)
    x = np.arange(n_bins)

    group_w = 0.7
    box_w = group_w / len(methods)
    offsets = [(-group_w / 2) + box_w * (m_idx + 0.5) for m_idx in range(len(methods))]

    dropped_outliers_count = 0

    for ax_idx, (ax, bm) in enumerate(zip(axes, bms)):
        cfg = logs_by_benchmark[bm]
        bm_stats = {}

        for m_idx, m in enumerate(methods):
            if m not in cfg:
                continue
            dirs = cfg[m]
            per_seed_bins = {b: [] for b in range(n_bins)}

            for seed_i, d in enumerate(dirs):
                rd = _resolve_valid_traj_dir(d, root, savpop)
                pdl_p = rd / "population_diversity_log.jsonl"
                rs_p = rd / "race_summary.jsonl"
                meta = _read_meta(rd)
                B = meta.get("B") or 1.0

                if not (pdl_p.exists() and rs_p.exists()):
                    continue

                pdl = _read_jsonl(pdl_p)
                rs_map = {r["gen_id"]: r for r in _read_jsonl(rs_p)}

                bin_cvs = {b: [] for b in range(n_bins)}
                for r in pdl:
                    g = r.get("gen_id")
                    if g in rs_map:
                        frac = rs_map[g].get("budget_consumed_cum", 0) / B
                        b = np.clip(np.searchsorted(edges, frac, side="left") - 1, 0, n_bins - 1)
                        mean = r.get("fitness_mean_survivors")
                        std = r.get("fitness_std_survivors")
                        if mean is not None and std is not None:
                            cv = abs(std / mean) * 100.0 if mean != 0 else 0.0
                            if drop_outliers and cv > outlier_cutoff:
                                dropped_outliers_count += 1
                                continue
                            bin_cvs[b].append(cv)

                for b in range(n_bins):
                    med = np.median(bin_cvs[b]) if bin_cvs[b] else np.nan
                    per_seed_bins[b].append(med)

            # Boxplot data: list of arrays per bin
            data = [per_seed_bins[b] for b in range(n_bins)]
            bm_stats[m] = data

            ax.boxplot(
                [v or [np.nan] for v in data],
                positions=x + offsets[m_idx],
                patch_artist=True,
                widths=box_w * 0.85,
                manage_ticks=False,
                boxprops=dict(facecolor=colors[m], edgecolor=BOX_EDGE_COLOR, linewidth=1.0),
                medianprops=dict(color=BOX_EDGE_COLOR, linewidth=1.4),
                whiskerprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                capprops=dict(color=BOX_EDGE_COLOR, linewidth=1.0),
                flierprops=dict(marker="o", markersize=3.5, markerfacecolor="none",
                               markeredgecolor=BOX_EDGE_COLOR, markeredgewidth=0.8),
            )

        ax.set_xticks(x)
        fs_xtick = min(fs["tick"], 11.0)
        ax.set_xticklabels(labels, fontsize=fs_xtick)
        ax.tick_params(axis="y", labelsize=fs["tick"])
        ax.set_xlim(-0.6, n_bins - 0.4)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Normalized budget", fontsize=fs["axis_label"])
        ax.set_title(bm, fontsize=fs["title"])
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

        if ax_idx == 0 or not share_y:
            ax.set_ylabel("CV (%)", fontsize=fs["axis_label"])

        if verbose:
            print(f"[{framework}/{bm}] Cost Dispersion CV (%) per interval:")
            for b in range(n_bins):
                b_name = f"{edges[b]:.1f}–{edges[b+1]:.1f}"
                ada_v = [v for v in bm_stats.get("AdaEva-R", [[]])[b] if not np.isnan(v)]
                esh_v = [v for v in bm_stats.get("E-SH", [[]])[b] if not np.isnan(v)]
                ada_med = np.median(ada_v) if ada_v else np.nan
                esh_med = np.median(esh_v) if esh_v else np.nan
                print(f"  interval {labels[b]}: "
                      f"AdaEva-R median={ada_med:.4f}%, AdaEva-S median={esh_med:.4f}%")

    if verbose and drop_outliers:
        print(f"[{framework}] Total generation rows dropped as outliers (CV > {outlier_cutoff}%): {dropped_outliers_count}")

    handles = [
        Patch(facecolor=colors[m], edgecolor=BOX_EDGE_COLOR, linewidth=1.0, label=m)
        for m in methods
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles=handles, loc="upper center", ncol=len(methods),
               fontsize=min(fs["legend"], 13.0), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(framework, fontsize=fs["title"], y=1.06)

    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, axes



def _load_tradeoff_summary(rd, root=pathlib.Path(".")):
    """Load generation rows for a run directory from race_summary or baseline logs."""
    p = (root / rd) if not pathlib.Path(rd).is_absolute() else pathlib.Path(rd)
    candidates = [
        p / "race_summary.jsonl",
        pathlib.Path(str(p) + "_savpop") / "race_summary.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return [json.loads(l) for l in open(c)]

    eval_candidates = [
        p / "evaluations_per_gen.json",
        pathlib.Path(str(p) + "_savpop") / "evaluations_per_gen.json",
    ]
    for c in eval_candidates:
        if c.exists():
            with open(c) as f:
                ev = json.load(f)
            cum = 0
            rows = []
            for g in ev.get("generations", []):
                evals = g.get("total_instance_evals", 0)
                cum += evals
                rows.append({
                    "gen_id": g.get("gen_id", 0) + 1,
                    "total_evaluations_spent": evals,
                    "n_candidates_init": g.get("num_heuristics", 0),
                    "budget_consumed_cum": cum,
                })
            return rows

    tj_candidates = [
        (p / "trajectory.json", p / "all_candidates.jsonl"),
        (pathlib.Path(str(p) + "_savpop") / "trajectory.json", pathlib.Path(str(p) + "_savpop") / "all_candidates.jsonl"),
    ]
    for p_tj, p_cand in tj_candidates:
        if p_tj.exists() and p_cand.exists():
            with open(p_tj) as f:
                tj = json.load(f).get("trajectory", [])
            n_cands = sum(1 for _ in open(p_cand))
            if tj and n_cands > 0:
                gens = len(tj)
                cands_per_g = n_cands / gens
                last_b = tj[-1].get("used_budget", 0)
                b_per_g = last_b / gens
                cum = 0
                rows = []
                for i in range(gens):
                    cum += b_per_g
                    rows.append({
                        "gen_id": i + 1,
                        "total_evaluations_spent": b_per_g,
                        "n_candidates_init": cands_per_g,
                        "budget_consumed_cum": cum,
                    })
                return rows
    return []


def _load_tradeoff_terminal_quality(rd, root=pathlib.Path(".")):
    """Load terminal quality from valid_trajectory or trajectory."""
    p = (root / rd) if not pathlib.Path(rd).is_absolute() else pathlib.Path(rd)
    for fn in ["valid_trajectory_mean_cost.json", "valid_trajectory.json", "trajectory.json"]:
        for cand in [p / fn, pathlib.Path(str(p) + "_savpop") / fn]:
            if cand.exists():
                with open(cand) as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    if "gap_pct" in data and isinstance(data["gap_pct"], list) and data["gap_pct"]:
                        return float(data["gap_pct"][-1])
                    if "score" in data and isinstance(data["score"], list) and data["score"]:
                        return float(data["score"][-1])
                    if "trajectory" in data and isinstance(data["trajectory"], list) and data["trajectory"]:
                        last = data["trajectory"][-1]
                        if "gap_pct" in last: return float(last["gap_pct"])
                        if "gap" in last:
                            g = float(last["gap"])
                            return g * 100.0 if g < 1.0 else g
                        if "score" in last: return float(last["score"])
                elif isinstance(data, list) and data:
                    last = data[-1]
                    if isinstance(last, dict):
                        if "gap_pct" in last: return float(last["gap_pct"])
                        if "gap" in last:
                            g = float(last["gap"])
                            return g * 100.0 if g < 1.0 else g
                        if "score" in last: return float(last["score"])
    return float("nan")


def plot_depth_generation_tradeoff(
    cells: dict,
    output_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    fontsizes: Optional[dict] = None,
    center: str = "mean",
    whisker: str = "std",
    adaptive_marker: str = "o",
    legend_loc: str = "upper right",
    legend_ncol: Optional[int] = None,
    show_individual_runs: bool = False,
    styles: Optional[dict] = None,
    candidates_per_gen: Optional[dict] = None,
    root: Union[str, pathlib.Path] = pathlib.Path("."),
    verbose: bool = True,
):
    """Plot the depth–generations tradeoff under a fixed evaluation budget.

    Axes:
      x: Generations completed per 1,000 evaluations (natural / linear scale).
      y: Mean evaluations per candidate (natural / linear scale).

    Features:
      - Draws the theoretical hyperbola (x * y = 1000 / candidates_per_gen) as a dashed bound.
      - Unified square marker with color-filled surface matching def _build_styles.
      - Option A: Symmetric errorbars with marker strictly at the center (Mean +/- Std).
      - Accurate candidate evaluation depth (from candidate_log.jsonl or new offspring count).
      - Clean presentation without overlapping individual run shadows.
    """
    import matplotlib.cm as cm
    root = pathlib.Path(root)

    # Normalize cells dictionary
    sample_key = next(iter(cells.keys()))
    if isinstance(cells[sample_key], list):
        cells_dict = {"EoH / OBP": cells}
    else:
        cells_dict = cells

    cell_keys = list(cells_dict.keys())
    is_grid = all(isinstance(k, (tuple, list)) and len(k) == 2 for k in cell_keys)

    if is_grid:
        frameworks = list(dict.fromkeys(k[0] for k in cell_keys))
        benchmarks = list(dict.fromkeys(k[1] for k in cell_keys))
        nrows, ncols = len(frameworks), len(benchmarks)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5.2 * ncols, 4.4 * nrows + 0.6), squeeze=False)
        cell_map = {(fw, bm): axes[f_idx, b_idx]
                    for f_idx, fw in enumerate(frameworks)
                    for b_idx, bm in enumerate(benchmarks)}
    else:
        n_panels = len(cell_keys)
        if n_panels == 1:
            nrows, ncols = 1, 1
            fig, axes = plt.subplots(1, 1, figsize=figsize or (7.5, 5.0), squeeze=False)
        elif n_panels == 6:
            nrows, ncols = 2, 3
            fig, axes = plt.subplots(2, 3, figsize=figsize or (15.0, 8.6), squeeze=False)
        else:
            nrows, ncols = 1, n_panels
            fig, axes = plt.subplots(1, n_panels, figsize=figsize or (5.0 * n_panels, 4.8), squeeze=False)
        cell_map = {k: axes.flat[idx] for idx, k in enumerate(cell_keys)}

    fs = _resolve_fontsizes(fig, fontsizes)
    tab10 = cm.get_cmap("tab10")

    def _get_color(m: str, idx: int = 0):
        if styles and m in styles and "color" in styles[m]:
            return styles[m]["color"]
        if m == "AdaEva-R":
            return tab10(1)   # orange
        elif m == "E-SH":
            return tab10(2)   # green
        elif m == "Uniform-1":
            return "gray"
        elif m in ("Uniform-10", "Fixed-50%"):
            return tab10(3)   # red
        elif m in ("Uniform-25", "Uniform-64"):
            return tab10(0)   # blue
        elif m == "Uniform-16":
            return tab10(4)   # purple
        else:
            return tab10(idx % 10)

    all_methods_seen = []

    for cell_key, ax in cell_map.items():
        if cell_key not in cells_dict:
            ax.set_visible(False)
            continue

        method_dict = cells_dict[cell_key]
        cell_title = f"{cell_key[0]} / {cell_key[1]}" if isinstance(cell_key, (tuple, list)) else str(cell_key)

        cell_stats = {}
        for m, dirs in method_dict.items():
            if m not in all_methods_seen:
                all_methods_seen.append(m)
            xs, ys, gens, qs = [], [], [], []
            for d in dirs:
                p = (root / d) if not pathlib.Path(d).is_absolute() else pathlib.Path(d)
                cl_path = p / "candidate_log.jsonl"
                if not cl_path.exists() and pathlib.Path(str(p) + "_savpop").exists():
                    cl_path = pathlib.Path(str(p) + "_savpop") / "candidate_log.jsonl"

                md = np.nan
                if cl_path.exists():
                    try:
                        cl = [json.loads(l) for l in open(cl_path)]
                        evals = [c["n_instances_evaluated"] for c in cl if "n_instances_evaluated" in c]
                        if evals:
                            md = float(np.mean(evals))
                    except Exception:
                        pass

                rows = _load_tradeoff_summary(d, root=root)
                if not rows:
                    continue

                if np.isnan(md):
                    tot_eval = sum(r.get("total_evaluations_spent", 0) for r in rows)
                    tot_cand = sum(r.get("n_candidates_init", 0) for r in rows)
                    if "racing" in str(d) or "sh" in str(d):
                        tot_cand = tot_cand / 2.0
                    md = tot_eval / tot_cand if tot_cand else np.nan

                g = max(r.get("gen_id", 0) for r in rows)
                fb = rows[-1].get("budget_consumed_cum", 0)
                x = g / (fb / 1000.0) if fb else np.nan
                q = _load_tradeoff_terminal_quality(d, root=root)
                xs.append(x)
                ys.append(md)
                gens.append(g)
                qs.append(q)

            if xs and ys:
                x_mean, x_std = np.mean(xs), np.std(xs)
                y_mean, y_std = np.mean(ys), np.std(ys)
                x_med, y_med = np.median(xs), np.median(ys)

                if center == "mean":
                    cx, cy = x_mean, y_mean
                else:
                    cx, cy = x_med, y_med

                cell_stats[m] = {
                    "xs": np.array(xs), "ys": np.array(ys), "gens": np.array(gens), "qs": np.array(qs),
                    "cx": cx, "cy": cy,
                    "x_mean": x_mean, "x_std": x_std,
                    "y_mean": y_mean, "y_std": y_std,
                    "x_med": x_med, "x_min": np.min(xs), "x_max": np.max(xs),
                    "y_med": y_med, "y_min": np.min(ys), "y_max": np.max(ys),
                    "g_mean": np.mean(gens),
                    "q_mean": np.nanmean(qs) if qs else np.nan,
                }

        if not cell_stats:
            continue

        # Candidates per generation for hyperbola
        if candidates_per_gen and cell_key in candidates_per_gen:
            c_per_g = candidates_per_gen[cell_key]
        else:
            uniform_prods = []
            for m, s in cell_stats.items():
                if "Uniform" in m:
                    for x, y in zip(s["xs"], s["ys"]):
                        if x > 0 and y > 0:
                            uniform_prods.append(1000.0 / (x * y))
            c_per_g = float(np.median(uniform_prods)) if uniform_prods else 20.0

        all_xs = np.concatenate([s["xs"] for s in cell_stats.values()])
        all_ys = np.concatenate([s["ys"] for s in cell_stats.values()])
        x_max = max(all_xs) * 1.08
        y_max = max(all_ys) * 1.08

        # Draw Hyperbola in natural (linear) scale
        x_hyp_start = max(0.5, 1000.0 / (c_per_g * (y_max * 1.05)))
        x_hyp = np.linspace(x_hyp_start, x_max * 1.02, 300)
        y_hyp = 1000.0 / (c_per_g * x_hyp)

        ax.plot(x_hyp, y_hyp, linestyle="--", color="#555555", linewidth=1.5, alpha=0.85,
                label=f"Fixed bound ({c_per_g:.0f} cands/gen)", zorder=1)

        # Draw Uniform baselines first, then adaptive methods (E-SH, AdaEva-R) on top with higher zorder
        sorted_methods = sorted(cell_stats.items(), key=lambda kv: 0 if "Uniform" in kv[0] else 1)
        for m_idx, (m, s) in enumerate(sorted_methods):
            col = _get_color(m, m_idx)
            is_uniform = "Uniform" in m

            # Optional individual runs
            if show_individual_runs:
                ax.scatter(s["xs"], s["ys"], facecolors=[col], edgecolors="black",
                           alpha=0.35, s=36, marker="s", linewidths=0.8, zorder=2)

            # Whiskers (Symmetric Mean +/- Std, or min-max / IQR)
            if whisker == "std":
                xerr = np.array([[s["x_std"]], [s["x_std"]]]) if s["x_std"] > 1e-4 else None
                yerr = np.array([[s["y_std"]], [s["y_std"]]]) if s["y_std"] > 1e-4 else None
            elif whisker == "iqr":
                x_25, x_75 = np.percentile(s["xs"], 25), np.percentile(s["xs"], 75)
                y_25, y_75 = np.percentile(s["ys"], 25), np.percentile(s["ys"], 75)
                xerr = np.array([[s["cx"] - x_25], [x_75 - s["cx"]]]) if (x_75 - x_25) > 1e-4 else None
                yerr = np.array([[s["cy"] - y_25], [y_75 - s["cy"]]]) if (y_75 - y_25) > 1e-4 else None
            else:  # minmax
                xerr = np.array([[s["cx"] - s["x_min"]], [s["x_max"] - s["cx"]]]) if (s["x_max"] - s["x_min"]) > 1e-4 else None
                yerr = np.array([[s["cy"] - s["y_min"]], [s["y_max"] - s["cy"]]]) if (s["y_max"] - s["y_min"]) > 1e-4 else None

            if xerr is not None or yerr is not None:
                eb_zorder = 3 if is_uniform else 6
                ax.errorbar([s["cx"]], [s["cy"]],
                            xerr=xerr, yerr=yerr,
                            fmt="none", ecolor=col, elinewidth=1.8,
                            capsize=5.0, capthick=1.8, zorder=eb_zorder)

            if is_uniform:
                # Uniform-K: square marker with color-filled surface and black edge
                ax.scatter([s["cx"]], [s["cy"]],
                           s=100, marker="s",
                           facecolors=[col], edgecolors="black", linewidths=1.2, zorder=4)
            else:
                # Adaptive methods (E-SH, AdaEva-R): unfilled marker with matching whisker color,
                # rendered with higher zorder than all Uniform-K markers and whiskers
                m_shape = adaptive_marker
                ax.scatter([s["cx"]], [s["cy"]],
                           s=60, marker=m_shape,
                           facecolors="none", edgecolors=[col], linewidths=1.8, zorder=7)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle=":", alpha=0.5)
        title_fs = min(fs["title"], 14.0) if nrows == 1 else fs["title"]
        ax.set_title(cell_title, fontsize=title_fs, fontweight="medium", pad=10)

        # Console reporting per cell
        if verbose:
            print(f"=== [{cell_title}] Depth–Generations Tradeoff Diagnostics ===")
            print(f"  Reference hyperbola: {c_per_g:.1f} candidates/gen  (x * y = {1000.0/c_per_g:.2f})")
            uniform_arms = []
            for m, s in cell_stats.items():
                conf_k = None
                if "Uniform-" in m:
                    try:
                        conf_k = float(m.split("-")[1])
                        uniform_arms.append((m, conf_k, s["g_mean"], s["cy"]))
                    except Exception:
                        pass
                depth_ratio_str = f"{s['cy']/conf_k:.4f}" if conf_k else "N/A (adaptive outcome)"
                k_str = f"{conf_k:g}" if conf_k else "None"
                print(f"  {m:12s}: mean_gens={s['g_mean']:5.1f} | gens/1k={s['cx']:6.3f} +/- {s['x_std']:.3f} | "
                      f"mean_depth={s['cy']:6.3f} +/- {s['y_std']:.3f} | K={k_str:4s} | depth/K={depth_ratio_str} | "
                      f"quality={s['q_mean']:.4f}")

            if len(uniform_arms) >= 2:
                uniform_arms.sort(key=lambda item: item[1])
                shallow = uniform_arms[0]
                deep = uniform_arms[-1]
                ratio = shallow[2] / deep[2] if deep[2] > 0 else np.nan
                print(f"  Generation ratio (deepest {deep[0]} vs shallowest {shallow[0]}): "
                      f"{deep[2]:.1f} vs {shallow[2]:.1f} gens (shallow/deep factor = {ratio:.2f}x)")
            print()

    # Outer axis labels and tick params
    for r_idx in range(nrows):
        for c_idx in range(ncols):
            ax = axes[r_idx, c_idx]
            if r_idx == nrows - 1:
                ax.set_xlabel("Generations completed per 1,000 evaluations",
                              fontsize=min(fs["axis_label"], 12.5) if nrows == 1 else fs["axis_label"])
            if c_idx == 0:
                ax.set_ylabel("Mean evaluations per candidate",
                              fontsize=min(fs["axis_label"], 12.5) if nrows == 1 else fs["axis_label"])
            ax.tick_params(labelsize=min(fs["tick"], 11.0) if nrows == 1 else fs["tick"])

    # Shared Legend: unified square markers filled with _build_styles colors
    uniform_keys = [m for m in all_methods_seen if "Uniform-" in m]
    try:
        uniform_keys.sort(key=lambda x: float(x.split("-")[1]))
    except Exception:
        pass
    adaptive_keys = [m for m in ["E-SH", "AdaEva-R"] if m in all_methods_seen]
    other_keys = [m for m in all_methods_seen if m not in uniform_keys and m not in adaptive_keys]

    ordered_methods = uniform_keys + adaptive_keys + other_keys
    legend_handles = []
    for idx, m in enumerate(ordered_methods):
        col = _get_color(m, idx)
        if "Uniform" in m:
            legend_handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=col,
                                         markeredgecolor="black", markeredgewidth=1.0, markersize=8.5 if nrows==1 and ncols==1 else 9, label=m))
        else:
            m_shape = adaptive_marker
            legend_handles.append(Line2D([0], [0], marker=m_shape, color=col, markerfacecolor="none",
                                         markeredgecolor=col, markeredgewidth=1.8, markersize=7.0 if nrows==1 and ncols==1 else 7.5, linewidth=1.8, label=m))
    legend_handles.append(Line2D([0], [0], linestyle="--", color="#555555", linewidth=1.5, label="Fixed bound"))

    if nrows == 1 and ncols == 1:
        if legend_loc in ("upper right", "ur", "inside"):
            ax = axes[0, 0]
            ncol = legend_ncol if legend_ncol is not None else 1
            legend_fs = min(fs["legend"], 10.0)
            ax.legend(handles=legend_handles, loc="upper right", ncol=ncol,
                      fontsize=legend_fs, frameon=True, framealpha=0.92,
                      edgecolor="#cccccc", borderpad=0.5, handletextpad=0.5)
            fig.tight_layout()
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            legend_fs = min(fs["legend"], 11.0)
            fig.legend(handles=legend_handles, loc="upper center", ncol=3,
                       fontsize=legend_fs, frameon=False, bbox_to_anchor=(0.5, 1.01))
    elif nrows == 1:
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(legend_handles), 6),
                   fontsize=min(fs["legend"], 12.0), frameon=False, bbox_to_anchor=(0.5, 0.99))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(legend_handles), 6),
                   fontsize=min(fs["legend"], 12.0), frameon=False, bbox_to_anchor=(0.5, 0.99))

    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, axes



def plot_budget_trajectories(events: pd.DataFrame, n_bins: int = 8,
                             title: Optional[str] = None, output_path=None,
                             figsize=(10, 9), fontsizes: Optional[dict] = None):
    """The four trajectories A-D against normalised budget, one line per scenario.

    (A) P(Friedman significant)   (B) P(Mmax > 0) and median Mmax | F
    (C) dR_max / SE               (D) eliminations per test
    """
    import matplotlib.cm as cm
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.searchsorted(edges, events["budget_fraction"].values,
                                  side="left") - 1, 0, n_bins - 1)
    d = events.assign(_bin=idx)
    scen = (list(pd.unique(d.scenario.dropna()))
            if "scenario" in d.columns and d.scenario.notna().any() else [None])
    colors = cm.tab10(np.linspace(0, 1, max(len(scen), 3)))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fs = _resolve_fontsizes(fig, fontsizes)
    panels = [
        ("friedman_significant", "mean", r"(A) $P(p_F<\alpha)$", "P(Friedman significant)"),
        ("Mmax", "pgt0", r"(B) $P(M_{\max}>0)$", "P(separable)"),
        ("delta_over_se", "median", r"(C) $\Delta R_{\max}/SE$", r"median $\Delta R_{\max}/SE$"),
        ("num_candidates_eliminated", "mean", "(D) eliminations / test", "eliminations / test"),
    ]
    for ax, (col, how, ttl, ylab) in zip(axes.ravel(), panels):
        for si, sc in enumerate(scen):
            s = d if sc is None else d[d.scenario == sc]
            ys = []
            for b in range(n_bins):
                v = s.loc[s._bin == b, col].dropna()
                if v.empty:
                    ys.append(np.nan)
                elif how == "mean":
                    ys.append(v.mean())
                elif how == "median":
                    ys.append(v.median())
                else:
                    ys.append((v > 0).mean())
            ax.plot(centers, ys, marker="o", markersize=4, linewidth=1.8,
                    color=colors[si], label=sc or "all")
        ax.set_title(ttl, fontsize=fs["title"])
        ax.set_ylabel(ylab, fontsize=fs["axis_label"])
        ax.set_xlabel(r"$\beta=B_{used}/B_{total}$", fontsize=fs["axis_label"])
        ax.tick_params(labelsize=fs["tick"])
        ax.grid(True, linestyle=":", alpha=0.5)
        if scen != [None]:
            ax.legend(fontsize=fs["legend"], frameon=False)
    if title:
        fig.suptitle(title, fontsize=fs["title"])
    fig.tight_layout()
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ConoverResult:
    """Everything one scenario's investigation produced."""
    events: pd.DataFrame          # Level 1
    binned: pd.DataFrame          # Level 2
    diagnostic: pd.Series         # Level 3
    pairs: pd.DataFrame           # per-pair records (empty unless keep_pairs)
    scenario: Optional[str] = None
    fig: object = None
    axes: object = None
    ratio_fig: object = None      # dR/SE figure (plot_rank_separation_boxplot)
    ratio_axes: object = None

    def __repr__(self) -> str:
        return (f"ConoverResult(scenario={self.scenario!r}, tests={len(self.events)}, "
                f"runs={self.events.run.nunique() if len(self.events) else 0}, "
                f"pairs={len(self.pairs)})")

    def save(self, out_dir, prefix: Optional[str] = None) -> dict:
        """Write the event/pair/summary tables to CSV for later reuse."""
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pre = prefix or (self.scenario or "conover").replace("/", "_")
        written = {}
        drop = [c for c in ("rank_matrix", "cost_matrix", "candidates", "instances")
                if c in self.events.columns]
        p = out_dir / f"{pre}_test_events.csv"
        self.events.drop(columns=drop).to_csv(p, index=False); written["events"] = p
        p = out_dir / f"{pre}_budget_bins.csv"
        self.binned.to_csv(p); written["binned"] = p
        p = out_dir / f"{pre}_diagnostic.csv"
        self.diagnostic.to_csv(p, header=["value"]); written["diagnostic"] = p
        if not self.pairs.empty:
            p = out_dir / f"{pre}_pairwise.csv"
            self.pairs.to_csv(p, index=False); written["pairs"] = p
        return written


def investigate_conover(run_dirs: Sequence, root=pathlib.Path("."),
                        scenario: Optional[str] = None, title: Optional[str] = None,
                        alpha: float = ALPHA, n_bins: int = 5, by: str = "budget_fraction",
                        savpop: bool = True, keep_pairs: bool = False,
                        keep_matrices: bool = False, plot: bool = True,
                        output_path=None, figsize=None,
                        fontsizes: Optional[dict] = None,
                        legend_loc: str = "best", legend_ncol: int = 1,
                        n_jobs: int = 1, progress: bool = False,
                        bootstrap: bool = False,
                        verbose: bool = True) -> ConoverResult:
    """Run the full Friedman-Conover investigation for one scenario.

    Arguments:
        run_dirs: Run directories for the scenario, e.g. ``settings["AdaEva-R"]``.
        root: Repo root the run paths are relative to (``ROOT`` in the notebooks).
        scenario: Short label recorded in every row, e.g. ``"TSP"``.
        title: Figure suptitle; defaults to ``scenario``.
        alpha: Significance level; must match the racing run's.
        n_bins: Number of progress bins for the Level-2 summary.
        by: Progress axis -- ``"budget_fraction"`` (default) or ``"gen_pct"``.
        savpop: Resolve the ``_savpop`` variant of each run directory.
        keep_pairs: Also extract per-candidate-pair records (large).
        keep_matrices: Carry the raw cost/rank matrices on each event.
        plot: Draw the Kendall's W / Q_var / Friedman-rate boxplot.
        output_path: Save the figure here (parent dirs are created).
        figsize: Figure size for the boxplot (default :data:`FIGSIZE`).
        fontsizes: Per-figure font-size overrides merged over :data:`FONTSIZES`,
            e.g. ``{"axis_label": 20, "tick": 12}``.
        legend_loc: Matplotlib ``loc`` for the in-axes legend box (default
            ``"best"``); e.g. ``"upper right"`` to pin it.
        legend_ncol: Columns in the legend box.
        n_jobs: Worker processes for the extraction (one run each; ``-1`` = all
            cores). Fine to raise for light scenarios extracted inline; heavy ones
            are better cached offline with :func:`build_events_cache`.
        progress: Show extraction progress (tqdm in a terminal, periodic heartbeat
            lines when stderr is a file).
        bootstrap: Add bootstrap CIs to the Level-2 table.
        verbose: Print the Level-2 and Level-3 tables.

    Returns:
        A :class:`ConoverResult` with ``.events``, ``.binned``, ``.diagnostic``,
        ``.pairs`` and ``.fig``.
    """
    events, pairs = extract_test_events(
        run_dirs, root=root, alpha=alpha, savpop=savpop, scenario=scenario,
        keep_pairs=keep_pairs, keep_matrices=keep_matrices,
        n_jobs=n_jobs, progress=progress)
    if events.empty:
        raise ValueError(f"no elimination tests reconstructed for {scenario!r}; "
                         "check the run paths and the savpop flag")

    binned = budget_bin_summary(events, n_bins=n_bins, by=by, bootstrap=bootstrap)
    diag = scenario_diagnostic(events)

    if verbose:
        head = f"[{scenario}] " if scenario else ""
        print(f"{head}{len(events)} tests from {events.run.nunique()} runs"
              + (f", {len(pairs)} pairwise records" if len(pairs) else ""))
        print(f"\n--- Level 2: by {by} ---")
        cols = ["num_tests", "median_k", "median_n", "friedman_sig_rate",
                "all_tied_rate", "median_W", "median_Qvar", "P(Mmax>0)",
                "median_Mmax", "median_deltaR", "median_SE", "median_deltaR/SE",
                "eliminations_per_test"]
        print(binned[cols].to_string(float_format=lambda v: f"{v:.3f}"))
        print("\n--- Level 3: diagnostic ---")
        print(diag.to_string())

    fig = ax = None
    if plot:
        fig, ax = plot_racing_dynamics_boxplot(events, n_bins=n_bins, by=by,
                                               title=title or scenario,
                                               output_path=output_path,
                                               figsize=figsize, fontsizes=fontsizes,
                                               legend_loc=legend_loc,
                                               legend_ncol=legend_ncol,
                                               verbose=verbose)
    return ConoverResult(events=events, binned=binned, diagnostic=diag, pairs=pairs,
                         scenario=scenario, fig=fig, axes=ax)


def compare_scenarios(scenarios: dict, root=pathlib.Path("."), alpha: float = ALPHA,
                      n_bins: int = 5, by: str = "budget_fraction", savpop: bool = True,
                      keep_pairs: bool = False, plot: bool = True,
                      output_path=None, bootstrap: bool = False,
                      verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Investigate several scenarios and build the Level-3 comparison.

    Arguments:
        scenarios: ``{label: run_dirs}``, e.g.
            ``{"TSP": tsp_runs, "OBP": obp_runs, "FSSP": fssp_runs}``.

    Returns:
        ``(diagnostic_table, results)`` -- the Level-3 table with one column per
        scenario, and the per-scenario :class:`ConoverResult` objects. The pooled
        event frame is available as ``results["_events"]``.
    """
    results, cols, frames = {}, {}, []
    for label, dirs in scenarios.items():
        res = investigate_conover(dirs, root=root, scenario=label, alpha=alpha,
                                  n_bins=n_bins, by=by, savpop=savpop,
                                  keep_pairs=keep_pairs, plot=False,
                                  bootstrap=bootstrap, verbose=False)
        results[label] = res
        cols[label] = res.diagnostic
        frames.append(res.events)

    table = pd.DataFrame(cols)
    pooled = pd.concat(frames, ignore_index=True)
    results["_events"] = pooled

    if verbose:
        print("--- Level 3: cross-scenario diagnostic ---")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
        print(f"\n--- Level 2: by {by} ---")
        print(budget_bin_summary(pooled, n_bins=n_bins, by=by).to_string(
            float_format=lambda v: f"{v:.3f}"))

    if plot:
        results["_fig"], results["_axes"] = plot_budget_trajectories(
            pooled, title="Friedman–Conover decision chain vs. budget",
            output_path=output_path)
    return table, results


# --------------------------------------------------------------------------- #
# Offline entry point
# --------------------------------------------------------------------------- #
def _cli(argv=None) -> int:
    """Build event caches from the command line, in parallel.

    Two ways to name the runs:

      # explicit run directories
      python -m analyses.statistics_investigation --root . --n-jobs 20 \
          --scenario OBP-hetero --runs .logs/racing_.../run_0 .logs/.../run_1

      # or a JSON file of {scenario: [run_dirs, ...]} -- e.g. the notebook's
      # `settings` dict dumped with json.dump(settings, open("s.json", "w"))
      python -m analyses.statistics_investigation --root . --n-jobs 20 \
          --scenarios-json s.json --only AdaEva-R --label OBP-hetero
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract Friedman-Conover event tables and cache them "
                    "(the heavy step; the notebook then only visualises).")
    ap.add_argument("--root", default=".", help="repo root the run paths are relative to")
    ap.add_argument("--cache-dir", default=None,
                    help=f"where to write caches (default {CACHE_DIR})")
    ap.add_argument("--scenario", default=None, help="scenario label for --runs")
    ap.add_argument("--runs", nargs="*", default=None, help="run directories")
    ap.add_argument("--scenarios-json", default=None,
                    help='JSON file mapping {"label": [run_dirs...]}')
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict --scenarios-json to these keys")
    ap.add_argument("--label", default=None,
                    help="rename a single --only key in the cache")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="worker processes (-1 = all cores)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--no-savpop", action="store_true",
                    help="do not prefer the _savpop run directory")
    ap.add_argument("--keep-pairs", action="store_true",
                    help="also cache per-candidate-pair records (large)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave caches that already exist untouched")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable tqdm progress bars (they go to stderr)")
    a = ap.parse_args(argv)

    scenarios: dict = {}
    if a.scenarios_json:
        with open(a.scenarios_json) as fh:
            allsc = json.load(fh)
        keys = a.only if a.only else list(allsc)
        for k in keys:
            if k not in allsc:
                ap.error(f"{k!r} not in {a.scenarios_json} (have: {list(allsc)})")
            label = a.label if (a.label and len(keys) == 1) else k
            scenarios[label] = allsc[k]
    if a.runs:
        if not a.scenario:
            ap.error("--runs requires --scenario")
        scenarios[a.scenario] = a.runs
    if not scenarios:
        ap.error("give --runs/--scenario or --scenarios-json")

    written = build_events_cache(
        scenarios, root=a.root, cache_dir=a.cache_dir, alpha=a.alpha,
        savpop=not a.no_savpop, keep_pairs=a.keep_pairs, n_jobs=a.n_jobs,
        overwrite=not a.skip_existing, progress=not a.no_progress, verbose=True)
    print("\ncached:")
    for k, v in written.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
