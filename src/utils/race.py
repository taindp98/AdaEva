import math
import time
from typing import Any, Callable, Literal, Optional
import numpy as np

from .config import Config, RaceState
from .ranking import rank_costs, rank_matrix, cost_matrix
from .tests import friedman_eliminate, ttest_eliminate


# ----------------------------------------------------------------------------
# irace-style race log (mirrors the stdout table produced by R irace).
#
# R reference:
#   - table widths / column names: `packages/irace/R/race.R:305-312`.
#   - markers header: `packages/irace/R/race.R:313-320`.
#   - per-task row: `packages/irace/R/race.R:340-369` (`race_print_task_common`).
#   - footer: `packages/irace/R/race.R:399-412` (`race_print_footer`).
#   - iteration header: `packages/irace/R/irace.R:647-654`.
#   - concordance (rho, KenW): `packages/irace/R/utils.R:350-381`.
#   - dataVariance (Qvar): `packages/irace/R/utils.R:391-414`.
# ----------------------------------------------------------------------------

# Field widths for the no-capping table (R race.R:305).
_TABLE_WIDTHS = (1, 11, 11, 16, 16, 11, 8, 5, 4, 6)
_TABLE_COLS = (
    " ",
    "Instance",
    "Alive",
    "Best",
    "Mean best",
    "Exp so far",
    "W time",
    "rho",
    "KenW",
    "Qvar",
)

# R race.R:313-320. The `c` and `:` markers are capping-only paths in R
# (race.R:1000 sits inside `if (capping)`); utils implements no capping, so
# those markers are intentionally omitted — they can never be emitted here.
_MARKERS_HEADER = (
    "# Markers:\n"
    "     x No test is performed.\n"
    "     - The test is performed and some configurations are discarded.\n"
    "     = The test is performed but no configuration is discarded.\n"
    "     ! The test is performed and configurations could be discarded but "
    "elite configurations are preserved.\n"
    "     . Alive configurations were already evaluated on this instance and "
    "nothing is discarded.\n"
)


def print_markers_header() -> None:
    """Print the marker legend once, before the racing process starts.

    In R irace this legend is printed once per race (race.R:844); here it is
    a standalone call so it can be shown a single time at the top of an
    iterated run instead of repeating before every race.
    """
    print(_MARKERS_HEADER)


def _table_hline() -> str:
    """Render `+---+---+...` rule (R race.R:299-300)."""
    return "+" + "+".join("-" * w for w in _TABLE_WIDTHS) + "+"


def _table_row(cells: list[str]) -> str:
    """Render `| a | b |...` row, right-justified per column (R race.R:302-303)."""
    return "|" + "|".join(c.rjust(w) for c, w in zip(cells, _TABLE_WIDTHS)) + "|"


def _fmt_elapsed(seconds: float) -> str:
    """Format wall-clock seconds as HH:MM:SS (R race.R:333-338)."""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _fmt_perf(x: float) -> str:
    """Mirror R's `.irace.format.perf` = "%#16.10g" (R zzz.R:3)."""
    return f"{x:#.10g}"


def _concordance(data: np.ndarray) -> tuple[float, float]:
    """Kendall's W and Spearman's rho over a cost matrix.

    R reference: `concordance` in `packages/irace/R/utils.R:350-381`.
    `data` has instances in rows (judges) and configs in columns (objects).
    """
    n, k = data.shape  # n judges, k objects
    if n <= 1 or k <= 1:
        return float("nan"), float("nan")
    # Per-instance ranks with average ties (R: rowRanks ties.method="average").
    r = np.apply_along_axis(_avg_rank, 1, data)
    ties_sum = 0.0
    for row in r:
        _, counts = np.unique(row, return_counts=True)
        ties_sum += np.sum(counts**3 - counts)
    if np.allclose(r, r[:, :1]) and np.all(np.ptp(r, axis=1) == 0):
        w = 1.0
    else:
        col_sums = r.sum(axis=0)
        num = 12 * np.sum((col_sums - n * (k + 1) / 2) ** 2)
        den = (n**2 * (k**3 - k)) - (n * ties_sum)
        w = num / den if den != 0 else 1.0
    rho = (n * w - 1) / (n - 1)
    return float(w), float(rho)


def _avg_rank(row: np.ndarray) -> np.ndarray:
    """Rank a 1-D array with ties resolved by averaging (R ties.method='average')."""
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty(len(row), dtype=float)
    ranks[order] = np.arange(1, len(row) + 1, dtype=float)
    # Average ranks within tied groups.
    vals = row[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


def _data_variance(data: np.ndarray) -> float:
    """Instance-set heterogeneity in [0, 1].

    R reference: `dataVariance` in `packages/irace/R/utils.R:391-414`.
    0 = homogeneous instances, 1 = heterogeneous.
    """
    n, k = data.shape
    if n <= 1 or k <= 1:
        return float("nan")
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    z = (data - mean) / std
    return float(np.mean(z.var(axis=0, ddof=1)))


def print_iteration_header(
    iteration: int,
    n_iterations: int,
    experiments_used: int,
    remaining_budget: int,
    current_budget: int,
    nb_configurations: int,
) -> None:
    """Print the per-iteration banner (R irace.R:647-654)."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"# {stamp}: Iteration {iteration} of {n_iterations}")
    print(f"# experimentsUsed: {experiments_used}")
    print(f"# remainingBudget: {remaining_budget}")
    print(f"# currentBudget: {current_budget}")
    print(f"# nbConfigurations: {nb_configurations}")


def print_elite_configs(configs: list[Config], metric: str = "sum_ranks") -> None:
    """Print the elite-configuration listing shown between iterations.

    R reference: `packages/irace/R/irace.R` elite-print block after each race.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    key = (lambda c: c.sum_ranks) if metric == "sum_ranks" else (lambda c: c.mean_cost)
    ranked = sorted(configs, key=key)
    print(
        f"# {stamp}: Elite configurations (first number is the configuration"
        f" ID; listed from best to worst according to {metric}):"
    )
    if not ranked:
        print("  (none)")
        return
    param_names = list(ranked[0].params.keys())
    print("  ID   " + " ".join(f"{p:>10}" for p in param_names))
    for c in ranked:
        vals = " ".join(f"{str(c.params[p]):>10}" for p in param_names)
        print(f"  {str(c.id):<5}" + vals)


def _race_table_row(
    configurations: list[Config],
    current_task: int,
    inst_idx: int,
    seen_inst_indices: list[int],
    which_exe: list[Config],
    test_ran: bool,
    prev_nb_alive: int,
    nb_alive: int,
    protection_active: bool,
    experiments_used: int,
    start_time: float,
    metric: str,
) -> str:
    """Render one per-task row of the irace race table.

    Returns `(row_str, row_data)`: the formatted table line, plus a structured
    dict mirroring the Best / Mean best / Exp so far columns so callers can log
    the per-task race trajectory.

    R reference: `race_print_task_common` in `packages/irace/R/race.R:340-369`
    and the marker logic in `race.R:1150-1154`.
    """
    # Marker (R race.R:313-320, 1150-1154). Note: R prints `:` only inside the
    # capping pre-execution path (race.R:1000) — it is NOT a "protection is
    # active" marker. utils has no capping, so a non-test row is `.` when
    # every alive config already had a result for this instance, otherwise `x`
    # ("no test performed") — regardless of whether elites are present.
    if test_ran:
        if nb_alive < prev_nb_alive:
            marker = "-"  # test discarded some configs
        elif protection_active:
            marker = "!"  # could discard but elites preserved
        else:
            marker = "="  # test ran, nothing discarded
    elif not which_exe:
        marker = "."  # all alive already evaluated here
    else:
        marker = "x"  # no test performed

    alive = [c for c in configurations if c.alive]
    scores = _config_scores(alive, seen_inst_indices, metric)
    best = min(alive, key=lambda c: scores[id(c)]) if alive else None
    id_best = str(best.id) if best is not None else "NA"

    # Logged `mean_best` (the trajectory `score`, and `gap` derived from it):
    # the best config's *lifetime* mean over EVERY instance it has been
    # evaluated on (all of `costs_by_inst`), not just the instances seen in the
    # current race. This is a DELIBERATE divergence from irace, whose per-task
    # "Mean best" is per-race: `mean(Results[seq_len(current_task), best])`
    # (race.R:1178 slices to this race's tasks, race.R:352 takes the mean). We
    # report lifetime instead so a carried-over elite's score is stable across
    # generations and matches the final full-pool eval once coverage is complete
    # (consistent with the lifetime `n_instances` count below).
    if best is not None and best.costs_by_inst:
        mean_best_val = float(
            sum(best.costs_by_inst.values()) / len(best.costs_by_inst)
        )
    else:
        mean_best_val = float("inf")
    mean_best = _fmt_perf(mean_best_val) if best is not None else "NA"
    # Logged `n_instances`: the best config's *lifetime* evaluation coverage
    # (total distinct instances in `costs_by_inst`), NOT the per-race count.
    # A carried-over elite keeps its full history across races, so this does
    # not reset to 1 at each generation boundary the way `seen_inst_indices`
    # does — it reflects "how many instances the best-so-far was validated on".
    # `mean_best` above is averaged over exactly these same lifetime instances,
    # so `score` and `n_instances` are now consistent.
    n_seen_best = len(best.costs_by_inst) if best is not None else 0

    # Concordance over the balanced instance subset (R race.R:357-364).
    balanced = [k for k in seen_inst_indices if all(c.has(k) for c in alive)]
    if current_task > 1 and len(alive) > 1 and len(balanced) > 1:
        data = np.array([[c.costs_by_inst[k] for c in alive] for k in balanced])
        w, rho = _concordance(data)
        qvar = _data_variance(data)
        rho_s = f"{rho:+4.2f}"
        kenw_s = f"{w:.2f}"
        qvar_s = f"{qvar:.4f}"
    else:
        rho_s, kenw_s, qvar_s = "NA", "NA", "NA"

    row_str = _table_row(
        [
            marker,
            str(inst_idx),
            str(len(alive)),
            id_best,
            mean_best,
            str(experiments_used),
            _fmt_elapsed(time.time() - start_time),
            rho_s,
            kenw_s,
            qvar_s,
        ]
    )
    # Structured mirror of the three columns callers log a trajectory from:
    # Best (cand_id), Mean best (mean_best), Exp so far (experiments_used).
    row_data = {
        "inst_idx": inst_idx,
        "n_alive": len(alive),
        "cand_id": id_best,
        "mean_best": mean_best_val,
        "n_instances": n_seen_best,
        "experiments_used": experiments_used,
    }
    return row_str, row_data


def _debug_candidate_line(
    alive: list[Config],
    current_task: int,
    inst_idx: int,
    which_exe: list[Config],
) -> str:
    """One line listing every alive candidate's cost *on this instance*.

    Candidates are ordered best-first **by their cost on this very instance**
    (`inst_idx`) — i.e. the per-task-instance ranking — so the leftmost config
    is the one that performed best on this instance, regardless of its overall
    race standing. Each entry shows `id=cost(rank)<flags>`, where `rank` is the
    per-instance rank (1 = best on this instance) and the flag suffixes are:

        * — this config is an elite parent still under protection
            (`is_elite_credit > 0`): it cannot be eliminated yet.
        ~ — this config did NOT execute here (an elite reusing a prior
            result — skip-on-prior).

    A config can carry both (`(r2)*~` = a protected elite reusing a result).
    Configs with no result on this instance are listed last as `id=NA`.
    """
    exe_ids = {id(c) for c in which_exe}
    have = [c for c in alive if c.costs_by_inst.get(inst_idx) is not None]
    missing = [c for c in alive if c.costs_by_inst.get(inst_idx) is None]
    have.sort(key=lambda c: c.costs_by_inst[inst_idx])
    ranks = rank_costs([c.costs_by_inst[inst_idx] for c in have]) if have else []
    parts = []
    for c, r in zip(have, ranks):
        elite = "*" if c.is_elite_credit > 0 else ""
        skip = "" if id(c) in exe_ids else "~"
        parts.append(f"{c.id}={c.costs_by_inst[inst_idx]:.4g}(r{r:g}){elite}{skip}")
    parts += [f"{c.id}=NA{'*' if c.is_elite_credit > 0 else ''}" for c in missing]
    return f"  task {current_task:>2} inst {inst_idx:>3}: " + "  ".join(parts)


def _debug_global_rank_line(
    alive: list[Config],
    seen_inst_indices: list[int],
    metric: str,
) -> str:
    """One line listing every alive candidate's *global* race standing so far.

    Where `_debug_candidate_line` shows the per-instance (local) ranking — who
    won this one instance — this shows the cumulative cross-instance ranking:
    the same score used to choose `Best` and to decide which top-`N_min`
    candidates survive into the next iteration (R race.R:1160-1171).

    The score is `_config_scores` over every instance seen up to and including
    the current task. Candidates are ordered best-first; each entry is
    `id(gN)` with `gN` the 1-based global rank (1 = current overall leader).
    """
    if not alive:
        return " " * 21 + "global: (none alive)"
    scores = _config_scores(alive, seen_inst_indices, metric)
    ordered = sorted(alive, key=lambda c: scores[id(c)])
    granks = rank_costs([scores[id(c)] for c in ordered])
    parts = [f"{c.id}(g{r:g})" for c, r in zip(ordered, granks)]
    return " " * 14 + "global rank:  " + "  ".join(parts)


def _debug_cost_matrix(configurations: list[Config], inst_order: list[int]) -> str:
    """Full per-config cost matrix over the instances processed this race.

    Rows = configs (with alive/elim status), columns = 1-based instance ids in
    processing order. A blank cell means the config was never evaluated there.
    Useful as an end-of-race dump to inspect the whole pool at once.
    """
    if not inst_order:
        return "  (no instances processed)"
    lines = ["  --- cost matrix (rows=configs, cols=instances) ---"]
    head = (
        "  " + "config".ljust(12) + "stat  " + "".join(f"{i:>10}" for i in inst_order)
    )
    lines.append(head)
    for c in configurations:
        status = "alive" if c.alive else "elim "
        cells = "".join(
            f"{c.costs_by_inst[i]:>10.4g}" if i in c.costs_by_inst else f"{'.':>10}"
            for i in inst_order
        )
        lines.append("  " + str(c.id).ljust(12) + status + " " + cells)
    return "\n".join(lines)


def _elitist_init_instances(
    next_instance: int,
    n_total: int,
    elitist_new_instances: int,
    sample_instances: bool,
    deterministic: bool,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Build the race's instance ordering for elitist mode.

    R reference: `packages/irace/R/race_state.R:244-280`
    (`elitist_init_instances`). The R function returns
    `c(new_instances, past_instances, future_instances)` — `past_instances`
    is `sample.int(next_instance - 1L)` when `sampleInstances` is TRUE,
    else `seq_len(next_instance - 1L)`. Our implementation mirrors that
    line-for-line.

    Returns the 1-based instance indices in race order:
        [new_instances..., shuffled past_instances..., future_instances...]

    The "new" instances are prepended (T^new of them), followed by past
    instances seen in prior races (shuffled when `sample_instances=True`),
    followed by any not-yet-seen future instances in their original order.

    On the very first race (`next_instance == 1`) the full pool is returned
    in original order, matching R.

    Returns:
        (race_instances, effective_t_new) where `effective_t_new` is the
        actual number of new instances added (may be less than the requested
        `elitist_new_instances` in the deterministic-and-limited case).
    """
    if next_instance == 1:
        return list(range(1, n_total + 1)), 0

    new_block: list[int] = []
    last_new = next_instance - 1 + elitist_new_instances
    effective_t_new = elitist_new_instances

    if elitist_new_instances > 0:
        if last_new > n_total:
            if not deterministic:
                # Should not happen unless caller supplied too-few instances.
                last_new = n_total
            if next_instance <= n_total:
                last_new = n_total
                new_block = list(range(next_instance, last_new + 1))
                effective_t_new = len(new_block)
            else:
                effective_t_new = 0
        else:
            new_block = list(range(next_instance, last_new + 1))

    past_indices = list(range(1, next_instance))
    if sample_instances and not deterministic:
        past_perm = rng.permutation(len(past_indices)).tolist()
        past_block = [past_indices[i] for i in past_perm]
    else:
        past_block = past_indices

    future_block: list[int] = []
    if last_new + 1 <= n_total:
        future_block = list(range(last_new + 1, n_total + 1))

    return new_block + past_block + future_block, effective_t_new


def _no_elitist_init_instances(
    next_instance: int,
    n_total: int,
    deterministic: bool,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Build the race's instance ordering for NON-elitist mode.

    Always races the full pool (1..n_total). When not deterministic the order
    is shuffled per-race via `rng` to reduce overfitting to a fixed evaluation
    order and increase the chance of discriminating candidates early.
    """
    indices = list(range(1, n_total + 1))
    if not deterministic and rng is not None:
        rng.shuffle(indices)
        print(
            f"  [instances] shuffled instance order: {indices}"
        )
    return indices


def _update_is_elite_credit(configs: list[Config], which_exe_ids: set[str]) -> None:
    """Decrement `is_elite_credit` for elites that did NOT execute this task.

    R reference: `packages/irace/R/race.R:593-599` (`update_is_elite`).
    Elites that were skipped on this task because they already had a prior
    result have their remaining protection-window credit decreased by one.
    """
    for c in configs:
        if c.is_elite_credit > 0 and c.id not in which_exe_ids:
            c.is_elite_credit -= 1


def _reject_invalid(
    which_exe: list[Config], inst_idx: int, crash_penalty: Optional[float] = None
) -> None:
    """Handle configs whose result on `inst_idx` is non-finite (a crash).

    Two modes, selected by the racing runner's ``--deal-with-crashed``:

    - ``crash_penalty is None`` — REJECTION (default, faithful R `irace`,
      `packages/irace/R/race.R:1064-1073`): an infinite cost denotes an
      *invalid* result (a crash/exception in the generated heuristic, as opposed
      to a finite timeout penalty, which is kept). irace rejects such a
      configuration immediately: it strips elite protection (`is_elite[...] <- 0`)
      and removes it from the alive set (`alive[...] <- FALSE`), so a broken
      heuristic cannot survive behind the protection window.
    - ``crash_penalty is not None`` — PENALTY: convert the crash to this finite
      worst-case cost and KEEP the config alive, so it undergoes the
      Friedman/Conover test instead of being dropped. This preserves a fragile
      (e.g. LLaMEA) population from collapsing when generated heuristics crash
      frequently. The config still ranks last on this task.

    Only the configs evaluated this task (`which_exe`) get a fresh result, so
    only they are checked. A config carrying a *finite* timeout penalty from a
    prior race is unaffected — that is the timeout/rejection split irace makes.
    Crashes are never marked ``timed_out_insts``, so PENALTY-kept configs are not
    swept by `_reject_timed_out`.
    """
    for c in which_exe:
        cost = c.costs_by_inst.get(inst_idx)
        if cost is not None and not np.isfinite(cost):
            if crash_penalty is not None:
                pen = float(crash_penalty)
                c.costs_by_inst[inst_idx] = pen
                # Keep the legacy costs list consistent (this task's record is
                # the last appended entry for every just-evaluated config).
                if c.costs and not np.isfinite(c.costs[-1]):
                    c.costs[-1] = pen
                continue
            was_protected = c.is_elite_credit > 0
            c.is_elite_credit = 0
            c.is_elite = False
            c.alive = False
            print(
                f"  [reject-crash] {c.id} returned invalid (inf) on inst "
                f"{inst_idx}"
                + ("  (was protected elite)" if was_protected else "")
                + " -> protection stripped + eliminated",
                flush=True,
            )


def _reject_timed_out(configs: list[Config]) -> None:
    """Strip protection from and eliminate any config that has ever timed out.

    Unlike a finite timeout *penalty* (which irace keeps), this project treats a
    timeout as disqualifying: a configuration that was killed by the per-eval
    wall-clock cap on ANY instance — even a protected elite, and even if the
    timeout happened in a prior race (carried over via `timed_out_insts`) — has
    its elite protection removed (`is_elite_credit = 0`) and is eliminated
    (`alive = False`). This diverges from irace's PAR-penalty semantics by
    design: a heuristic too slow to finish within the cap should not survive
    behind the protection window.
    """
    for c in configs:
        if c.timed_out_insts and (c.alive or c.is_elite_credit > 0):
            # Only act/print on the transition (a config still alive or still
            # carrying protection); a config already disqualified stays quiet.
            was_protected = c.is_elite_credit > 0
            c.is_elite_credit = 0
            c.is_elite = False
            c.alive = False
            print(
                f"  [disqualify-timeout] {c.id} timed out on inst(s) "
                f"{sorted(c.timed_out_insts)}"
                + ("  (was protected elite)" if was_protected else "")
                + " -> protection stripped + eliminated",
                flush=True,
            )


def _record_eval_meta(c, inst_idx, wall_s, task_step) -> None:
    """Record per-(config, instance) eval metadata (``wall_s``, ``race_step``) for the
    manuscript ``instance_seed_perf`` log. Best-effort; never raises into the race."""
    try:
        c.meta_by_inst[inst_idx] = {
            "wall_s": (round(float(wall_s), 4) if wall_s is not None else None),
            "race_step": (int(task_step) if task_step is not None else None),
        }
    except Exception:
        pass


def _eval_one_instance_elitist(
    state: RaceState,
    inst_idx: int,
    instance: Any,
    which_exe: list[Config],
    target_runner: Callable,
    target_evaluator: Optional[Callable],
    eval_pool: Optional[Any] = None,
    pool_runner: Optional[Callable] = None,
    eval_timeout: Optional[float] = None,
    timeout_cost: float = float("inf"),
    crash_penalty: Optional[float] = None,
    pool_recreate: Optional[Callable] = None,
    task_step: Optional[int] = None,
) -> Optional[Any]:
    """Evaluate only `which_exe` configs on `instance` and record by inst_idx.

    R reference: `packages/irace/R/race.R:1022-1058` (the
    `race_wrapper(...)` call site that runs only `which_exe` configs and
    writes their costs into `Results[current_task, which_has_cost]`).
    Our function is the per-task scalar equivalent.

    Does NOT touch the legacy `costs` / `ranks` lists for elites that were
    skipped — only the configs in `which_exe` get a new observation.

    Branches (see inline comments):
        0. `which_exe` empty -> nothing to run, return.
        1. parallel process-pool scoring (opt-in: `eval_pool` supplied).
        2. serial in-process scoring -- THE USUAL CASE (default path).
        3. batch `target_evaluator` scoring (cross-config metric).

    Per-evaluation timeout (branch 1 only):
        When `eval_timeout` is set, each parallel evaluation is collected with
        `fut.result(timeout=eval_timeout)`. A heuristic that hangs (or runs
        longer than the cap) is given `timeout_cost` — so the Friedman test
        eliminates it like any poor performer. Because a `ProcessPoolExecutor`
        cannot kill one runaway worker without breaking the pool, the whole
        pool is then torn down and rebuilt via `pool_recreate(old_pool)`, and
        the *other* configs of this task (which shared the dead pool) are
        salvaged if already finished, else re-submitted to the fresh pool for
        a fair full re-evaluation.

    Returns:
        The process pool to use for the next task — the same `eval_pool` in
        the common case, or a freshly rebuilt one if a timeout forced a
        recreate. Non-parallel branches return `eval_pool` unchanged.
    """
    # Branch 0 — nothing to run. Every alive config already holds a result
    # for this instance (all elites, all skipped). The irace table marks
    # such a task with `.`. Return without touching any counter.
    if not which_exe:
        return eval_pool

    if target_evaluator is None:
        # --- Independent per-config scoring (no cross-config metric). ---
        if eval_pool is not None and pool_runner is not None and len(which_exe) >= 1:
            # Branch 1 — POOL DISPATCH. Opt-in: a process pool is supplied.
            # Each config is scored in its own worker process via
            # ``pool_runner(config, instance)``; the parent collects results
            # in order and records them, so all ``state`` / ``Config``
            # mutation stays single-threaded. The MAB driver uses this path
            # even for single-config batches (``len(which_exe) == 1``)
            # because ``Future.result(timeout=eval_timeout)`` is the only
            # in-tree mechanism for enforcing a wall-clock cap on
            # arbitrary LLM-generated heuristics that may infinite-loop.
            # Used for
            # CPU-bound heuristics when the caller passes `eval_pool`.
            from concurrent.futures import TimeoutError as _CFTimeout
            from concurrent.futures.process import BrokenProcessPool as _BrokenPool

            # A pool_runner may return either a bare cost or a
            # ``(cost, cpu_seconds)`` pair; ``_split`` accepts both, folding the
            # per-worker CPU time (summed across cores) into ``state.cpu_seconds``.
            # Returns ``(cost, cpu_seconds_or_None)``. The per-worker CPU (≈ wall for the
            # single-threaded CPU-bound evals) is folded into ``state.cpu_seconds`` AND
            # recorded per (config, instance) as ``wall_s`` for instance_seed_perf.
            def _split(res):
                if isinstance(res, tuple):
                    state.cpu_seconds += float(res[1])
                    return float(res[0]), float(res[1])
                return float(res), None

            n = len(which_exe)
            costs: list[Optional[float]] = [None] * n
            walls: list[Optional[float]] = [None] * n
            futures = [eval_pool.submit(pool_runner, c, instance) for c in which_exe]
            idx = 0
            while idx < n:
                if costs[idx] is not None:
                    # Already filled (salvaged after a pool recreate below).
                    idx += 1
                    continue
                try:
                    costs[idx], walls[idx] = _split(futures[idx].result(timeout=eval_timeout))
                except (_CFTimeout, _BrokenPool):
                    # This evaluation hung past `eval_timeout` (or the pool
                    # broke). Penalise this config; a runaway worker will not
                    # stop on its own, so kill+rebuild the whole pool.
                    costs[idx] = timeout_cost
                    walls[idx] = float(eval_timeout) if eval_timeout else None
                    # The killed worker still burned ~eval_timeout of CPU on its
                    # core before being cut off — count it so the CPU odometer
                    # reflects the real compute spent.
                    if eval_timeout:
                        state.cpu_seconds += float(eval_timeout)
                    # Mark the timeout explicitly so the race can strip this
                    # config's elite protection (see `_reject_timed_out`),
                    # independent of the finite penalty value recorded above.
                    _to_cfg = which_exe[idx]
                    _to_cfg.timed_out_insts.add(inst_idx)
                    print(
                        f"  [timeout] {_to_cfg.id} hit eval_timeout on inst "
                        f"{inst_idx}"
                        + ("  (was protected elite)" if _to_cfg.is_elite_credit > 0 else "")
                        + " -> penalised + marked for disqualification",
                        flush=True,
                    )
                    if pool_recreate is None:
                        raise
                    # Salvage siblings that already finished, BEFORE the kill.
                    done_now: dict[int, float] = {}
                    for j in range(idx + 1, n):
                        if futures[j].done():
                            try:
                                done_now[j] = _split(futures[j].result(timeout=0))
                            except Exception:
                                pass
                    eval_pool = pool_recreate(eval_pool)
                    # Re-submit siblings that had not finished -> fair, full
                    # re-evaluation on the fresh pool.
                    for j in range(idx + 1, n):
                        if j in done_now:
                            costs[j], walls[j] = done_now[j]
                        else:
                            futures[j] = eval_pool.submit(
                                pool_runner, which_exe[j], instance
                            )
                except Exception:
                    # Worker raised a non-timeout exception (e.g. SyntaxError,
                    # NameError from LLM-generated code). Assign penalty and
                    # continue — no pool recreate needed, the pool is intact.
                    costs[idx] = timeout_cost
                    walls[idx] = None
                idx += 1
            for c, cost, wall in zip(which_exe, costs, walls):
                c.record(inst_idx, float(cost))
                state.total_evaluations += 1
                _record_eval_meta(c, inst_idx, wall, task_step)
            _reject_invalid(which_exe, inst_idx, crash_penalty)
            _reject_timed_out(which_exe)
            return eval_pool
        else:
            # Branch 2 — SERIAL, in-process. THE USUAL CASE: this runs
            # whenever no `eval_pool` was supplied (the default), and is also
            # the fallback when only one config needs running. Each config is
            # scored sequentially via `target_runner(c.params, instance)`.
            # Note: `c.params` for ConfigAS resolves to `c.callable` which
            # calls exec() — that can raise SyntaxError or other compile
            # errors before target_runner is entered. Catching here keeps
            # Branch 2 consistent with Branch 1's exception handling.
            for c in which_exe:
                _w0 = time.perf_counter()
                try:
                    res = target_runner(c.params, instance)
                    # target_runner may return (cost, cpu_seconds) or a bare cost.
                    if isinstance(res, tuple):
                        state.cpu_seconds += float(res[1])
                        cost = float(res[0])
                    else:
                        cost = float(res)
                except Exception:
                    cost = timeout_cost
                _wall = time.perf_counter() - _w0     # exact per-execution wall
                c.record(inst_idx, cost)
                state.total_evaluations += 1
                _record_eval_meta(c, inst_idx, _wall, task_step)
            _reject_invalid(which_exe, inst_idx, crash_penalty)
            _reject_timed_out(which_exe)
    else:
        # Branch 3 — BATCH EVALUATOR. Used only when the caller supplies a
        # `target_evaluator` because the cost depends on cross-config
        # information for this instance: `target_runner` here returns raw
        # *artifacts*, then the evaluator turns the whole batch into costs in
        # one call. Always serial (the evaluator needs every artifact at once).
        _b0 = time.perf_counter()
        artifacts = [target_runner(c.params, instance) for c in which_exe]
        state.total_evaluations += len(which_exe)
        costs = target_evaluator(which_exe, instance, artifacts)
        _bwall = (time.perf_counter() - _b0) / max(1, len(which_exe))  # amortized per config
        for c, cost in zip(which_exe, costs):
            c.record(inst_idx, float(cost))
            _record_eval_meta(c, inst_idx, _bwall, task_step)
        _reject_invalid(which_exe, inst_idx, crash_penalty)
        _reject_timed_out(which_exe)
    return eval_pool


def _run_test(state: RaceState, alpha: float) -> None:
    """Fire the configured elimination test on the currently alive set.

    Used by TSP racing scripts that manage their own evaluation loop and call
    this directly after each instance batch. Sets `c.alive = False` on every
    config the test eliminates; the best config is always kept. Does nothing
    if fewer than 2 configs are still alive.
    """
    alive = state.alive_configs()
    if len(alive) < 2:
        return
    n = state.evaluated_instances
    if state.test_type == "friedman":
        rmat = rank_matrix(alive, n)
        _, keep, _p = friedman_eliminate(rmat, alpha=alpha, posthoc_test_type=state.posthoc_test_type)
    else:
        cmat = cost_matrix(alive, n)
        _, keep, _p = ttest_eliminate(cmat, alpha=alpha)
    for c, k in zip(alive, keep):
        if not k:
            c.alive = False


def _cost_matrix_by_inst(configs: list[Config], inst_indices: list[int]) -> np.ndarray:
    """Build a (n_configs, n_instances) cost matrix using `costs_by_inst`.

    R reference: `packages/irace/R/race.R:1116` (`do_test(Results[seq_len(
    current_task), ], alive, which_alive)`). R passes the full `Results`
    matrix sliced to `seq_len(current_task)`; we slice by explicit instance
    indices because our storage is a sparse dict rather than a dense matrix.

    Order of rows matches `configs`; order of columns matches `inst_indices`.
    Used by the elitist race's test step so the test sees results regardless
    of the order in which each config encountered each instance (elites may
    have results for instance X from a prior race; non-elites only after
    executing X in the current race).
    """
    n_c, n_i = len(configs), len(inst_indices)
    out = np.empty((n_c, n_i), dtype=float)
    for i, c in enumerate(configs):
        for j, k in enumerate(inst_indices):
            out[i, j] = c.costs_by_inst[k]
    return out


def _rank_matrix_by_inst(configs: list[Config], inst_indices: list[int]) -> np.ndarray:
    """Per-instance ranks across the given (alive) configs.

    R reference: `packages/irace/R/race.R:1166` (`get_ranks(tmpResults,
    test = stat_test)`). R recomputes ranks from the slice of `Results`
    rather than reading the stored `race_ranks`, for the same reason we do:
    the alive set during the race may differ from the alive set at original
    evaluation time.

    Ranks are recomputed here from `costs_by_inst` rather than read from the
    stored `ranks_by_inst`, because the alive set during the elitist race
    may differ from the alive set at original evaluation time.
    """
    cmat = _cost_matrix_by_inst(configs, inst_indices)
    rmat = np.empty_like(cmat)
    for j in range(cmat.shape[1]):
        rmat[:, j] = rank_costs(cmat[:, j].tolist())
    return rmat


def _config_scores(
    configs: list[Config],
    seen_inst_indices: list[int],
    metric: str,
) -> dict[int, float]:
    """Lower-is-better score per config, used to pick `Best` and order survivors.

    R reference: `packages/irace/R/race.R:560-589` (`overall_ranks`) and the
    comment at 563-565: *"Given two configurations, the one evaluated on more
    instances is ranked better. Otherwise, break ties according to the criteria
    of the stat test."* Elite selection / survivor ordering must respect this
    instance-count tiering, otherwise a freshly-sampled offspring evaluated on
    only a handful of (lucky) instances can out-rank a long-lived elite that has
    been evaluated on the full pool — the "false alarm" this guards against.

    For the rank metric we therefore mirror `overall_ranks`:
        1. Group configs by how many instances they have been evaluated on. The
           count is the config's *lifetime* coverage (`len(c.costs_by_inst)`),
           mirroring irace's `ninstances <- colSums2(!is.na(Results))`
           (race.R:570), where `Results` is pre-seeded with carried-over elite
           data (race.R:758). It is NOT the per-race `seen_inst_indices` count:
           early in a race only a few instances are processed, which would put a
           long-lived elite in the same tier as a fresh offspring and let the
           offspring win — the exact false alarm this guards against.
        2. Process groups from most-evaluated to least-evaluated. Within a group
           rank only over the instances every member of *that group* shares
           (the per-tier balanced subset = irace's `complete.cases`), using the
           same per-instance rank-sum as irace's `get_ranks`.
        3. Offset each lower tier's ranks by the max rank of the tier above, so
           any config with more instances is strictly better than any config
           with fewer — the statistical rank only breaks ties *within* a tier.

    In elitist mode `_eval_one_instance_elitist` records only costs (no ranks),
    so `Config.sum_ranks` is 0 for every config; ranks are recomputed here from
    `costs_by_inst`.

    Keyed by `id(config)`. Falls back to `mean_cost` when `metric='mean_cost'`.
    When every config has the same lifetime coverage (the common mid-race case)
    this reduces to a single tier — i.e. the plain rank-sum, unchanged.

    `seen_inst_indices` does not affect the tier count; the within-tier balanced
    subset is intersected with each tier's own instances below.
    """
    if not configs:
        return {}
    if metric == "mean_cost" or len(configs) < 2:
        return {id(c): c.mean_cost for c in configs}

    def _n_lifetime(c: Config) -> int:
        # Lifetime instance coverage = irace's colSums2(!is.na(Results)).
        # Any candidate with a crash penalty (cost >= 1e5) is assigned 0 coverage
        # so it is placed in the lowest tier behind all valid non-crashed candidates.
        if any(v >= 1e5 for v in c.costs_by_inst.values()):
            return 0
        return len(c.costs_by_inst)

    # Group configs by lifetime instance count, most-evaluated first.
    counts = {id(c): _n_lifetime(c) for c in configs}
    tiers = sorted({n for n in counts.values()}, reverse=True)

    scores: dict[int, float] = {}
    last_r = 0.0
    for n in tiers:
        members = [c for c in configs if counts[id(c)] == n]
        if n == 0:
            # No instances at all — keep relative order stable, just place them
            # strictly after every more-evaluated config.
            for c in members:
                scores[id(c)] = last_r + 1.0
            last_r = last_r + 1.0
            continue
        # Within-tier balanced subset: instances every member of THIS tier
        # shares (irace's `complete.cases` over the tier's columns). Members of
        # one tier all have the same lifetime count but not necessarily the same
        # instances, so intersect their `costs_by_inst` keys.
        common = set.intersection(*(set(c.costs_by_inst) for c in members))
        balanced = sorted(common)
        if len(members) < 2 or not balanced:
            # Single member, or nothing shared within the tier: rank 1 in-tier.
            for c in members:
                scores[id(c)] = last_r + 1.0
            last_r = last_r + 1.0
            continue
        cmat = _cost_matrix_by_inst(members, balanced)
        rsum = np.zeros(len(members))
        for j in range(cmat.shape[1]):
            rsum += np.asarray(rank_costs(cmat[:, j].tolist()), dtype=float)
        for c, r in zip(members, rsum):
            scores[id(c)] = last_r + float(r)
        last_r = last_r + float(rsum.max())
    return scores


def _run_elitist_test(
    state: RaceState,
    alive: list[Config],
    seen_inst_indices: list[int],
    alpha: float,
) -> tuple[bool, bool, Optional[float]]:
    """Run the elimination test on the alive set over all seen instances.

    R reference:
        - test dispatch (friedman / t.none / t.holm / t.bonferroni):
          `packages/irace/R/race.R:674-682` (`do_test <- switch(stat_test, ...)`).
        - test schedule and merge with `cap_alive`:
          `packages/irace/R/race.R:1110-1135`.
        - force-keeping elites with `is_elite > 0`:
          `packages/irace/R/race.R:1131-1135`
          (`alive <- alive | (is_elite > 0L)`).

    Returns `(test_ran, any_dropped, p_value)` (p is the omnibus/vs-best p, or
    None when no test ran). Elites whose protection-window credit is still
    positive are force-kept.
    """
    if len(alive) < 2 or len(seen_inst_indices) < 2:
        return False, False, None
    if state.test_type == "friedman":
        rmat = _rank_matrix_by_inst(alive, seen_inst_indices)
        _, keep, pval = friedman_eliminate(rmat, alpha=alpha, posthoc_test_type=state.posthoc_test_type)
    else:
        cmat = _cost_matrix_by_inst(alive, seen_inst_indices)
        _, keep, pval = ttest_eliminate(cmat, alpha=alpha)

    any_dropped = False
    for c, k in zip(alive, keep):
        if not k and c.is_elite_credit == 0:
            c.alive = False
            any_dropped = True
    return True, any_dropped, pval


def _render_debug_table(
    configurations: list[Config],
    inst_order: list[int],
    debug_inst_data: dict[int, dict[str, tuple[float, float, bool, bool]]],
    debug_elim_at: dict[str, int],
    instance_label: Optional[Callable[[int], str]] = None,
) -> str:
    """Render the refined per-instance debug table with markers.

    Structure: ID | Status | Inst <idx> ... | Rank
    Handles horizontal overflow by chunking instances.

    ``instance_label``: optional ``inst_idx -> str`` labeler. When supplied, the
    per-instance column headers and the ``Elim @`` status use its label (e.g.
    ``i<base_idx>_s<seed>``) instead of the bare ``Inst <inst_idx>``; the columns
    are widened to fit. When ``None`` the legacy ``Inst <idx>`` labels are used.
    """
    # One labeler for both the column headers and the `Elim @` status, so the
    # two always agree. `inst_hdr` maps a 1-based inst_idx to its display label.
    inst_hdr = instance_label if instance_label is not None else (lambda i: f"Inst {i}")
    if not inst_order:
        return "  (no instances processed)"

    # Calculate total rank sum for each config (over all instances they were in)
    config_rsums = {}
    for c in configurations:
        rsum = 0.0
        for inst_idx in inst_order:
            if inst_idx in debug_inst_data and c.id in debug_inst_data[inst_idx]:
                rsum += debug_inst_data[inst_idx][c.id][1]
        config_rsums[c.id] = rsum

    # Sort configs: alive first (by rsum), then eliminated (by rsum)
    def sort_key(c):
        return (0 if c.alive else 1, config_rsums[c.id])

    ordered_configs = sorted(configurations, key=sort_key)

    # Assign ranks based on the sorted order (handling ties)
    id2rank = {}
    for i, c in enumerate(ordered_configs):
        if i > 0 and sort_key(ordered_configs[i]) == sort_key(ordered_configs[i - 1]):
            id2rank[c.id] = id2rank[ordered_configs[i - 1].id]
        else:
            id2rank[c.id] = float(i + 1)

    # Column widths. When a custom labeler is in play the labels are wider
    # (e.g. `i127_s2147483647` = 16 chars, `Elim @ i127_s2147483647` = 23), so
    # widen the instance and status columns to fit; otherwise keep the legacy
    # widths for the compact `Inst <idx>` layout.
    max_id_len = max((len(str(c.id)) for c in configurations), default=12)
    col_id_w = max(12, max_id_len + 1)
    if instance_label is not None:
        col_stat_w = 26
        col_inst_w = 18
    else:
        col_stat_w = 16
        col_inst_w = 16
    col_rank_w = 10

    # We chunk instances into groups of 4 to fit terminal width
    chunk_size = 4
    all_chunks = [
        inst_order[i : i + chunk_size] for i in range(0, len(inst_order), chunk_size)
    ]

    output_lines = []

    for chunk_idx, chunk in enumerate(all_chunks):
        if len(all_chunks) > 1:
            output_lines.append(
                f"\n--- (Instances {chunk_idx * chunk_size + 1} to {chunk_idx * chunk_size + len(chunk)}) ---"
            )

        header_cols = ["ID", "Status"] + [inst_hdr(i) for i in chunk] + ["Rank"]
        widths = [col_id_w, col_stat_w] + [col_inst_w] * len(chunk) + [col_rank_w]

        hline = "+" + "+".join("-" * w for w in widths) + "+"
        output_lines.append(hline)

        # Header row
        output_lines.append(
            "|" + "|".join(c.center(w) for c, w in zip(header_cols, widths)) + "|"
        )
        output_lines.append(hline)

        for c in ordered_configs:
            # ID
            row_cells = [c.id.center(col_id_w)]

            # Status
            if c.alive:
                status = "Alive"
            else:
                elim_inst = debug_elim_at.get(c.id, "?")
                if instance_label is not None and elim_inst != "?":
                    status = f"Elim @ {inst_hdr(elim_inst)}"
                else:
                    status = f"Elim @ Inst {elim_inst}"
            row_cells.append(status.center(col_stat_w))

            # Instances in this chunk
            for inst_idx in chunk:
                if inst_idx in debug_inst_data and c.id in debug_inst_data[inst_idx]:
                    cost, rank, prot, skip = debug_inst_data[inst_idx][c.id]
                    markers = ""
                    if prot:
                        markers += "*"
                    if skip:
                        markers += "~"

                    r_str = f"r{rank:g}"
                    cell = f"{cost:.4g} ({r_str}){markers}"
                    row_cells.append(cell.center(col_inst_w))
                else:
                    row_cells.append("--".center(col_inst_w))

            # Rank (final standing in the race)
            row_cells.append(f"{id2rank[c.id]:g}".center(col_rank_w))

            output_lines.append("|" + "|".join(row_cells) + "|")

        output_lines.append(hline)

    return "\n".join(output_lines)


def elitist_race(
    configurations: list[Config],
    target_runner: Callable[[dict, Any], float],
    instances_log: list[Any],
    max_experiments: int,
    next_instance: int = 1,
    elitist: bool = True,
    elitist_new_instances: int = 1,
    elitist_limit: int = 2,
    early_stopping_non_elitist: bool = False,
    first_test: int = 5,
    each_test: int = 1,
    test_type: Literal["friedman", "ttest"] = "friedman",
    posthoc_test_type: Literal["conover", "nemenyi"] = "conover",
    metric: Literal["sum_ranks", "mean_cost"] = "sum_ranks",
    alpha: float = 0.05,
    sample_instances: bool = True,
    deterministic: bool = False,
    target_evaluator: Optional[Callable] = None,
    seed: Optional[int] = None,
    min_survival: int = 1,
    verbose: int = 1,
    eval_pool: Optional[Any] = None,
    pool_runner: Optional[Callable] = None,
    eval_timeout: Optional[float] = None,
    timeout_cost: float = float("inf"),
    crash_penalty: Optional[float] = None,
    pool_recreate: Optional[Callable] = None,
    instance_label: Optional[Callable[[int], str]] = None,
) -> dict:
    """A single elitist race over a pool of pre-built `Config` objects.

    R reference:
        - top-level function: `packages/irace/R/race.R:633-1240`
          (`elitist_race <- function(...)`).
        - instance ordering: elitist -> `elitist_init_instances`
          (`race_state.R:244-280`); non-elitist -> `no_elitist_init_instances`
          (`race_state.R:234-242`). The shared `next_instance` pointer is
          advanced by the driver (`irace.R:1144`).
        - protection window / `elite_safe`: `packages/irace/R/race.R:735-740,
          853-893, 1131-1135`.
        - `elitist_limit` early-stop:
          `packages/irace/R/race.R:918-922, 1184-1191`.
        - `minSurvival` (= `min_survival`): `packages/irace/R/race.R:897-902`.
        - budget guard:
          `packages/irace/R/race.R:910-916`.

    Configs whose `costs_by_inst` is non-empty on entry are treated as
    elites carried over from a previous race; their existing results seed the per-instance cost
    matrix and their `is_elite_credit` is set to the number of past
    instances they bring with them. Freshly-sampled configs (empty
    `costs_by_inst`) participate as non-elites.

    Returns:
        A dict with:
            - "survivors": alive `Config` objects, sorted by `metric`.
            - "race_instances": 1-based instance indices in the order they
              were *scheduled* for this race (the full planned ordering).
            - "seen_instances": the 1-based instance indices actually
              processed (a prefix of "race_instances"; shorter when the race
              stopped early).
            - "elitist_new_instances": the effective T^new actually used
              (may differ from the request in the deterministic/limited case).
            - "next_instance": updated shared 1-based pointer to feed into the
              next race in an iterated loop — one past the highest *distinct*
              instance evaluated so far. Use this for BOTH modes.
            - "break_reason": human-readable string describing why the race
              ended (matches R's `break_msg`).
            - "experiments_used": number of (config, instance) evaluations
              consumed in this race.

    Notes:
        - `elitist=True` and `elitist=False` differ in two ways: (1) elites are
          protected from elimination during the protection window, and (2) the
          instance ordering — elitist prepends T^new new instances and re-uses
          past instances, whereas non-elitist races only the fresh forward
          slice `next_instance:n_total` and never replays past instances.
        - `instances_log` must contain enough entries for the whole race;
          `next_instance + elitist_new_instances - 1 <= len(instances_log)`
          except in the deterministic/limited case.
        - `verbose` integer flag:
            - 0: silent
            - 1: prints the irace-style aggregate race table.
            - 2: additionally prints the detailed per-instance performance
              matrix for all configurations (including survivors and eliminated ones).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    n_total = len(instances_log)
    # Instance ordering: elitist and non-elitist use *different* R functions.
    #   elitist     -> elitist_init_instances    (race_state.R:244-280)
    #   non-elitist -> no_elitist_init_instances  (race_state.R:234-242)
    if elitist:
        race_instances, effective_t_new = _elitist_init_instances(
            next_instance=next_instance,
            n_total=n_total,
            elitist_new_instances=elitist_new_instances,
            sample_instances=sample_instances,
            deterministic=deterministic,
            rng=rng,
        )
    else:
        race_instances = _no_elitist_init_instances(
            next_instance=next_instance,
            n_total=n_total,
            deterministic=deterministic,
            rng=rng,
        )
        effective_t_new = 0

    # Seed is_elite_credit from carried-over data.
    for c in configurations:
        c.alive = True
        c.is_elite_credit = len(c.costs_by_inst) if elitist else 0
        c.is_elite = c.is_elite_credit > 0

    # Carried-over timeouts disqualify a config from the start of every race:
    # an elite that timed out on any instance in a prior race (its
    # `timed_out_insts` survived) loses protection and is eliminated before the
    # race begins. Runs AFTER the seeding loop so it is not re-enabled above.
    if elitist:
        _reject_timed_out(configurations)

    n_prior_instances = next_instance - 1
    elite_safe = effective_t_new + n_prior_instances if elitist else 0

    state = RaceState(
        configs=configurations,
        instances=[instances_log[i - 1] for i in race_instances],
        instances_log=list(instances_log),
        first_test=first_test,
        each_test=each_test,
        test_type=test_type,
        posthoc_test_type=posthoc_test_type,
        metric=metric,
        elitist=elitist,
        elitist_new_instances=effective_t_new,
        elitist_limit=elitist_limit,
        elite_safe=elite_safe,
        sample_instances=sample_instances,
        deterministic=deterministic,
        next_instance=next_instance,
    )

    no_elimination = 0
    experiments_used = 0
    break_reason: Optional[str] = None
    seen_inst_indices: list[int] = []
    start_time = time.time()

    # Debug data collection for the refined per-instance table.
    # debug_inst_data: inst_idx -> {config_id: (cost, rank, was_protected, was_skipped)}
    debug_inst_data: dict[int, dict[str, tuple[float, float, bool, bool]]] = {}
    debug_elim_at: dict[str, int] = {}

    # Buffer the irace-style table so it always prints as one contiguous
    # block; per-candidate debug detail is collected separately and emitted
    # below the table instead of interleaved into it.
    table_lines: list[str] = []
    # Structured per-task table rows (Best / Mean best / Exp so far), one per
    # printed table line — returned so callers can log the full race
    # trajectory, including when the Best config changes mid-race.
    race_rows: list[dict] = []
    step_trace: list[dict] = []          # manuscript per-instance-step trace
    if verbose >= 1:
        # irace-style race table header (R race.R:844, 322-323). The marker
        # legend is NOT printed here — call print_markers_header() once at the
        # start of the run instead.
        table_lines.append(_table_hline())
        table_lines.append(_table_row(list(_TABLE_COLS)))
        table_lines.append(_table_hline())

    for current_task, inst_idx in enumerate(race_instances, start=1):
        instance = instances_log[inst_idx - 1]
        alive = [c for c in configurations if c.alive]
        if not alive:
            break_reason = "no alive configs"
            break

        # Determine which alive configs need to execute on this instance.
        if (
            elitist
            and current_task <= elite_safe
            and any(c.is_elite_credit > 0 for c in alive)
        ):
            which_exe = [c for c in alive if not c.has(inst_idx)]
        else:
            which_exe = list(alive)

        # Budget guard.
        if experiments_used + len(which_exe) > max_experiments and len(which_exe) > 0:
            break_reason = f"experiments_used + {len(which_exe)} would exceed max_experiments={max_experiments}"
            break

        protected_ids = {c.id for c in alive if c.is_elite_credit > 0}
        if which_exe:
            eval_pool = _eval_one_instance_elitist(
                state,
                inst_idx,
                instance,
                which_exe,
                target_runner,
                target_evaluator,
                eval_pool=eval_pool,
                pool_runner=pool_runner,
                eval_timeout=eval_timeout,
                timeout_cost=timeout_cost,
                crash_penalty=crash_penalty,
                pool_recreate=pool_recreate,
                task_step=current_task,
            )
            experiments_used += len(which_exe)
            which_exe_ids = {c.id for c in which_exe}
            _update_is_elite_credit(alive, which_exe_ids)
        else:
            which_exe_ids = set()
            _update_is_elite_credit(alive, set())

        seen_inst_indices.append(inst_idx)
        state.evaluated_instances = current_task

        protection_active = elitist and any(c.is_elite_credit > 0 for c in alive)
        should_test = (
            current_task >= first_test
            and current_task % each_test == 0
            and len([c for c in configurations if c.alive]) > 1
        )

        test_ran = False
        _step_p = None
        prev_nb_alive = sum(1 for c in configurations if c.alive)
        nb_alive = prev_nb_alive
        if should_test:
            alive_now = [c for c in configurations if c.alive]
            test_inst_indices = [
                k for k in seen_inst_indices if all(c.has(k) for c in alive_now)
            ]
            prev_nb_alive = len(alive_now)
            test_ran, _, _step_p = _run_elitist_test(state, alive_now, test_inst_indices, alpha)
            nb_alive = sum(1 for c in configurations if c.alive)
            if (
                (elitist or early_stopping_non_elitist)
                and test_ran
                and not protection_active
            ):
                if nb_alive == prev_nb_alive:
                    no_elimination += 1
                else:
                    no_elimination = 0

        for c in configurations:
            if not c.alive and c.id not in debug_elim_at:
                debug_elim_at[c.id] = inst_idx

        # Manuscript per-step trace (structured; collected regardless of verbose).
        _alive_now = [c for c in configurations if c.alive]
        _bal = [k for k in seen_inst_indices if all(c.has(k) for c in _alive_now)]
        if len(_alive_now) >= 2 and len(_bal) >= 2:
            _d = np.array([[c.costs_by_inst[k] for c in _alive_now] for k in _bal])
            _w, _rho = _concordance(_d)
            _qv = _data_variance(_d)
        else:
            _w = _rho = _qv = float("nan")
        step_trace.append({
            "task_step": current_task,
            "inst_idx": inst_idx,
            "n_alive_before": prev_nb_alive,
            "n_alive_after": nb_alive,
            "eliminated_cand_ids": [c.id for c in configurations
                                    if debug_elim_at.get(c.id) == inst_idx],
            "test_ran": bool(test_ran),
            "protection_active": bool(protection_active),
            "p_value_omnibus": _step_p,
            "q_var": _qv, "kendall_w": _w, "rho": _rho,
            "cum_evaluations": experiments_used,
        })

        if verbose >= 1:
            row_str, row_data = _race_table_row(
                configurations,
                current_task,
                inst_idx,
                seen_inst_indices,
                which_exe,
                test_ran,
                prev_nb_alive,
                nb_alive,
                protection_active,
                experiments_used,
                start_time,
                metric,
            )
            table_lines.append(row_str)
            race_rows.append(row_data)
            if verbose >= 2:
                have = [c for c in configurations if c.has(inst_idx)]
                have.sort(key=lambda c: c.costs_by_inst[inst_idx])
                ranks = rank_costs([c.costs_by_inst[inst_idx] for c in have])
                id2rank = {c.id: r for c, r in zip(have, ranks)}

                inst_info = {}
                for c in configurations:
                    if c.id in id2rank:
                        inst_info[c.id] = (
                            c.costs_by_inst[inst_idx],
                            id2rank[c.id],
                            c.id in protected_ids,
                            c.id not in which_exe_ids,
                        )
                debug_inst_data[inst_idx] = inst_info

        if (
            (elitist or early_stopping_non_elitist)
            and elitist_limit > 0
            and not protection_active
            and no_elimination >= elitist_limit
        ):
            break_reason = f"tests without elimination ({no_elimination}) >= elitist_limit ({elitist_limit})"
            break

        if sum(1 for c in configurations if c.alive) <= min_survival:
            break_reason = f"alive configs ({sum(1 for c in configurations if c.alive)}) <= min_survival ({min_survival})"
            break

    if break_reason is None:
        break_reason = f"all instances ({len(race_instances)}) evaluated"

    survivors = [c for c in configurations if c.alive]
    final_scores = _config_scores(survivors, seen_inst_indices, metric)
    survivors.sort(key=lambda c: final_scores[id(c)])

    if verbose >= 1:
        table_lines.append(_table_hline())
        print("\n".join(table_lines))
        print(f"# Stopped because {break_reason}")
        if survivors:
            valid_survivors = [
                c for c in survivors
                if all(np.isfinite(v) and v < 1e5 for v in c.costs_by_inst.values())
            ]
            best = valid_survivors[0] if valid_survivors else survivors[0]
            best_lifetime = [
                v for v in best.costs_by_inst.values() if np.isfinite(v) and v < 1e5
            ]
            best_mean = (
                float(sum(best_lifetime) / len(best_lifetime)) if best_lifetime else float("inf")
            )
            print(
                f"Best-so-far configuration: {str(best.id):>11}"
                f"    mean value: {_fmt_perf(best_mean)}"
            )
            print("Description of the best-so-far configuration:")
            try:
                is_dict = isinstance(best.params, dict)
            except Exception:
                is_dict = False
            if is_dict:
                print("  ID   " + " ".join(f"{p:>10}" for p in best.params))
                print(
                    f"  {str(best.id):<5}"
                    + " ".join(f"{str(v):>10}" for v in best.params.values())
                )
            else:
                print(f"  ID {best.id}  ({getattr(best, 'name', type(best).__name__)})")
        if verbose >= 2:
            legend = "r = local rank" + (
                "; * = protected elite; ~ = skipped" if elitist else ""
            )
            print(
                f"\n# --- per-instance candidate evaluation (ordered by cost; {legend}) ---"
            )
            print(
                _render_debug_table(
                    configurations, seen_inst_indices, debug_inst_data, debug_elim_at,
                    instance_label=instance_label,
                )
            )
        print()

    if seen_inst_indices:
        next_instance_out = max(seen_inst_indices) + 1
    else:
        next_instance_out = next_instance

    # Build full sorted candidate list: survivors first (by rank-sum), then
    # eliminated sorted by mean cost over the instances they individually saw
    # (inf for configs that were never evaluated), for diversity top-up.
    seen_set = set(seen_inst_indices)

    def _mean_cost(c: Config) -> float:
        vals = [v for k, v in c.costs_by_inst.items() if k in seen_set]
        return float(sum(vals) / len(vals)) if vals else float("inf")

    eliminated = [c for c in configurations if not c.alive]
    eliminated_sorted = sorted(eliminated, key=_mean_cost)
    all_candidates_sorted = list(survivors) + eliminated_sorted

    return {
        "survivors": survivors,
        "all_candidates_sorted": all_candidates_sorted,
        "race_instances": race_instances,
        "seen_instances": list(seen_inst_indices),
        "elitist_new_instances": effective_t_new,
        "next_instance": next_instance_out,
        "break_reason": break_reason,
        "experiments_used": experiments_used,
        "cpu_seconds": float(state.cpu_seconds),
        "table_rows": race_rows,
        "step_trace": step_trace,
    }
