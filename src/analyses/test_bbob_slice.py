"""Evaluate SLICED candidate checkpoints of an --exp run on the BBOB benchmark.

Combines the two sibling analyses:

  * from ``test_tsplib_slice.py`` — load candidates at budget proportions
    ``[0.2, 0.4, 0.6, 0.8, 1.0]`` of the final used_budget (trajectory OR
    incumbents), evaluate each, and save the per-candidate result list to a
    single ``*_slice_s{seed}.json`` (written incrementally after each candidate).

  * from ``test_bbob.py`` — score each candidate on BBOB by STRICTLY following its
    ``eval_fid`` protocol (by default 24 functions x {5,10,20}D x 5 instances x 5 reps,
    ioh logging), then the LLaMEA-benchmark AUC via iohinspector. ``run_evaluation``
    and ``get_class_name`` are IMPORTED (not copied) so the evaluation is byte-for
    byte identical to ``test_bbob.py`` at the default settings. The tested functions
    (``--n-test-functions``) and per-function instance ids (``--inst-indices``) are
    configurable and forwarded to ``run_evaluation``; AUC scoring auto-adapts to
    whatever ioh data those settings produce.

Per candidate we save the AUC (overall), AUC_per_dim, AUC_per_function, AND the
full RAW per-run AOCC list (per function x instance x run) for later deep analysis.

Output -> ``bbob_slice_s{seed}.json`` in the experiment directory.
"""

import sys
import json
import time
import shutil
import argparse
import multiprocessing
import pathlib
from concurrent.futures import TimeoutError as _CFTimeout

import numpy as np
import iohinspector
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse the EXACT evaluation protocol from test_bbob.py (strictly follow eval_fid):
# run_evaluation() pools eval_fid over the 24 functions with the same
# initializer/globals, so slices are scored identically to test_bbob.py.
# We ALSO import the pool internals (_worker_init / eval_fid) so
# run_evaluation_timeout() below can enforce a per-fid wall-clock cap (mirroring
# src/racing/llamea_bbob.py) without touching test_bbob.py.
from analyses.test_bbob import (
    get_class_name, run_evaluation, _worker_init, eval_fid,
)

_DIMS = (5, 10, 20)   # dims eval_fid sweeps per fid (must match test_bbob.eval_fid)
_N_REPS = 5           # reps per (fid, dim, instance) in eval_fid


def _auto_eval_timeout(budget_factor: int, dim: int) -> float:
    """Per-(candidate, instance) wall-clock cap that scales with the func-eval
    budget (= ``budget_factor * dim``); mirrors ``reprod.llamea_bbob._auto_eval_timeout``
    (inlined here to avoid importing the LLaMEA dependency chain). Examples
    (budget_factor=2000): dim 5 -> 160s, dim 10 -> 260s, dim 20 -> 460s."""
    return 10

_SLICES = [0.2, 0.4, 0.6, 0.8, 1.0]
_AUC_LOWER, _AUC_UPPER = 1e-8, 1e8          # LLaMEA-benchmark bounds (== test_bbob.compute_auc)


def _load_slice_entries(exp_path, trajectory_path, has_incumbent, incumbent_mode):
    """Return the ``[{'cand_id','used_budget'}, ...]`` list to slice: the raw
    ``trajectory`` list (default) or, with ``has_incumbent``, the aligned
    ``cand_id``/``used_budget`` series from ``valid_trajectory_<incumbent_mode>.json``
    (``incumbent_mode`` in {``mean_rank``, ``mean_cost``})."""
    if has_incumbent:
        vt_path = exp_path / f"valid_trajectory_{incumbent_mode}.json"
        if not vt_path.exists():
            raise FileNotFoundError(
                f"Missing: {vt_path}; run the racing validation (_final_eval) to emit it.")
        with open(vt_path) as f:
            vt = json.load(f)
        cids = vt.get("cand_id")
        ubs = vt.get("used_budget")
        if not isinstance(cids, list) or not cids:
            raise KeyError(f"{vt_path} has no non-empty 'cand_id' list")
        if not isinstance(ubs, list) or len(ubs) != len(cids):
            raise KeyError(f"{vt_path} has no 'used_budget' list aligned to 'cand_id'")
        return [{"cand_id": c, "used_budget": u} for c, u in zip(cids, ubs)]
    with open(trajectory_path) as f:
        traj_data = json.load(f)
    lst = traj_data.get("trajectory")
    if not isinstance(lst, list) or not lst:
        raise KeyError(
            f"trajectory.json at {trajectory_path} has no non-empty 'trajectory' list.")
    return lst


def load_gen_heuristics(exp_path: pathlib.Path, has_incumbent: bool = False,
                        incumbent_mode: str = "mean_rank") -> list:
    """Return ``[{'cand_id','source','used_budget','slice'}, ...]`` for the sliced
    budget checkpoints ``_SLICES`` (mirrors ``test_tsplib_slice.load_gen_heuristics``).

    The candidate for each slice is the trajectory/incumbent entry whose
    ``used_budget`` is closest to ``slice * final_budget``; its source is resolved
    from ``heuristics.json``."""
    trajectory_path = exp_path / "trajectory.json"
    heuristic_path = exp_path / "heuristics.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing: {trajectory_path}")
    if not heuristic_path.exists():
        raise FileNotFoundError(f"Missing: {heuristic_path}")

    lst = _load_slice_entries(exp_path, trajectory_path, has_incumbent, incumbent_mode)

    final_budget = lst[-1]["used_budget"]
    target_budgets = [s * final_budget for s in _SLICES]

    with open(heuristic_path) as f:
        heuristics_data = json.load(f)
    src_by_id = {h["cand_id"]: h["source"] for h in heuristics_data["heuristics"]}

    selected = []
    for s, tb in zip(_SLICES, target_budgets):
        closest = min(lst, key=lambda entry: abs(entry["used_budget"] - tb))
        cid = closest["cand_id"]
        source = src_by_id.get(cid)
        if source is None:
            continue
        selected.append({
            "cand_id": cid,
            "source": source,
            "used_budget": closest["used_budget"],
            "slice": s,
        })
    return selected


def _per_fid_timeout(inst_indices, per_unit_timeout: float) -> float:
    """Per-fid wall-clock cap = sum over the dims eval_fid sweeps of the per-(config,
    instance) auto cap, times the instances x reps eval_fid runs per dim.

    ``eval_fid(fid)`` runs {5,10,20}D x len(inst_indices) instances x 5 reps, so a single
    fid task legitimately takes far longer than ONE (config, instance) eval. We size the
    cap from _auto_eval_timeout (which already scales with dim/budget) summed across dims
    and multiplied by (#instances x #reps), so a healthy heavy heuristic never false-times
    out while an infinite loop is still cut."""
    n_units = max(1, len(list(inst_indices))) * _N_REPS
    if per_unit_timeout is not None and per_unit_timeout > 0:
        per_dim = per_unit_timeout                       # fixed per-(config,instance) cap
        return float(per_dim) * len(_DIMS) * n_units
    # auto: _auto_eval_timeout scales per dim (budget_factor fixed at 2000, as in eval_fid).
    return float(sum(_auto_eval_timeout(2000, d) for d in _DIMS)) * n_units


def run_evaluation_timeout(source, alg_name, out_dir, n_proc, n_functions, inst_indices,
                           eval_timeout):
    """Hang-safe drop-in for ``test_bbob.run_evaluation``: pools ``eval_fid`` over the fids
    with a PER-FID wall-clock cap, mirroring the per-eval timeout in
    ``src/racing/llamea_bbob.py`` (Future.result-style ``.get(timeout=...)`` + terminate/
    rebuild the pool on a hang). A fid whose heuristic infinite-loops is penalised (skipped,
    with a WARN) and the pool rebuilt so the run continues; its partial ioh logs (whatever
    completed before the cap) are still scored by ``compute_auc_slice``.

    ``eval_timeout``: >0 = fixed cap per (config, instance) unit; <0 = auto-scale via
    _auto_eval_timeout; 0 or None = DISABLED (falls back to the stock, uncapped
    run_evaluation so behaviour is byte-identical to test_bbob.py)."""
    if eval_timeout is None or eval_timeout == 0:
        run_evaluation(source, alg_name, out_dir, n_proc,
                       n_functions=n_functions, inst_indices=inst_indices)
        return

    inst_indices = list(inst_indices)
    per_fid = _per_fid_timeout(inst_indices, eval_timeout)
    fids = list(range(1, int(n_functions) + 1))
    print(f"\nStarting evaluation in parallel across {n_functions} functions "
          f"(instances {inst_indices}); per-fid timeout = {per_fid:.0f}s "
          f"(eval_timeout={'auto' if eval_timeout < 0 else eval_timeout})...", flush=True)
    start_time = time.time()

    def _new_pool():
        return multiprocessing.Pool(
            processes=n_proc, initializer=_worker_init,
            initargs=(source, str(out_dir), alg_name, inst_indices))

    def _kill_pool(pool):
        """terminate() + join the workers so a process stuck in an infinite-loop
        heuristic is actually killed (mirrors racing's _shutdown_pool)."""
        try:
            pool.terminate()
        except Exception:
            pass
        try:
            pool.join()
        except Exception:
            pass

    pool = _new_pool()
    try:
        for fid in fids:
            # One fid at a time so a hung fid only wastes its own worker; result() with a
            # per-fid cap lets us reclaim + rebuild the pool on a hang.
            ar = pool.apply_async(eval_fid, (fid,))
            try:
                done_fid = ar.get(timeout=per_fid)
                print(f"Function {done_fid:<2} completed successfully.", flush=True)
            except (multiprocessing.TimeoutError, _CFTimeout):
                print(f"    [WARN] fid={fid} exceeded per-fid timeout {per_fid:.0f}s "
                      f"(heuristic likely infinite-loops) -> skipped; rebuilding pool.",
                      flush=True)
                _kill_pool(pool)          # hard-kill the hung worker(s)
                pool = _new_pool()        # fresh pool for the remaining fids
            except Exception as e:
                # A crash inside eval_fid still propagates here (eval_fid only guards the
                # inner alg(problem) call). Log and move on, matching the crash-guard intent.
                print(f"    [WARN] fid={fid} raised {type(e).__name__}: {e} -> skipped.",
                      flush=True)
    finally:
        _kill_pool(pool)
    print(f"Evaluation finished in {time.time() - start_time:.1f}s.", flush=True)


def compute_auc_slice(output_dir: pathlib.Path, alg_name: str):
    """Score the ioh data under ``output_dir`` exactly as ``test_bbob.compute_auc``
    (LLaMEA-benchmark AOCC, lower=1e-8 upper=1e8, per-dim then averaged), but ALSO
    return the per-function breakdown AND the full RAW per-run score list for later
    deep analysis.

    Returns ``(per_dim, overall, per_function, raw_scores)``:
      * ``per_dim[dim]``      -> mean AUC over the tested functions (or None).
      * ``overall``           -> mean of ``per_dim`` over the dims with data.
      * ``per_function[dim]`` -> ``{function_name: AUC}`` (mean over its runs).
      * ``raw_scores[dim]``   -> the FULL list of individual per-run AOCCs
        ``[{"function_name","instance","run_id","AOCC"}, ...]`` (n_functions x
        len(inst_indices) x 5 reps entries/dim); ``None`` when a dim has no data.
        The aggregations above are exact reductions of these raw scores.

      All aggregations are computed over whatever (function, instance) data is present
      in ``output_dir``, so they auto-adapt to the configured --n-test-functions /
      --inst-indices (no fixed 24x5 assumption)."""
    # Finest granularity get_aocc yields cleanly: one AOCC per (function, instance,
    # run). data_id (the per-run id) is already the internal group key, so it must
    # NOT be a free_var; function+instance+run_id uniquely separate every run.
    RAW_FV = ["function_name", "instance", "run_id", "algorithm_name"]

    manager = iohinspector.DataManager()
    manager.add_folder(str(output_dir))
    df = manager.load(True, True)   # polars DataFrame
    have_raw = all(c in df.columns for c in RAW_FV)

    per_dim: dict = {}
    per_function: dict = {}
    raw_scores: dict = {}
    for dim in [5, 10, 20]:
        df_dim = df.filter(pl.col("dimension") == dim)
        if df_dim.height == 0:
            per_dim[dim] = None
            per_function[dim] = None
            raw_scores[dim] = None
            continue
        budget = 2000 * dim
        df_eaf = iohinspector.metrics.transform_fval(df_dim, _AUC_LOWER, _AUC_UPPER)

        # Aggregations — IDENTICAL to test_bbob.compute_auc.
        per_fn = iohinspector.metrics.get_aocc(
            df_eaf, budget, free_vars=["function_name", "algorithm_name"])
        aucs = np.asarray(per_fn["AOCC"], dtype=float)
        per_dim[dim] = float(aucs.mean())
        fn_names = per_fn["function_name"].to_list()
        per_function[dim] = {str(fn): float(a) for fn, a in zip(fn_names, aucs)}

        # Raw per-run scores (kept in full, no rounding).
        if have_raw:
            raw = iohinspector.metrics.get_aocc(df_eaf, budget, free_vars=RAW_FV)
            raw_scores[dim] = [
                {"function_name": str(r["function_name"]),
                 "instance": int(r["instance"]),
                 "run_id": int(r["run_id"]),
                 "AOCC": float(r["AOCC"])}
                for r in raw.to_dict(orient="records")
            ]
        else:
            raw_scores[dim] = None

        n_raw = "n/a" if raw_scores[dim] is None else len(raw_scores[dim])
        print(f"    --- {dim}D (budget {budget}): AUC = {per_dim[dim]:.4f}  "
              f"(mean over {len(per_fn)} functions; {n_raw} raw per-run scores) ---", flush=True)

    valid = [v for v in per_dim.values() if v is not None]
    overall = float(np.mean(valid)) if valid else None
    return per_dim, overall, per_function, raw_scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an --exp run's SLICED candidate checkpoints on BBOB "
                    "(test_bbob's eval_fid protocol), saving per-candidate AUC / "
                    "AUC_per_dim / AUC_per_function.")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment directory with trajectory.json + heuristics.json.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Run/output index (e.g. a SLURM array task id). Names the "
                             "output file bbob_slice_s{seed}.json and the per-candidate ioh "
                             "folders; the BBOB eval itself follows test_bbob.eval_fid "
                             "(5 instances x 5 reps per function), so it is already a robust "
                             "average and is NOT reseeded here.")
    parser.add_argument("--has-incumbent", action="store_true",
                        help="Slice the INCUMBENT entries instead of raw trajectory entries.")
    parser.add_argument("--incumbent-mode", type=str, default="mean_rank",
                        choices=["mean_rank", "mean_cost"],
                        help="With --has-incumbent, which validated incumbent series to slice: "
                             "'mean_rank' -> valid_trajectory_mean_rank.json, "
                             "'mean_cost' -> valid_trajectory_mean_cost.json.")
    parser.add_argument("--n-proc", type=int, default=12,
                        help="Parallel workers for the per-candidate BBOB eval (one per function).")
    parser.add_argument("--n-test-functions", type=int, default=24,
                        help="Number of BBOB functions to test (fids 1..N; default 24 = full suite).")
    parser.add_argument("--inst-indices", type=int, nargs="+", default=[4, 5, 6, 7, 8],
                        help="Instance ids to score per function (default 4 5 6 7 8). "
                             "E.g. --inst-indices 4 5 6 7 8.")
    parser.add_argument("--keep-ioh", action="store_true",
                        help="Keep the raw per-candidate ioh output folders (default: delete "
                             "after AUC is computed to save disk).")
    parser.add_argument("--eval-timeout", type=float, default=-1.0,
                        help="Per-(config, instance) wall-clock cap (s) guarding against "
                             "infinite-loop heuristics; enforced as a per-FID cap (scaled by "
                             "the {5,10,20} dims x instances x 5 reps eval_fid runs) with a "
                             "pool terminate/rebuild on a hang, mirroring src/racing/"
                             "llamea_bbob.py. -1 = auto-scale with budget/dim (via "
                             "_auto_eval_timeout); 0 = disabled (uncapped, byte-identical to "
                             "test_bbob.py). Default -1.")
    args = parser.parse_args()

    exp_path = pathlib.Path(args.exp) if pathlib.Path(args.exp).is_absolute() else ROOT / args.exp
    candidates = load_gen_heuristics(exp_path, args.has_incumbent, args.incumbent_mode)
    tag = "incumbent" if args.has_incumbent else "gen"

    print(f"Loaded {len(candidates)} candidate checkpoints from {exp_path} (source={tag})")
    print(f"BBOB test grid: {args.n_test_functions} functions x {{5,10,20}}D x "
          f"instances {args.inst_indices} x 5 reps")
    for c in candidates:
        print(f"  cand_id={c['cand_id']:<15s} budget={c['used_budget']:<8} slice={c['slice']:<4.1f}")

    results_file = exp_path / f"bbob_slice_s{args.seed}.json"
    # Raw ioh logs live UNDER the experiment folder (temporary; deleted after
    # scoring unless --keep-ioh), so the experiment dir holds everything — just
    # like test_tsplib_slice keeps its output there.
    ioh_root = exp_path / f"bbob_slice_ioh_s{args.seed}"
    output_json = {
        "exp": str(exp_path),
        "source": tag,
        "seed": args.seed,
        "auc_bounds": [_AUC_LOWER, _AUC_UPPER],
        "n_test_functions": args.n_test_functions,
        "inst_indices": list(args.inst_indices),
        "candidates": [],
    }

    for idx, cand in enumerate(candidates, 1):
        cand_id = cand["cand_id"]
        source = cand["source"]
        used_budget = cand["used_budget"]
        slice_val = cand["slice"]
        alg_name = get_class_name(source)

        print()
        print(f"[{idx}/{len(candidates)}] Evaluating {cand_id} (class={alg_name}, "
              f"used_budget={used_budget}, slice={slice_val}) on BBOB ...", flush=True)

        # Each candidate gets an ISOLATED ioh output dir (under the experiment
        # folder) so their logs (and the AUC computed from them) never mix; wipe
        # any stale data from a previous run.
        out_dir = ioh_root / f"slice{slice_val}_{cand_id}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_dim, overall, per_function, raw_scores = {}, None, {}, {}
        try:
            run_evaluation_timeout(source, alg_name, out_dir, args.n_proc,
                                   n_functions=args.n_test_functions,
                                   inst_indices=args.inst_indices,
                                   eval_timeout=args.eval_timeout)
            per_dim, overall, per_function, raw_scores = compute_auc_slice(out_dir, alg_name)
        except Exception as e:
            print(f"  [WARN] evaluation/scoring failed for {cand_id}: "
                  f"{type(e).__name__}: {e}", flush=True)
        finally:
            if not args.keep_ioh and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

        cand_entry = {
            "cand_id": cand_id,
            "alg_name": alg_name,
            "used_budget": used_budget,
            "slice": slice_val,
            "AUC": overall,
            "AUC_per_dim": {str(k): v for k, v in per_dim.items()},
            "AUC_per_function": {str(k): v for k, v in per_function.items()},
            "raw_scores": {str(k): v for k, v in raw_scores.items()},
        }
        output_json["candidates"].append(cand_entry)

        # Incremental save so a long run is resumable/inspectable mid-flight.
        with open(results_file, "w") as f:
            json.dump(output_json, f, indent=4)
        auc_str = f"{overall:.4f}" if overall is not None else "n/a"
        print(f"  -> slice {slice_val} cand={cand_id} AUC={auc_str}", flush=True)

    # Remove the (now-empty) ioh parent folder unless the raw logs were kept.
    if not args.keep_ioh and ioh_root.exists():
        shutil.rmtree(ioh_root, ignore_errors=True)

    print()
    print(f"Seed {args.seed} finished across {len(output_json['candidates'])} candidate checkpoints!")
    for c in output_json["candidates"]:
        auc_str = f"{c['AUC']:.4f}" if c["AUC"] is not None else "n/a"
        print(f"  slice {c['slice']:<4.1f} (budget {c['used_budget']:<8}) "
              f"cand={c['cand_id']:<15s} AUC={auc_str}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
