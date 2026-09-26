#!/usr/bin/env python3
"""Evaluate BASELINE optimizers on the BBOB-24 x {5,10,20}D benchmark.

Uses the EXACT same evaluation loop and AUC computation as ``test_bbob.py``
(imported, not duplicated), so a baseline's AUC is directly comparable to the
LLaMEA-generated candidates scored by ``test_bbob.py`` — same 24 functions, 5
instances x 5 reps, budget 2000*dim, fresh-algorithm-per-run + crash guard, and the
same "Area under the EAF curve" metric (transform_fval(1e-8,1e8) -> per-function
get_aocc -> mean over 24 functions, per dimension + overall).

Select which baseline to run with ``--baseline NAME``. To add a baseline later:
drop a .py file that defines ONE top-level algorithm class with the standard ioh
interface (``__init__(self, budget, dim=...)``, ``__call__(self, func)``), then add
an entry to ``BASELINES`` below.
"""
import sys
import json
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# Import the SHARED machinery from test_bbob (same folder) so scoring is identical.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_bbob import get_class_name, run_evaluation, compute_auc

# name -> source file. Each file defines ONE top-level algorithm class with the ioh
# interface: __init__(self, budget, dim=...) and __call__(self, func).
BASELINES = {
    "ERADS": ROOT / ".logs" / "ioh_bbob_outputs" / "baseline_ERADS"
                  / "ERADS_QuantumFluxUltraRefined.py",
    # "CMA": ROOT / "src" / "analyses" / "baselines" / "cma.py",   # add more here
}


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a named baseline optimizer on BBOB (same protocol as test_bbob.py).")
    p.add_argument("--baseline", type=str, required=True, choices=sorted(BASELINES),
                   help="Which registered baseline to evaluate (see BASELINES).")
    p.add_argument("--n-proc", type=int, default=12, help="Number of worker processes.")
    args = p.parse_args()

    src_path = pathlib.Path(BASELINES[args.baseline])
    if not src_path.exists():
        raise FileNotFoundError(f"baseline '{args.baseline}' source not found: {src_path}")
    source = src_path.read_text()
    alg_name = get_class_name(source)
    print(f"Baseline: {args.baseline}  (class {alg_name})  from {src_path}")

    output_dir = ROOT / ".logs" / "ioh_bbob_outputs" / f"baseline_{args.baseline}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir.resolve()}")
    with open(output_dir / "terminal.txt", "w") as f:
        f.write(f"Baseline: {args.baseline}\nAlgorithm: {alg_name}\nSource: {src_path}\n")

    # Shared eval + score (identical to test_bbob.py).
    run_evaluation(source, alg_name, output_dir, args.n_proc)
    per_dim, overall = compute_auc(output_dir, alg_name)

    result_row = {
        "ID": alg_name,
        "baseline": args.baseline,
        "AUC": (round(overall, 4) if overall is not None else None),
        "AUC_per_dim": {str(k): (round(v, 4) if v is not None else None)
                        for k, v in per_dim.items()},
        "source": str(src_path),
    }
    with open(output_dir / "auc.json", "w") as f:
        json.dump(result_row, f, indent=2)
    print(f"AUC row -> {output_dir / 'auc.json'}")
    print(f"Raw ioh data saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
