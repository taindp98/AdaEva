"""Thin wandb wrapper used by racing_eoh_obp and reprod_eoh_obp.

Provides a WandbLogger that either forwards to a live wandb run or silently
no-ops when wandb is disabled (--use-wandb not passed). Callers never need to
guard with `if wandb_logger is not None`.
"""

from __future__ import annotations

from typing import Any

import pathlib


def make_log_dir(
    root: pathlib.Path, label: str, dt_stamp: str, seed: int, tag: str = "default"
) -> pathlib.Path:
    log_dir = root / ".logs" / label / f"{dt_stamp}_{seed}_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


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


def mirror_stdout_to(path: "pathlib.Path") -> "pathlib.Path":
    """Mirror stdout+stderr to ``path`` (a ``terminal.txt``-style live log), the
    same way the racing runners do.  Restores the original streams and closes the
    file at interpreter exit.  Returns ``path`` so callers can print it.
    """
    import atexit
    import sys

    f = open(path, "w", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, f)
    sys.stderr = _Tee(orig_err, f)

    def _restore():
        sys.stdout, sys.stderr = orig_out, orig_err
        f.close()

    atexit.register(_restore)
    return path


class WandbLogger:
    """Active logger: wraps a live wandb run.

    Logs each trajectory row as a step keyed by `used_budget`.
    Metrics are nested under a `train/` prefix so they group in the
    W&B dashboard automatically.
    """

    def __init__(self, project: str, name: str, config: dict):
        import wandb  # import deferred so wandb is an optional dependency

        self._run = wandb.init(project=project, name=name, config=config)

    def log_trajectory_row(self, row: dict) -> None:
        import wandb

        step = int(row["used_budget"])
        metrics: dict[str, Any] = {
            "train/gap_pct": float(row.get("gap", row["score"])) * 100,
            "train/score": float(row["score"]),
            "train/n_instances": int(row["n_instances"]),
        }
        # EoH / racing rows carry gen_id; FunSearch rows carry cand_id.
        if "gen_id" in row:
            metrics["train/gen_id"] = int(row["gen_id"])
        elif "cand_id" in row:
            metrics["train/cand_id"] = int(row["cand_id"])
        wandb.log(metrics, step=step)

    def log_final_eval(self, final_eval: dict) -> None:
        import wandb

        wandb.summary["final/gap_pct"] = float(final_eval.get("gap", final_eval["mean_cost"])) * 100
        wandb.summary["final/score"] = float(final_eval["mean_cost"])
        wandb.summary["final/n_instances"] = int(final_eval["num_eval_instances"])
        wandb.summary["final/n_failures"] = int(final_eval["n_failures"])
        wandb.summary["final/cand_id"] = final_eval["cand_id"]

    def finish(self) -> None:
        import wandb

        wandb.finish()


class _NoopLogger:
    """Disabled logger: all methods are silent no-ops."""

    def log_trajectory_row(self, row: dict) -> None:
        pass

    def log_final_eval(self, final_eval: dict) -> None:
        pass

    def finish(self) -> None:
        pass


def make_wandb_logger(
    enabled: bool,
    project: str,
    name: str,
    config: dict,
) -> "WandbLogger | _NoopLogger":
    """Return a live WandbLogger, or a no-op logger when disabled OR when wandb
    cannot be initialised.

    A broken metrics logger must never abort the actual experiment — the run's real
    output is ``trajectory.json`` / ``heuristics.json``, not W&B. So when
    ``--use-wandb`` is set but wandb fails to import or ``wandb.init`` raises (e.g. a
    wandb/protobuf version skew, or offline auth), we degrade to the no-op logger
    with a loud warning instead of crashing the whole job.
    """
    if not enabled:
        return _NoopLogger()
    try:
        return WandbLogger(project=project, name=name, config=config)
    except Exception as exc:
        import sys
        print(
            f"  [wandb] DISABLED — could not initialise W&B "
            f"({type(exc).__name__}: {exc}). Continuing WITHOUT W&B logging; "
            f"trajectory.json / heuristics.json are still written normally.",
            file=sys.stderr,
            flush=True,
        )
        return _NoopLogger()
