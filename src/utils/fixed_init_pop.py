"""Loading and validation for paired, fixed initial heuristic populations."""

from __future__ import annotations

import json
from pathlib import Path


def load_fixed_initial_population(path: str | Path, expected_size: int) -> list[dict]:
    """Load exactly ``expected_size`` source-bearing heuristic records.

    Fixed populations are deliberately re-evaluated in the current run: their
    archived scores may have been produced with another instance pool or budget.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load fixed initial population {path}: {exc}") from exc

    heuristics = payload.get("heuristics") if isinstance(payload, dict) else None
    if not isinstance(heuristics, list):
        raise ValueError(f"Fixed initial population {path} must contain a 'heuristics' list.")
    if len(heuristics) != expected_size:
        raise ValueError(
            f"Fixed initial population {path} has {len(heuristics)} heuristics, "
            f"but --pop-size requires exactly {expected_size}."
        )
    if any(not isinstance(item, dict) or not isinstance(item.get("source"), str)
           or not item["source"].strip() for item in heuristics):
        raise ValueError(f"Every heuristic in {path} must have a non-empty string 'source'.")
    return heuristics


def install_fixed_initial_population(method, path: str | Path) -> None:
    """Make an LLM4AD ``EoH`` *method instance* seed generation 0 from the fixed
    heuristics in ``path`` (re-evaluated in this run) instead of sampling them from
    the LLM. Monkey-patches only this instance's init hook — the evolutionary phase
    is unchanged — so all allocation methods can share one initial population.

    ``--pop-size`` must equal the file's heuristic count (the loader enforces this).
    Shared by the three EoH reprod runners (OBP / TSP-GLS / FSSP-GLS)."""
    import threading

    from llm4ad.base import TextFunctionProgramConverter
    from llm4ad.method.eoh.profiler import EoHProfiler

    pop_size = int(getattr(method, "_pop_size"))
    heuristics = load_fixed_initial_population(path, pop_size)
    sources = [h["source"] for h in heuristics]
    print(f"  [fixed-init-pop] seeding generation 0 with {len(sources)} heuristics "
          f"from {path} (re-evaluated this run, not sampled from the LLM)", flush=True)

    queue = list(sources)
    lock = threading.Lock()

    # EoH's Population.register_function DEDUPS by ``str(func)`` OR ``func.score``. The
    # score-dedup collapses fixed heuristics that tie on the objective — very common
    # here (e.g. several fail this run's eval and all score -inf, or two valid ones
    # land on the same integer makespan). When collapsed, ``_next_gen_pop`` never
    # reaches pop_size, the generation counter stays at 0, and EoH fills the remainder
    # with LLM offspring that then get recorded as generation 0 — polluting the fixed
    # P0. Bypass the dedup ONLY for the injected heuristics (marked ``_fixed_init``) so
    # all pop_size distinct-code heuristics occupy generation 0; LLM offspring in later
    # generations still dedup normally.
    _pop = method._population
    _orig_has_dup = _pop.has_duplicate_function

    def _has_dup_skip_fixed(func):
        if getattr(func, "_fixed_init", False):
            return False
        return _orig_has_dup(func)

    _pop.has_duplicate_function = _has_dup_skip_fixed

    def _register_fixed(source: str) -> None:
        # Mirror EoH._sample_evaluate_register, minus the LLM call: compile the fixed
        # source, evaluate it on THIS run's instances, and register it like a sample.
        func = TextFunctionProgramConverter.text_to_function(source)
        if func is None:
            return
        program = TextFunctionProgramConverter.function_to_program(func, method._template_program)
        if program is None:
            return
        score, eval_time = method._evaluation_executor.submit(
            method._evaluator.evaluate_program_record_time, program).result()
        # Coerce a FAILED eval to a large finite penalty (-1e6), NOT None/-inf/nan.
        # "Failed" means score is None OR non-finite (-inf/+inf/nan) — the two distinct
        # ways a fixed heuristic can fail: the eval wrapper returns None (subprocess
        # timeout/crash), or the eval completes but returns -inf (e.g. FSSP/TSP GLS scores
        # -mean(makespan) and an INVALID makespan makes that -inf). Both must become a
        # finite penalty, for two reasons:
        #   (1) EoH's Population.register_function DROPS ``score is None`` funcs at
        #       generation 0, which would shrink the injected population below pop_size,
        #       stall the generation counter, and let LLM offspring leak into generation 0.
        #   (2) Population.selection() filters parents with ``not math.isinf(f.score)``, so
        #       an -inf heuristic (even though it is KEPT in gen-0) is EXCLUDED from being a
        #       parent; if too few finite heuristics survive (e.g. FSSP often has 3-4/10
        #       finite), the run aborts at gen-1 via the ``len(population) < selection_num``
        #       guard in eoh.py.
        # A finite penalty keeps every fixed heuristic a viable (lowest-ranked) parent so
        # the fixed P0 stays exactly pop_size AND gen-1 has enough parents. -1e6 matches the
        # failure sentinel already used in the init_pop JSON files and sits far below any
        # real evaluated score. (isfinite via ``x == x and -inf < x < inf`` — no math import.)
        FAILED_PENALTY = -1e6
        _finite = (score is not None and score == score
                   and float("-inf") < float(score) < float("inf"))
        func.score = float(score) if _finite else FAILED_PENALTY
        func.evaluate_time = eval_time
        func._fixed_init = True   # exempt from register_function's score/code dedup
        if not getattr(func, "algorithm", None):
            func.algorithm = "fixed-init"
        func.sample_time = 0.0
        if method._profiler is not None:
            method._profiler.register_function(func, program=str(program))
            if isinstance(method._profiler, EoHProfiler):
                method._profiler.register_population(method._population)
            method._tot_sample_nums += 1
        method._population.register_function(func)

    def _init_from_fixed() -> None:
        # Runs in each of EoH's sampler threads; the lock hands out each fixed source
        # exactly once (mirrors the LLM init loop, which the run() call replaces).
        while True:
            with lock:
                if not queue:
                    return
                src = queue.pop(0)
            try:
                _register_fixed(src)
            except Exception as exc:  # never abort init on one bad heuristic
                print(f"  [fixed-init-pop] WARN: failed to register a heuristic: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    # run() calls ``self._iteratively_init_population`` (looked up at call time), so an
    # instance attribute cleanly overrides the LLM initialisation for this run only.
    method._iteratively_init_population = _init_from_fixed
