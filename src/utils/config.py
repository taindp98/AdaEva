from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class Config:
    """A single candidate configuration participating in a race.

    R reference:
        - `is_elite` integer vector: `packages/irace/R/race.R:756`
          (allocation), `:593` (`update_is_elite`).
        - `Results` matrix (per-(instance, config) cost storage):
          `packages/irace/R/race.R:743-758`. Our `costs_by_inst` dict is the
          per-row slice of that matrix for one configuration.
        - `.ID.` column (stable ID across iterations):
          `packages/irace/R/configurations.R` (uses `.ID.` throughout).

    Storage model:
        - `costs` / `ranks` are flat lists kept in evaluation order (legacy
          API used by `run_race` and external EoH drivers).
        - `costs_by_inst` / `ranks_by_inst` are instance-indexed dicts that
          mirror R irace's `Results` matrix. They are the primary storage
          consulted by the elitist race (which needs O(1) per-instance lookup
          across races) and by `mean_cost` / `sum_ranks` when present.
        - `is_elite_credit` mirrors R's `is_elite[i]`: the number of past
          instances this elite has already been evaluated on, decremented
          every task in which it is skipped because it already has a result.

    Arguments:
        id: Stable identifier (e.g. "c000"). Used for logging/output only.
        params: Parameter dictionary handed to `target_runner`.
        costs: Costs observed so far (one per evaluation), oldest first.
        ranks: Per-instance ranks observed so far, oldest first.
        costs_by_inst: Map instance_idx (1-based) -> cost. Used by the
            elitist race; optional for the legacy path.
        ranks_by_inst: Map instance_idx (1-based) -> rank.
        n_evals: Number of instances on which this config has been evaluated.
        alive: False once the config has been eliminated by a statistical test.
        is_elite_credit: Remaining "protection window" for this elite. Configs
            sampled fresh inside a race have 0; elites carried over from a
            previous race start with `len(costs_by_inst)`.
        is_elite: Whether this config entered the race as an elite carried
            over from a previous race.
    """

    id: str
    params: dict
    costs: list[float] = field(default_factory=list)
    ranks: list[float] = field(default_factory=list)
    costs_by_inst: dict[int, float] = field(default_factory=dict)
    ranks_by_inst: dict[int, float] = field(default_factory=dict)
    n_evals: int = 0
    alive: bool = True
    is_elite_credit: int = 0
    is_elite: bool = False
    # Instances on which this config was killed by the per-eval wall-clock cap
    # (a timeout, as opposed to a crash). Recorded explicitly so the race can
    # strip elite protection from a timed-out config regardless of the
    # configured penalty value. Persists across races (carried with the elite)
    # until reset_history().
    timed_out_insts: set = field(default_factory=set)
    # Per-instance eval metadata for the manuscript instance_seed_perf log:
    # inst_idx -> {"wall_s": float, "race_step": int}. Populated at fresh-eval time
    # by the race engine; parallels costs_by_inst and is cleared with it.
    meta_by_inst: dict = field(default_factory=dict)

    @property
    def mean_cost(self) -> float:
        """Average cost over all observed evaluations.

        Prefers the instance-indexed dict when populated (elitist path), and
        falls back to the legacy list (non-elitist path). Returns +inf when
        nothing has been recorded.

        Example:
            >>> Config(id='x', params={}, costs=[1.0, 3.0]).mean_cost
            2.0
        """
        if self.costs_by_inst:
            return sum(self.costs_by_inst.values()) / len(self.costs_by_inst)
        return sum(self.costs) / len(self.costs) if self.costs else float("inf")

    @property
    def sum_ranks(self) -> float:
        """Sum of per-instance ranks; prefers dict view when populated.

        Example:
            >>> Config(id='x', params={}, ranks=[1, 2, 1]).sum_ranks
            4
        """
        if self.ranks_by_inst:
            return sum(self.ranks_by_inst.values())
        return sum(self.ranks)

    def record(self, inst_idx: int, cost: float, rank: Optional[float] = None) -> None:
        """Record a single (instance, cost[, rank]) observation in both views.

        Features:
            - Appends to the legacy `costs` / `ranks` lists in call order.
            - Writes to `costs_by_inst` / `ranks_by_inst` keyed by `inst_idx`.
            - Increments `n_evals`.
        """
        self.costs.append(float(cost))
        self.costs_by_inst[inst_idx] = float(cost)
        if rank is not None:
            self.ranks.append(float(rank))
            self.ranks_by_inst[inst_idx] = float(rank)
        self.n_evals += 1

    def reset_history(self) -> None:
        """Clear all accumulated cost/rank history so the config enters the
        next race with a clean slate.  Call before non-elitist races to prevent
        stale per-instance costs from previous generations contaminating
        mean_cost when a carried-over config is eliminated before it has been
        re-evaluated on every instance in the new race."""
        self.costs.clear()
        self.ranks.clear()
        self.costs_by_inst.clear()
        self.ranks_by_inst.clear()
        self.meta_by_inst.clear()
        self.timed_out_insts.clear()
        self.n_evals = 0

    def has(self, inst_idx: int) -> bool:
        """Has this config already been evaluated on instance `inst_idx`?"""
        return inst_idx in self.costs_by_inst

    def to_dict(self) -> dict:
        """Serialize the config to a plain dict suitable for race output."""
        return {
            "id": self.id,
            "params": self.params,
            "mean_cost": self.mean_cost,
            "sum_ranks": self.sum_ranks,
            "n_evals": self.n_evals,
            "alive": self.alive,
            "costs": list(self.costs),
            "costs_by_inst": dict(self.costs_by_inst),
            "is_elite": self.is_elite,
        }


@dataclass
class RaceState:
    """Mutable bookkeeping for an in-progress race.

    R reference:
        - R6 class `RaceState` definition: `packages/irace/R/race_state.R:1-50`.
        - `next_instance` field: `packages/irace/R/race_state.R:9, 248`.
        - `instances_log`: `packages/irace/R/race_state.R` (the
          `instances_log` data.table — we collapse it to a flat instance list).
        - `elitist_new_instances` (= `T^new * blockSize`):
          `packages/irace/R/race_state.R:43`.
        - `elitistLimit` (T^max): `packages/irace/R/irace-options.R` (option
          definition) and `packages/irace/R/race.R:918-922` (usage).

    Extends the original race state with the fields needed by the elitist
    race (`next_instance`, `instances_log`, `elitist*`, `elite_safe`). All
    new fields default to non-elitist values so legacy callers are unaffected.

    Arguments:
        configs: Full list of candidates; race only flips their `alive` flag.
        instances: Ordered instance stream consumed during *this* race (may be
            a permutation of `instances_log` indices when elitist is True).
        instances_log: Canonical pool of all instances available across races
            (used to reorder per-race when elitist is True). When empty, falls
            back to `instances`.
        evaluated_instances: How many instances processed in the current race.
        next_instance: 1-based index into `instances_log` of the first
            instance that is "new" in the next race. Mirrors R's
            `race_state$next_instance`.
        first_test, each_test: T^first / T^each.
        test_type, metric: same as before.
        total_evaluations: total (config, instance) evaluations consumed.
        elitist: enables the elitist race semantics.
        elitist_new_instances: T^new, number of fresh instances prepended.
        elitist_limit: T^max, max consecutive tests without elimination
            (0 disables this stop criterion).
        elite_safe: protection-window length for the current race (computed
            inside `elitist_race`).
        sample_instances: shuffle past instances each race (R's
            `scenario$sampleInstances`).
        deterministic: skip instance resampling.
    """

    configs: list
    instances: list[Any]
    instances_log: list[Any] = field(default_factory=list)
    evaluated_instances: int = 0
    next_instance: int = 1
    first_test: int = 5
    each_test: int = 1
    test_type: Literal["friedman", "ttest"] = "friedman"
    posthoc_test_type: Literal["conover", "nemenyi"] = "conover"
    metric: Literal["sum_ranks", "mean_cost"] = "sum_ranks"
    total_evaluations: int = 0
    # Cumulative CPU seconds (summed across worker processes / cores) consumed by
    # (config, instance) evaluations during this race. Populated only when the
    # runner returns a (cost, cpu_seconds) pair; stays 0.0 for scalar runners.
    cpu_seconds: float = 0.0
    elitist: bool = False
    elitist_new_instances: int = 0
    elitist_limit: int = 2
    elite_safe: int = 0
    sample_instances: bool = True
    deterministic: bool = False

    def alive_configs(self) -> list:
        """Return the subset of configs still in the race (alive=True)."""
        return [c for c in self.configs if c.alive]

    def elite_configs(self) -> list:
        """Return configs whose elite credit has not yet expired."""
        return [c for c in self.configs if c.is_elite_credit > 0]


class ConfigAS(Config):
    """A race candidate whose 'parameter' is an entire Python function.

    Extends `Config` for the *automated algorithm selection* setting: each
    candidate is a complete algorithm provided as source code (e.g. an LLM-
    generated heuristic) rather than a scalar parameter dict. The source is
    compiled lazily on first access and the resulting callable is cached.

    Features:
        - Stores the algorithm as text (`source`) plus the name of the
          callable to extract (`entry_point`). The text is the canonical
          representation; the compiled function is a derived artifact.
        - Lazy compilation via `exec` into a private namespace on first
          access to `.callable`. Subsequent calls are O(1).
        - `c.params` is overridden to *be* the compiled callable, so
          `target_runner(c.params, instance)` receives the algorithm directly.
        - `to_dict()` returns the source text and entry point (serializable)
          rather than the live function object.

    Arguments:
        id: Stable identifier (e.g. "alg000").
        source: Python source code defining the algorithm. Must contain a
            top-level function whose name matches `entry_point`.
        entry_point: Name of the function inside `source` to use as the
            callable.
        name: Optional human-readable label. Defaults to `entry_point`.
        global_ns: Optional dict of names made available to the compiled
            source (e.g. `{'np': numpy}`).
        costs: Optional pre-populated list of costs observed so far.
        ranks: Optional pre-populated list of per-instance ranks.
        costs_by_inst: Optional map `instance_idx (1-based) -> cost`.
        ranks_by_inst: Optional map `instance_idx (1-based) -> rank`.
        n_evals: Number of instances this candidate has been evaluated on.
        alive: Whether this candidate is still in the race.
        is_elite_credit: Remaining protection-window credit.
        is_elite: Marks this candidate as an elite carried over from a
            previous race.

    Example:
        >>> src = '''
        ... import numpy as np
        ... def update(x):
        ...     return float(np.sum(x ** 2))
        ... '''
        >>> c = ConfigAS(id='alg00', source=src, entry_point='update')
        >>> fn = c.callable                           # compiles once
        >>> fn is c.callable                           # cached
        True
    """

    _MODULE_TAG = "\x00configas-module\x00"

    def __init__(
        self,
        id: str,
        source: str,
        entry_point: str,
        name: Optional[str] = None,
        global_ns: Optional[dict] = None,
        costs: Optional[list[float]] = None,
        ranks: Optional[list[float]] = None,
        costs_by_inst: Optional[dict[int, float]] = None,
        ranks_by_inst: Optional[dict[int, float]] = None,
        n_evals: int = 0,
        alive: bool = True,
        is_elite_credit: int = 0,
        is_elite: bool = False,
    ):
        super().__init__(
            id=id,
            params={},
            costs=costs if costs is not None else [],
            ranks=ranks if ranks is not None else [],
            costs_by_inst=costs_by_inst if costs_by_inst is not None else {},
            ranks_by_inst=ranks_by_inst if ranks_by_inst is not None else {},
            n_evals=n_evals,
            alive=alive,
            is_elite_credit=is_elite_credit,
            is_elite=is_elite,
        )
        self.source = source
        self.entry_point = entry_point
        self.name = name or entry_point
        self.global_ns = global_ns
        self._callable = None

    @property
    def callable(self):
        """Compile (once) and return the algorithm as a Python callable.

        Raises `ValueError` if `entry_point` is not defined after executing
        the source. Raises whatever `exec` raises (SyntaxError, etc.) on
        bad source.

        Example:
            >>> c = ConfigAS(id='a', source='def f(x): return x + 1', entry_point='f')
            >>> c.callable(41)
            42
        """
        if self._callable is None:
            ns: dict[str, Any] = dict(self.global_ns) if self.global_ns else {}
            exec(self.source, ns)
            fn = ns.get(self.entry_point)
            if fn is None:
                raise ValueError(
                    f"entry point '{self.entry_point}' not found after exec; "
                    f"defined names: {sorted(k for k in ns if not k.startswith('_'))}"
                )
            if not callable(fn):
                raise ValueError(
                    f"'{self.entry_point}' is not callable: {type(fn).__name__}"
                )
            self._callable = fn
        return self._callable

    def __getstate__(self) -> dict:
        """Pickle support: ship source text, drop unpicklable derived objects.

        `_callable` is dropped (recompiled lazily in the worker). Module
        values in `global_ns` are replaced with `(_MODULE_TAG, module_name)`
        markers and re-imported in `__setstate__`.
        """
        import types

        state = self.__dict__.copy()
        state["_callable"] = None
        if self.global_ns:
            state["global_ns"] = {
                k: (
                    (ConfigAS._MODULE_TAG, v.__name__)
                    if isinstance(v, types.ModuleType)
                    else v
                )
                for k, v in self.global_ns.items()
            }
        return state

    def __setstate__(self, state: dict) -> None:
        ns = state.get("global_ns")
        if ns:
            import importlib

            state = dict(state)
            state["global_ns"] = {
                k: (
                    importlib.import_module(v[1])
                    if isinstance(v, tuple)
                    and len(v) == 2
                    and v[0] == ConfigAS._MODULE_TAG
                    else v
                )
                for k, v in ns.items()
            }
        self.__dict__.update(state)

    @property
    def params(self):
        """Override Config.params to expose the compiled callable.

        Makes `target_runner(c.params, instance)` resolve to
        `target_runner(algorithm_fn, instance)`.

        Example:
            >>> c = ConfigAS(id='a', source='def f(x): return x*2', entry_point='f')
            >>> c.params(3)
            6
        """
        return self.callable

    @params.setter
    def params(self, value):
        # Swallow assignments coming from Config.__init__'s placeholder.
        self._params_placeholder = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfigAS):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict:
        """Serialize the candidate for race outputs (JSON-friendly).

        Example:
            >>> c = ConfigAS(id='a', source='def f(x): return x', entry_point='f', name='identity')
            >>> d = c.to_dict()
            >>> d['id'], d['name'], d['entry_point']
            ('a', 'identity', 'f')
        """
        return {
            "id": self.id,
            "name": self.name,
            "entry_point": self.entry_point,
            "source": self.source,
            "mean_cost": self.mean_cost,
            "sum_ranks": self.sum_ranks,
            "n_evals": self.n_evals,
            "alive": self.alive,
            "costs": list(self.costs),
            "costs_by_inst": dict(self.costs_by_inst),
            "is_elite": self.is_elite,
        }
