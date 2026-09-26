"""Shared Successive-Halving mixin for the HiFo racing runners.

``sh/hifo_{obp,tsp_gls,fssp_gls}.py`` are the Successive-Halving counterparts of
``racing/hifo_{obp,tsp_gls,fssp_gls}.py`` (and the HiFo counterparts of
``sh/eoh_{obp,tsp_gls,fssp_gls}.py``): they drive the SAME HiFo-Prompt candidate
generation (InsightPool + EvolutionaryNavigator + the ``m3`` operator, with the nested
per-operator sub-races) but swap the Friedman/Nemenyi ``elitist_race`` for an elitist
**Successive-Halving** sub-race.

Design (mirrors ``sh/llamea_bbob.SHLLaMEA(RacingLLaMEA)``):
``SHHiFo(SHHiFoRaceMixin, RacingHiFo)``. The mixin is placed FIRST in the bases so its
``_race`` (the SH ladder) overrides ``RacingHiFo._race`` (the Friedman race) via the
MRO, while every candidate-generation method (``_sample_init`` / ``_sample_op_batch`` /
``_evolve`` / Hindsight-Foresight updates / logging) still resolves to ``RacingHiFo``.
Because ``RacingHiFo._evolve`` runs one sub-race per operator, each of the five
per-operator sub-races becomes an SH ladder (per-operator SH).

The SH sub-race itself is reused VERBATIM from ``sh.eoh_obp.SHEoH._race`` — it is fully
generic over ``RacingBase`` ``CandidateRecord``s (it only calls ``self._ensure_task_pool``
and ``self._make_runner`` plus the SH-schedule attributes this mixin sets), so the OBP /
TSP-GLS / FSSP-GLS HiFo runners share one implementation with the EoH SH runners.
"""

from __future__ import annotations

from sh.eoh_obp import SHEoH, _sh_schedule  # noqa: F401  (_sh_schedule re-exported)


class SHHiFoRaceMixin:
    """Turns a ``RacingHiFo`` into an elitist Successive-Halving runner.

    Sets the SH schedule parameters (``sh_reduction_factor``/``sh_min_instances``) that
    the reused ``_race`` reads, then defers the rest of construction to ``RacingHiFo``.
    """

    def __init__(self, *, sh_reduction_factor: float = 1.25,
                 sh_min_instances: int = 5, **kwargs):
        self.sh_reduction_factor = max(1.1, float(sh_reduction_factor))
        self.sh_min_instances = max(1, int(sh_min_instances))
        super().__init__(**kwargs)

    # Reuse SHEoH's generic SH sub-race (over RacingBase CandidateRecords). Placed first
    # in the MRO, this overrides RacingHiFo._race (the Friedman/Nemenyi race).
    _race = SHEoH._race
