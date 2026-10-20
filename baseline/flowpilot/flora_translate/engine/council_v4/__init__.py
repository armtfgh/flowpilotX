"""FLORA ENGINE v4 — Stage-gated multi-agent engineering design council.

Pipeline:
    Stage 0: Problem Framing  (parse + confirm reaction context)
    Stage 1: Designer         (candidate matrix + hard-gate filter)
    Stage 2: Domain Scoring   (Dr. Chemistry, Dr. Kinetics, Dr. Fluidics, Dr. Safety
                               each SCORE all surviving candidates 0–1)
    Stage 3: Skeptic Audit    (arithmetic verification, scope check, recalc enforcement)
    Final:   Chief Engineer   (weighted selection + DFMEA on winner)

Entry point:  CouncilV4().run(...)
"""

from flora_translate.engine.council_v4.chief import CouncilV4  # noqa: F401
from flora_translate.engine.council_v4.execution_config import (  # noqa: F401
    CouncilExecutionConfig,
)
