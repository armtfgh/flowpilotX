"""FLORA ENGINE v3 — Advocacy-based council.

Pipeline:
    Calculator center-point
        → Designer  (microfluidics expert; picks sampling strategy; tools generate)
        → Expert    (Kinetics/Fluidics/Photonics/Chemistry specialists ADVOCATE)
        → Skeptic   (attacks assumptions; cites safety rules)
        → (loop up to 2 rounds: Expert refines, Skeptic attacks again)
        → Chief     (resolves against user objectives; emits final FlowProposal patch)

Entry point:  CouncilV3().run(...)
"""

from flora_translate.engine.council_v3.chief import CouncilV3   # noqa: F401
