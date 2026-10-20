"""FlowPilot engineering package with lazy public imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_IMPORTS = {
    "CouncilV3": ("flora_translate.engine.council_v3", "CouncilV3"),
    "CouncilV4": ("flora_translate.engine.council_v4", "CouncilV4"),
    "call_llm": ("flora_translate.engine.llm_agents", "call_llm"),
    "beer_lambert": ("flora_translate.engine.tools", "beer_lambert"),
    "check_material_compatibility": ("flora_translate.engine.tools", "check_material_compatibility"),
    "calculate_bpr_required": ("flora_translate.engine.tools", "calculate_bpr_required"),
    "calculate_mixing_ratio": ("flora_translate.engine.tools", "calculate_mixing_ratio"),
    "calculate_reynolds": ("flora_translate.engine.tools", "calculate_reynolds"),
    "calculate_pressure_drop": ("flora_translate.engine.tools", "calculate_pressure_drop"),
    "check_redox_feasibility": ("flora_translate.engine.tools", "check_redox_feasibility"),
    "estimate_residence_time": ("flora_translate.engine.tools", "estimate_residence_time"),
    "compute_design_envelope": ("flora_translate.engine.tools", "compute_design_envelope"),
}

__all__ = list(_PUBLIC_IMPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
