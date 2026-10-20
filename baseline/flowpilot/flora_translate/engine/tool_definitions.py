"""
FLORA ENGINE — Tool schemas and dispatcher for council tool-calling.

Defines Anthropic-format tool schemas, an execute_tool dispatcher, and
named tool sets for each agent role.  OpenAI-format conversion is provided
via to_openai_tools().
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
#  Anthropic-format tool schemas
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_BEER_LAMBERT = {
    "name": "beer_lambert",
    "description": (
        "Compute Beer-Lambert absorbance for photocatalyst in a tube. "
        "Returns absorbance A, inner_filter_risk (LOW/MODERATE/HIGH), "
        "percent_absorbed, and recommended_action. "
        "Use to verify photon penetration for a specific concentration, "
        "extinction coefficient, and tubing ID."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "concentration_M": {
                "type": "number",
                "description": "Substrate/catalyst concentration in mol/L",
            },
            "extinction_coeff_M_cm": {
                "type": "number",
                "description": "Molar extinction coefficient in M\u207b\u00b9cm\u207b\u00b9",
            },
            "path_length_mm": {
                "type": "number",
                "description": "Tubing inner diameter in mm (= optical path length)",
            },
        },
        "required": ["concentration_M", "extinction_coeff_M_cm", "path_length_mm"],
    },
}

TOOL_CALCULATE_REYNOLDS = {
    "name": "calculate_reynolds",
    "description": (
        "Compute Reynolds number and flow regime for a candidate. "
        "Returns Re, velocity_m_s, flow_regime (laminar/transitional/turbulent). "
        "Use to verify that a modified Q or d stays laminar."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flow_rate_mL_min": {"type": "number"},
            "tubing_ID_mm": {"type": "number"},
            "solvent": {"type": "string"},
            "temperature_C": {"type": "number"},
        },
        "required": ["flow_rate_mL_min", "tubing_ID_mm", "solvent"],
    },
}

TOOL_CALCULATE_PRESSURE_DROP = {
    "name": "calculate_pressure_drop",
    "description": (
        "Compute Hagen-Poiseuille pressure drop \u0394P = 128\u03bcLQ/\u03c0d\u2074. "
        "Returns delta_P_bar. "
        "Use to probe sensitivity: what happens to \u0394P if Q increases 20%, "
        "or if there is a partial blockage?"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flow_rate_mL_min": {"type": "number"},
            "tubing_ID_mm": {"type": "number"},
            "length_m": {"type": "number"},
            "solvent": {"type": "string"},
        },
        "required": ["flow_rate_mL_min", "tubing_ID_mm", "length_m", "solvent"],
    },
}

TOOL_CALCULATE_MIXING_RATIO = {
    "name": "calculate_mixing_ratio",
    "description": (
        "Compute mixing ratio r_mix = t_mix/\u03c4 where t_mix = d\u00b2/(4D). "
        "Returns mixing_ratio and whether mixing is limiting "
        "(actionable when ratio > 0.20)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tubing_ID_mm": {"type": "number"},
            "residence_time_min": {"type": "number"},
        },
        "required": ["tubing_ID_mm", "residence_time_min"],
    },
}

TOOL_ESTIMATE_RESIDENCE_TIME = {
    "name": "estimate_residence_time",
    "description": (
        "Estimate flow residence time \u03c4 from batch time using class-level "
        "intensification factor. Returns tau_center_min, tau_low_min, tau_high_min. "
        "Use to independently verify whether a candidate's \u03c4 is adequate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "batch_time_h": {
                "type": "number",
                "description": "Batch reaction time in hours",
            },
            "reaction_class": {
                "type": "string",
                "description": "e.g. photoredox, thermal, radical, cross_coupling",
            },
            "if_override": {
                "type": "number",
                "description": "Optional: override the class IF with a specific value",
            },
        },
        "required": ["batch_time_h", "reaction_class"],
    },
}

TOOL_CHECK_REDOX_FEASIBILITY = {
    "name": "check_redox_feasibility",
    "description": (
        "Check SET thermodynamic feasibility. "
        "Returns feasible (bool), margin_V, verdict. "
        "Use to verify whether the photocatalyst excited state has sufficient "
        "driving force for the substrate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "E_excited_V_SCE": {
                "type": "number",
                "description": "Excited state reduction potential in V vs SCE",
            },
            "E_substrate_ox_V_SCE": {
                "type": "number",
                "description": "Substrate oxidation potential in V vs SCE",
            },
            "mode": {
                "type": "string",
                "enum": ["oxidative", "reductive"],
                "description": "Quenching mode",
            },
        },
        "required": ["E_excited_V_SCE", "E_substrate_ox_V_SCE"],
    },
}

TOOL_CHECK_MATERIAL_COMPATIBILITY = {
    "name": "check_material_compatibility",
    "description": (
        "Check tubing/fitting material compatibility with solvent at operating "
        "temperature. Returns compatible (bool), concern (string or null), "
        "temperature_margin_C."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "material": {
                "type": "string",
                "description": "e.g. FEP, PFA, PTFE, SS316, PEEK",
            },
            "solvent": {"type": "string"},
            "temperature_C": {"type": "number"},
        },
        "required": ["material", "solvent", "temperature_C"],
    },
}

TOOL_CALCULATE_BPR_REQUIRED = {
    "name": "calculate_bpr_required",
    "description": "Calculate minimum required BPR setting using Antoine equation. Returns P_min_bar, P_recommended_bar, P_vapor_bar. Gas-liquid systems always require BPR ≥ 5 bar. Use to verify BPR adequacy for safety.",
    "input_schema": {
        "type": "object",
        "properties": {
            "temperature_C": {"type": "number"},
            "solvent": {"type": "string"},
            "delta_P_system_bar": {"type": "number", "description": "System pressure drop in bar"},
            "is_gas_liquid": {"type": "boolean", "description": "True if gas-liquid system"}
        },
        "required": ["temperature_C", "solvent", "delta_P_system_bar"]
    }
}

TOOL_COMPUTE_DESIGN_ENVELOPE = {
    "name": "compute_design_envelope",
    "description": (
        "Compute the \u00b130% feasible operating window around a candidate's "
        "(\u03c4, Q, d, T, BPR). Returns min/center/max ranges for each parameter "
        "with constraints applied (Re<2300, \u0394P<80% pump_max). "
        "Use to assess how robust a winning candidate is to lab variability."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tau_center_min": {"type": "number"},
            "Q_center_mL_min": {"type": "number"},
            "ID_center_mm": {"type": "number"},
            "T_center_C": {"type": "number"},
            "BPR_center_bar": {"type": "number"},
            "solvent": {"type": "string"},
            "is_gas_liquid": {"type": "boolean"},
            "pump_max_bar": {"type": "number"},
        },
        "required": [
            "tau_center_min", "Q_center_mL_min", "ID_center_mm",
            "T_center_C", "BPR_center_bar", "solvent",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Named tool sets per agent role
# ═══════════════════════════════════════════════════════════════════════════════

SKEPTIC_TOOLS = [
    TOOL_BEER_LAMBERT,
    TOOL_CALCULATE_REYNOLDS,
    TOOL_CALCULATE_PRESSURE_DROP,
    TOOL_ESTIMATE_RESIDENCE_TIME,
]

KINETICS_TOOLS = [
    TOOL_ESTIMATE_RESIDENCE_TIME,
    TOOL_CHECK_REDOX_FEASIBILITY,
]

FLUIDICS_TOOLS = [
    TOOL_CALCULATE_PRESSURE_DROP,
    TOOL_CALCULATE_REYNOLDS,
    TOOL_CALCULATE_MIXING_RATIO,
]

PHOTONICS_TOOLS = [
    TOOL_BEER_LAMBERT,
]

CHEMISTRY_TOOLS = [
    TOOL_BEER_LAMBERT,
    TOOL_CHECK_MATERIAL_COMPATIBILITY,
    TOOL_CHECK_REDOX_FEASIBILITY,
]

SAFETY_TOOLS = [
    TOOL_CHECK_MATERIAL_COMPATIBILITY,
    TOOL_CALCULATE_BPR_REQUIRED,
]

CHIEF_TOOLS = [
    TOOL_COMPUTE_DESIGN_ENVELOPE,
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool executor dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

from flora_translate.engine.tools import (  # noqa: E402  (import after schemas)
    beer_lambert,
    calculate_reynolds,
    calculate_pressure_drop,
    calculate_mixing_ratio,
    estimate_residence_time,
    check_redox_feasibility,
    check_material_compatibility,
    compute_design_envelope,
    calculate_bpr_required,
)

_TOOL_MAP = {
    "beer_lambert": beer_lambert,
    "calculate_reynolds": calculate_reynolds,
    "calculate_pressure_drop": calculate_pressure_drop,
    "calculate_mixing_ratio": calculate_mixing_ratio,
    "estimate_residence_time": estimate_residence_time,
    "check_redox_feasibility": check_redox_feasibility,
    "check_material_compatibility": check_material_compatibility,
    "compute_design_envelope": compute_design_envelope,
    "calculate_bpr_required": calculate_bpr_required,
}


def execute_tool(name: str, input_dict: dict) -> dict:
    """Dispatch a tool call by name.  Returns a JSON-serialisable dict."""
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**input_dict)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenAI-format converter
# ═══════════════════════════════════════════════════════════════════════════════

def to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema list to OpenAI function_calling format."""
    result = []
    for t in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return result
