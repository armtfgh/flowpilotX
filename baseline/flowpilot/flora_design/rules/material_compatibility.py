"""Tubing material / solvent compatibility rules."""

# (material, solvent) → concern string
INCOMPATIBLE_COMBOS: dict[tuple[str, str], str] = {
    ("FEP", "THF"): "FEP swells in THF above 60°C",
    ("PTFE", "DCM"): "PTFE not recommended for DCM at elevated temperature",
    ("SS", "HCl"): "SS corrodes in HCl",
    ("FEP", "toluene"): "FEP may swell in toluene above 80°C",
}

# Recommended material by solvent when default is problematic
MATERIAL_OVERRIDE: dict[str, str] = {
    "THF": "PFA",
    "toluene": "PFA",
}


def get_reactor_material(solvent: str | None, temperature_C: float | None) -> str:
    """Return recommended tubing material given solvent and temperature."""
    if solvent and temperature_C and temperature_C > 50:
        override = MATERIAL_OVERRIDE.get(solvent)
        if override:
            return override
    return "FEP"  # default: transparent for photochemistry
