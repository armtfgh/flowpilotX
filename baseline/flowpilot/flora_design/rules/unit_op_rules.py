"""Rule-based unit operation selection and parameter defaults.

Reaction classes are grouped by chemistry type.
Photochemistry classes (legacy, fully preserved) are prefixed with
"photo" or use explicit names like "singlet_oxygen_reaction".
Non-photochemistry classes are additive — they do not affect any
existing photochem functionality.
"""

# Reaction class keyword lookup for post-processing classifier output
# ── Photochemistry (original) ──────────────────────────────────────────────
REACTION_KEYWORDS: dict[str, str] = {
    "cycloaddition": "photocycloaddition",
    "[2+2]": "photocycloaddition",
    "[4+2]": "photocycloaddition",
    "Diels-Alder": "photocycloaddition",
    "radical addition": "photoredox_radical_addition",
    "giese": "photoredox_radical_addition",
    "minisci": "photoredox_radical_addition",
    "decarboxylative": "photoredox_radical_addition",
    "C-H activation": "photo_CH_functionalization",
    "C-H functionalization": "photo_CH_functionalization",
    "oxidation": "photocatalytic_oxidation",
    "singlet oxygen": "singlet_oxygen_reaction",
    "1O2": "singlet_oxygen_reaction",
    "cross-coupling": "photoredox_cross_coupling",
    "Suzuki": "photoredox_cross_coupling",
    "Buchwald": "photoredox_cross_coupling",
    "isomerization": "photoisomerization",
    "E/Z": "photoisomerization",
    "reduction": "photocatalytic_reduction",
    "energy transfer": "energy_transfer_photocatalysis",
    "TTET": "energy_transfer_photocatalysis",
    "sensitization": "energy_transfer_photocatalysis",
    "HAT": "photoredox_radical_addition",
    "atom transfer": "photoredox_radical_addition",

    # ── Thermal / transition metal catalysis ──────────────────────────────
    "Suzuki-Miyaura": "thermal_cross_coupling",
    "Negishi": "thermal_cross_coupling",
    "Heck": "thermal_cross_coupling",
    "Sonogashira": "thermal_cross_coupling",
    "amination": "thermal_cross_coupling",
    "Buchwald-Hartwig": "thermal_cross_coupling",
    "hydrogenation": "thermal_hydrogenation",
    "asymmetric hydrogenation": "thermal_hydrogenation",
    "nitro reduction": "thermal_hydrogenation",
    "SNAr": "thermal_substitution",
    "nucleophilic substitution": "thermal_substitution",
    "acylation": "thermal_acylation",
    "esterification": "thermal_acylation",
    "amide coupling": "thermal_acylation",
    "aldol": "thermal_condensation",
    "Knoevenagel": "thermal_condensation",
    "Wittig": "thermal_condensation",
    "Grignard": "thermal_organometallic",
    "organolithium": "thermal_organometallic",
    "ozonolysis": "thermal_oxidation",
    "dihydroxylation": "thermal_oxidation",
    "epoxidation": "thermal_oxidation",

    # ── Electrochemistry ───────────────────────────────────────────────────
    "electrochemical": "electrochemical_synthesis",
    "electrosynthesis": "electrochemical_synthesis",
    "anodic oxidation": "electrochemical_synthesis",
    "cathodic reduction": "electrochemical_synthesis",
    "paired electrolysis": "electrochemical_synthesis",
    "Kolbe": "electrochemical_synthesis",

    # ── Biocatalysis ───────────────────────────────────────────────────────
    "enzymatic": "biocatalysis",
    "enzyme": "biocatalysis",
    "lipase": "biocatalysis",
    "transaminase": "biocatalysis",
    "whole cell": "biocatalysis",
    "biocatalysis": "biocatalysis",

    # ── Organocatalysis ────────────────────────────────────────────────────
    "organocatalytic": "organocatalysis",
    "NHC": "organocatalysis",
    "DMAP": "organocatalysis",
    "proline catalysis": "organocatalysis",
    "phase transfer": "organocatalysis",
}

# Default parameters by reaction class
PARAMETER_DEFAULTS: dict[str, dict] = {
    "photocycloaddition": {
        "residence_time_min": 10, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "photoredox_radical_addition": {
        "residence_time_min": 8, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "photo_CH_functionalization": {
        "residence_time_min": 15, "temperature_C": 40,
        "concentration_M": 0.05, "flow_rate_mL_min": 0.3,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "photocatalytic_oxidation": {
        "residence_time_min": 10, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "singlet_oxygen_reaction": {
        "residence_time_min": 5, "temperature_C": 0,
        "concentration_M": 0.05, "flow_rate_mL_min": 1.0,
        "tubing_ID_mm": 1.5, "BPR_bar": 10,
    },
    "photoredox_cross_coupling": {
        "residence_time_min": 20, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.3,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "photoisomerization": {
        "residence_time_min": 5, "temperature_C": 25,
        "concentration_M": 0.05, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 0,
    },
    "photocatalytic_reduction": {
        "residence_time_min": 12, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "energy_transfer_photocatalysis": {
        "residence_time_min": 15, "temperature_C": 25,
        "concentration_M": 0.05, "flow_rate_mL_min": 0.3,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    # ── Thermal / transition metal catalysis ──────────────────────────────
    "thermal_cross_coupling": {
        "residence_time_min": 20, "temperature_C": 80,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 10,
    },
    "thermal_hydrogenation": {
        "residence_time_min": 15, "temperature_C": 50,
        "concentration_M": 0.1, "flow_rate_mL_min": 1.0,
        "tubing_ID_mm": 1.0, "BPR_bar": 20,
    },
    "thermal_substitution": {
        "residence_time_min": 10, "temperature_C": 60,
        "concentration_M": 0.2, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "thermal_acylation": {
        "residence_time_min": 5, "temperature_C": 25,
        "concentration_M": 0.2, "flow_rate_mL_min": 1.0,
        "tubing_ID_mm": 1.0, "BPR_bar": 0,
    },
    "thermal_condensation": {
        "residence_time_min": 15, "temperature_C": 50,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "thermal_organometallic": {
        "residence_time_min": 5, "temperature_C": 0,
        "concentration_M": 0.2, "flow_rate_mL_min": 1.0,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
    "thermal_oxidation": {
        "residence_time_min": 10, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },

    # ── Electrochemistry ───────────────────────────────────────────────────
    "electrochemical_synthesis": {
        "residence_time_min": 15, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 0,
    },

    # ── Biocatalysis ───────────────────────────────────────────────────────
    "biocatalysis": {
        "residence_time_min": 30, "temperature_C": 37,
        "concentration_M": 0.05, "flow_rate_mL_min": 0.2,
        "tubing_ID_mm": 1.5, "BPR_bar": 0,
    },

    # ── Organocatalysis ────────────────────────────────────────────────────
    "organocatalysis": {
        "residence_time_min": 20, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.3,
        "tubing_ID_mm": 1.0, "BPR_bar": 0,
    },

    # ── Fallback ───────────────────────────────────────────────────────────
    "unknown": {
        "residence_time_min": 10, "temperature_C": 25,
        "concentration_M": 0.1, "flow_rate_mL_min": 0.5,
        "tubing_ID_mm": 1.0, "BPR_bar": 5,
    },
}

# Chemistry classes that require light (used by UnitOpSelector)
PHOTOCHEMISTRY_CLASSES = {
    "photocycloaddition",
    "photoredox_radical_addition",
    "photo_CH_functionalization",
    "photocatalytic_oxidation",
    "singlet_oxygen_reaction",
    "photoredox_cross_coupling",
    "photoisomerization",
    "photocatalytic_reduction",
    "energy_transfer_photocatalysis",
}

# Solvent boiling points (shared with config.py but kept here for selector)
SOLVENT_BP: dict[str, float] = {
    "MeCN": 82, "acetonitrile": 82, "DMF": 153, "DMSO": 189,
    "THF": 66, "DCM": 40, "dichloromethane": 40, "MeOH": 65,
    "EtOH": 78, "toluene": 111, "water": 100, "DMA": 165,
    "NMP": 202, "EtOAc": 77, "acetone": 56, "dioxane": 101,
}
