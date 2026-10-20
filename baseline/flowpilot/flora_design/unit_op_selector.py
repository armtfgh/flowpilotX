"""FLORA-Design — Unit Operation Selector: rule-based, no LLM call."""

import logging

from flora_design.rules.material_compatibility import get_reactor_material
from flora_design.rules.unit_op_rules import PHOTOCHEMISTRY_CLASSES, SOLVENT_BP
from flora_translate.schemas import ChemFeatures, UnitOperation

logger = logging.getLogger("flora.design.unit_ops")


class UnitOpSelector:
    """Deterministically select required unit operations from ChemFeatures."""

    def select(self, features: ChemFeatures) -> list[UnitOperation]:
        ops: list[UnitOperation] = []

        # --- ALWAYS: Pumps ---
        ops.append(UnitOperation(
            op_id="pump_a", op_type="pump",
            label="Pump A — Reagent Stream 1",
            parameters={"stream": "reagent_stream_1"},
            required=True,
            rationale="Primary reagent delivery",
        ))
        ops.append(UnitOperation(
            op_id="pump_b", op_type="pump",
            label="Pump B — Reagent Stream 2",
            parameters={"stream": "reagent_stream_2"},
            required=True,
            rationale="Secondary reagent or catalyst delivery",
        ))

        # --- ALWAYS: Mixer ---
        ops.append(UnitOperation(
            op_id="mixer_1", op_type="mixer",
            label="T-Mixer",
            parameters={"type": "T-mixer", "material": "PEEK"},
            required=True,
            rationale="Combine reagent streams before reactor",
        ))

        # --- CONDITIONAL: Deoxygenation ---
        if features.O2_sensitive:
            ops.append(UnitOperation(
                op_id="deoxy_1", op_type="deoxygenation_unit",
                label="Inline N2 Deoxygenation",
                parameters={"method": "inline_N2_sparging"},
                required=True,
                rationale="Reaction is O2-sensitive — deoxygenation required",
            ))

        # --- ALWAYS: Flow reactor (label and params depend on chemistry type) ---
        is_photochem = (
            features.reaction_class in PHOTOCHEMISTRY_CLASSES
            or features.wavelength_nm is not None
        )
        reactor_material = get_reactor_material(
            features.solvent, features.temperature_C
        )
        # Transparent tubing needed for photochem; opaque (SS/PTFE) fine for thermal
        if not is_photochem and features.temperature_C and features.temperature_C > 100:
            reactor_material = "SS"

        reactor_label = (
            f"{reactor_material} Photoreactor Coil"
            if is_photochem
            else f"{reactor_material} Flow Reactor Coil"
        )
        reactor_rationale = (
            "Photochemical reactor — transparent tubing required for light access"
            if is_photochem
            else "Continuous flow reactor coil"
        )
        ops.append(UnitOperation(
            op_id="reactor_1", op_type="coil_reactor",
            label=reactor_label,
            parameters={
                "material": reactor_material,
                "ID_mm": 1.0,
                "light_required": is_photochem,
                "wavelength_nm": features.wavelength_nm if is_photochem else None,
            },
            required=True,
            rationale=reactor_rationale,
        ))

        # --- CONDITIONAL: LED Module (photochemistry only) ---
        if is_photochem:
            wl_label = f" ({features.wavelength_nm:.0f} nm)" if features.wavelength_nm else ""
            ops.append(UnitOperation(
                op_id="led_1", op_type="led_module",
                label=f"LED Module{wl_label}",
                parameters={"wavelength_nm": features.wavelength_nm},
                required=True,
                rationale="Photoexcitation of photocatalyst",
            ))

        # --- CONDITIONAL: BPR ---
        bpr_needed = features.generates_gas
        if features.temperature_C and features.solvent:
            solvent_bp = SOLVENT_BP.get(features.solvent, 100)
            if features.temperature_C > solvent_bp - 20:
                bpr_needed = True
        if bpr_needed:
            ops.append(UnitOperation(
                op_id="bpr_1", op_type="bpr",
                label="Back-Pressure Regulator",
                parameters={"pressure_bar": 5},
                required=True,
                rationale="Maintain liquid phase / handle gas generation",
            ))

        # --- CONDITIONAL: Inline Filter ---
        if features.has_solid_catalyst or features.generates_precipitate:
            ops.append(UnitOperation(
                op_id="filter_1", op_type="inline_filter",
                label="Inline Filter (10 um)",
                parameters={"pore_size_um": 10},
                required=True,
                rationale="Remove solid catalyst or precipitate inline",
            ))

        # --- CONDITIONAL: Quench ---
        if features.hazard_level == "high" or features.exothermic:
            ops.append(UnitOperation(
                op_id="quench_1", op_type="quench_mixer",
                label="Inline Quench",
                parameters={"reagent": "TBD"},
                required=False,
                rationale="Recommended for exothermic or hazardous chemistry",
            ))

        # --- ALWAYS: Collector ---
        ops.append(UnitOperation(
            op_id="collector_1", op_type="collector",
            label="Product Collection",
            parameters={},
            required=True,
            rationale="Outlet collection",
        ))

        logger.info(f"    Selected {len(ops)} unit operations")
        for op in ops:
            logger.info(f"      {op.op_id}: {op.label}")

        return ops
