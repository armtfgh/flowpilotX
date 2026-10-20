"""Deterministic process-intensification mandate helpers.

Sets IntensificationMandate.tau_reduction_target by reasoning about WHICH
batch limitations the flow design can remove. The chemistry agent identifies
the rate-limiting step in batch (mass-transfer, photon penetration, heat
removal, etc.); each limitation maps to an empirical intensification ceiling
that flow can routinely deliver. The mandate target is the MAX of all
applicable limitation-driven IFs — because removing the worst bottleneck
unlocks at least that intensification factor, and removing several
simultaneously cannot make things worse than the largest of them.

Background motivation: a 15 h aerobic photo-oxidation batch IS NOT a 5×
intensification opportunity. The reason it took 15 h is balloon O2 saturation
and a 5 mm path length where the center of the tube saw no photons. Both are
flow-removable bottlenecks worth >>5×. The previous static default of 6×
for photochem chronically under-intensified long batches.
"""

from __future__ import annotations

import re

from flora_translate.schemas import BatchRecord, ChemistryPlan, IntensificationMandate


# Empirical intensification ceilings for each removable batch limitation.
# Values calibrated against literature flow-chemistry examples in the FLORA
# corpus. A protocol with multiple limitations adopts the max — removing
# the worst bottleneck unlocks at least that IF.
LIMITATION_IF_MAP = {
    "mass_transfer_gas_liquid": 20.0,    # kLa improvement in segmented/Taylor flow
    "photon_penetration": 15.0,           # μ-channel + thin annulus geometry
    "heat_removal": 10.0,                  # high A/V ratio
    "stirring_diffusion": 12.0,            # Taylor + Dean mixing in coils
    "thermodynamic_equilibrium": 1.5,      # cannot intensify around equilibrium
    "kinetic": 3.0,                        # intrinsic rate ceiling
}


# Keyword phrases that signal each batch limitation. Matched case-insensitively
# against the chemistry plan's _reasoning text and the batch description.
# Be conservative — false positives inflate IF; false negatives just keep
# the conservative class default.
LIMITATION_KEYWORDS: dict[str, list[str]] = {
    "mass_transfer_gas_liquid": [
        "o2 mass transfer", "o₂ mass transfer", "oxygen mass transfer",
        "gas mass transfer", "h2 mass transfer", "h₂ mass transfer",
        "hydrogen mass transfer", "gas-liquid mass transfer",
        "balloon saturation", "balloon-fed", "gas dissolution", "gas saturation",
        "kla", "k_l a", "interphase transfer", "co₂ mass transfer",
        "co2 mass transfer", "interfacial area",
        # Strong implicit signals — protocols with gas-fed batch that mention
        # bubbling or saturation are nearly always mass-transfer limited.
        "bubbled through", "bubbling through", "saturated the solution",
        "balloon",  # often "oxygen balloon" or "H2 balloon"
    ],
    "photon_penetration": [
        "photon delivery", "photon flux", "photon penetration",
        "light penetration", "irradiation depth", "beer-lambert",
        "inner filter", "inner-filter", "path-length-limited",
        "path length limited", "thin film irradiation",
        "limited by photon", "light absorption depth",
        # The presence of a screw-cap tube + external LEDs IMPLIES
        # photon-penetration limitation by geometry.
        "irradiated from",  # protocol language for external LED
    ],
    "heat_removal": [
        "heat removal", "heat transfer limit", "cooling-limited",
        "thermal management", "runaway", "exotherm", "exothermic",
        "temperature control limited",
    ],
    "stirring_diffusion": [
        "stirring limited", "agitation limited", "mixing limited",
        "diffusion limited", "diffusion-limited", "viscous mixing",
        "mass transfer in liquid", "stirring",  # broad — softer signal
    ],
    "thermodynamic_equilibrium": [
        "equilibrium-limited", "thermodynamic limit", "le chatelier",
        "reversible reaction at equilibrium",
    ],
}


# Limitations that should NOT count for the long-batch floor (Fix C). These
# are non-flow-removable (equilibrium) or already captured by kinetics.
NON_FLOW_REMOVABLE = {"thermodynamic_equilibrium", "kinetic"}


def detect_batch_limitations(
    *,
    reasoning_text: str = "",
    batch_description: str = "",
    mechanism_text: str = "",
    is_photochem: bool = False,
    is_gas_liquid: bool = False,
    is_exothermic: bool = False,
) -> tuple[list[str], str]:
    """Identify which batch limitations apply.

    Returns (limitations_list, reasoning_string). The reasoning string is
    human-readable: e.g. "mass_transfer_gas_liquid: matched 'balloon saturation'
    in chemistry reasoning + protocol uses balloon-fed O2".
    """
    hits: dict[str, list[str]] = {}
    haystack = " ".join([
        reasoning_text or "",
        batch_description or "",
        mechanism_text or "",
    ]).lower()

    for limitation, kws in LIMITATION_KEYWORDS.items():
        for kw in kws:
            if kw in haystack:
                hits.setdefault(limitation, []).append(kw)

    # Geometric / phase implications add limitations the keywords miss.
    # An aerobic / gas-liquid batch is mass-transfer-limited by definition
    # unless the protocol explicitly avoids it (e.g. dissolved-O2 reaction).
    if is_gas_liquid and "mass_transfer_gas_liquid" not in hits:
        hits["mass_transfer_gas_liquid"] = ["gas_liquid_phase_implies_kLa_limit"]

    # Bulk-irradiated batches (screw-cap tube + external LED, etc.) are
    # photon-penetration limited unless a stirred microreactor was used.
    if is_photochem and "photon_penetration" not in hits:
        # Only add when there's evidence of bulk geometry — keep conservative.
        if any(t in haystack for t in ("tube", "vial", "round-bottom", "flask")):
            hits["photon_penetration"] = ["bulk_photochem_geometry_implies_path_length_limit"]

    # Exothermic flag from chemistry plan is direct evidence.
    if is_exothermic and "heat_removal" not in hits:
        hits["heat_removal"] = ["exothermic_flag_set"]

    limitations = sorted(hits.keys())
    if not limitations:
        # Fall back to kinetic if nothing else fits.
        limitations = ["kinetic"]
        reasoning = "no specific batch limitation identified; defaulting to kinetic (intrinsic rate)"
    else:
        reasoning = "; ".join(
            f"{lim}: matched [{', '.join(kws[:3])}]"
            for lim, kws in sorted(hits.items())
        )
    return limitations, reasoning


def _limitation_driven_IF(limitations: list[str]) -> float:
    """Max IF across all applicable limitations.

    Rationale: if batch is limited by EITHER O2 mass transfer (IF=20) OR
    photon penetration (IF=15), flow can remove either bottleneck and gain
    that intensification. When both apply (typical aerobic photochem),
    removing both unlocks the larger of the two; we don't multiply.
    """
    if not limitations:
        return LIMITATION_IF_MAP["kinetic"]
    return max(LIMITATION_IF_MAP.get(lim, LIMITATION_IF_MAP["kinetic"]) for lim in limitations)


def build_intensification_mandate(
    batch_record: BatchRecord,
    plan: ChemistryPlan | None,
) -> IntensificationMandate:
    """Build a chemistry-aware intensification mandate.

    Replaces the previous static class-default approach with a
    limitation-driven target. The mandate target is the MAX of:
      - class-default IF for the reaction class (legacy floor)
      - limitation-driven IF from the chemistry plan's rate-limiting analysis
      - long-batch sanity floor (Fix C): when batch > 5 h AND at least one
        flow-removable limitation is identified, enforce >= 10×.

    Long batches with mass-transfer or photon-penetration limits CANNOT be
    only-2x intensifiable. The chemistry agent's "rate-limiting step in batch"
    output is the load-bearing signal — read it.
    """
    reaction_class = ((plan.reaction_class if plan else "") or "").lower()
    mechanism = ((plan.mechanism_type if plan else "") or "").lower()
    description = (batch_record.reaction_description or "").lower()

    text = " ".join([reaction_class, mechanism, description])
    is_photo = any(token in text for token in ("photo", "photoredox", "photochem", "light", "led"))
    # Detect gas-liquid systems broadly: explicit o2_is_reagent flag, common
    # aerobic/gas-feed phrases, or evidence in stream_logic that one stream
    # is a reagent gas (O2, H2, etc.). The previous narrower set missed
    # "oxygen bubbled" and "O2 gas" wording common in batch protocols.
    is_gas_liquid = bool(plan and getattr(plan, "o2_is_reagent", False)) or any(
        token in text for token in (
            "gas-liquid", "aerobic", "h2 gas", "h₂ gas", "co gas",
            "o2 gas", "o₂ gas", "oxygen gas",
            "oxygen bubbled", "o2 bubbled", "o₂ bubbled",
            "oxygen balloon", "o2 balloon", "o₂ balloon",
            "balloon-fed",
        )
    )
    # Backup: scan stream_logic for a pure gas reagent stream.
    if not is_gas_liquid and plan is not None:
        for stream in (getattr(plan, "stream_logic", []) or []):
            phase = (getattr(stream, "phase", "") or "").lower()
            reagents = [str(r).lower() for r in (getattr(stream, "reagents", []) or [])]
            if phase == "gas":
                is_gas_liquid = True
                break
            if any(g in r for g in ("o2", "o₂", "h2", "h₂", "co2", "co₂", "cl2", "cl₂", "oxygen", "hydrogen") for r in reagents):
                is_gas_liquid = True
                break
    has_hazardous_intermediate = any(
        token in text
        for token in (
            "diazonium", "peroxide", "organolithium", "azide", "nitrile oxide",
            "chlorine", "phosgene", "hazardous intermediate",
        )
    )
    exothermic = any(token in text for token in ("exotherm", "exothermic", "runaway", "strongly exothermic"))
    selectivity_sensitive = bool(plan and (plan.incompatible_pairs or plan.light_sensitive_reagents))
    if any(token in text for token in ("selectiv", "overreaction", "competing", "side reaction")):
        selectivity_sensitive = True

    # Class-default IF (legacy floor)
    if has_hazardous_intermediate or exothermic or is_photo:
        class_target = 6.0
    elif any(token in text for token in ("thermal", "coupling", "substitution", "addition", "condensation")):
        class_target = 3.0
    else:
        class_target = 2.5

    # ── Fix A: limitation-driven IF ──────────────────────────────────────
    reasoning_text = getattr(plan, "_reasoning", "") if plan else ""
    limitations, limitations_reasoning = detect_batch_limitations(
        reasoning_text=reasoning_text,
        batch_description=batch_record.reaction_description or "",
        mechanism_text=mechanism,
        is_photochem=is_photo,
        is_gas_liquid=is_gas_liquid,
        is_exothermic=exothermic,
    )
    limitation_IF = _limitation_driven_IF(limitations)

    # ── Fix C: long-batch sanity floor ───────────────────────────────────
    batch_time_h = float(batch_record.reaction_time_h or 0.0)
    flow_removable = [lim for lim in limitations if lim not in NON_FLOW_REMOVABLE]
    long_batch_floor = 0.0
    if batch_time_h > 5.0 and flow_removable:
        long_batch_floor = 10.0

    # ── Combine: take the max of all anchors ────────────────────────────
    target = max(class_target, limitation_IF, long_batch_floor)

    # Persist limitation info back into the plan (Fix A schema).
    if plan is not None:
        try:
            plan.batch_limitations = limitations
            plan.batch_limitations_reasoning = limitations_reasoning
        except (AttributeError, ValueError):
            # Older schema / non-pydantic — skip silently.
            pass

    # ── Advantage + regime (unchanged from legacy logic) ────────────────
    if exothermic:
        advantage = "heat_transfer"
    elif has_hazardous_intermediate:
        advantage = "hazardous_intermediate"
    elif "mass_transfer_gas_liquid" in limitations:
        advantage = "mass_transfer"
    elif "photon_penetration" in limitations:
        advantage = "photon_delivery"
    elif selectivity_sensitive:
        advantage = "selectivity"
    elif is_photo:
        advantage = "productivity"
    else:
        advantage = "productivity"

    if selectivity_sensitive or exothermic:
        regime = "enhanced_laminar_mixing"
    elif has_hazardous_intermediate:
        regime = "slug_flow"
    elif "mass_transfer_gas_liquid" in limitations:
        regime = "taylor_slug_flow"
    else:
        regime = "laminar_acceptable"

    features: list[str] = []
    if "mass_transfer_gas_liquid" in limitations:
        features.append("gas-liquid kLa enhancement (segmented flow)")
    if "photon_penetration" in limitations:
        features.append("photon penetration via small ID")
    if "heat_removal" in limitations:
        features.append("heat removal via high A/V")
    if "stirring_diffusion" in limitations:
        features.append("micromixing in coils")
    if has_hazardous_intermediate:
        features.append("hazardous-intermediate holdup reduction")
    if selectivity_sensitive:
        features.append("selectivity-sensitive stream logic")
    if not features:
        features.append("productivity and reactor-volume reduction")

    basis_parts = [
        f"Predicted screening target: {target:.1f}x residence-time reduction; "
        "this is a hypothesis unless the run explicitly selects the intensify policy.",
        f"Identified batch limitations: {', '.join(limitations)}.",
        f"Primary flow advantage: {advantage}.",
        f"Mechanisms: {', '.join(features)}.",
    ]
    if long_batch_floor > 0 and target == long_batch_floor:
        basis_parts.append(
            f"Long-batch floor applied: {batch_time_h:.1f} h batch with flow-removable bottlenecks."
        )
    if limitation_IF > class_target:
        basis_parts.append(
            f"Limitation-driven IF ({limitation_IF:.0f}x) exceeds class default ({class_target:.0f}x)."
        )
    basis = " ".join(basis_parts)

    return IntensificationMandate(
        tau_reduction_target=target,
        minimum_flow_advantage=advantage,
        required_mixing_regime=regime,
        flow_justification_basis=basis,
    )


def ensure_intensification_mandate(
    batch_record: BatchRecord,
    plan: ChemistryPlan,
) -> ChemistryPlan:
    mandate = getattr(plan, "intensification_mandate", None)
    needs_rebuild = (
        mandate is None
        or not getattr(mandate, "flow_justification_basis", "")
        # If the plan's batch_limitations haven't been populated yet, rebuild
        # so the new limitation-driven logic is applied.
        or not getattr(plan, "batch_limitations", None)
    )
    if needs_rebuild:
        plan.intensification_mandate = build_intensification_mandate(batch_record, plan)
    return plan
