"""FLORA-Translate — All Pydantic v2 data models."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Hardware / Lab Inventory
# ---------------------------------------------------------------------------


LAB_INVENTORY_SCHEMA_VERSION = "flowpilot_lab_inventory_v3.0"
LEGACY_LAB_INVENTORY_SCHEMA_VERSION = "flowpilot_lab_inventory_legacy"


_GAS_SPECIES_PATTERN = re.compile(
    r"^\s*(?:o2|oxygen(?:\s+gas)?|h2|hydrogen(?:\s+gas)?|co2|carbon dioxide|"
    r"carbon monoxide|n2|nitrogen(?:\s+gas)?|argon|air|cl2|chlorine|nh3|ammonia)"
    r"(?:\s|\(|$)",
    re.IGNORECASE,
)


def normalized_stream_phase(
    phase: Any,
    species: list[Any] | None = None,
    *,
    gas_flow_sccm: Any = None,
    gas_flow_actual_mL_min: Any = None,
) -> str:
    """Return a stable stream phase using structured fields, never prose.

    LLM reasoning often mentions gases that must be excluded from a liquid
    stream. Reasoning text is therefore intentionally not an input here.
    """

    declared = str(phase or "").strip().lower().replace("-", "_")
    aliases = {
        "g": "gas",
        "vapour": "gas",
        "vapor": "gas",
        "l": "liquid",
        "solution": "liquid",
        "slurry": "solid_liquid",
        "suspension": "solid_liquid",
        "s": "solid",
    }
    declared = aliases.get(declared, declared)
    if declared in {"gas", "liquid", "solid", "solid_liquid", "liquid_liquid"}:
        return declared
    if gas_flow_sccm not in (None, "", 0, 0.0) or gas_flow_actual_mL_min not in (
        None,
        "",
        0,
        0.0,
    ):
        return "gas"
    identities = [str(item or "").strip() for item in species or []]
    if any("(gas" in item.lower() for item in identities) or any(
        _GAS_SPECIES_PATTERN.search(item)
        and not re.match(r"(?i)^\s*hydrogen\s+peroxide\b", item)
        for item in identities
    ):
        return "gas"
    return "liquid"


class InventoryItemSpec(BaseModel):
    """Common identity and availability fields for physical lab equipment."""

    equipment_id: str = ""
    name: str = ""
    quantity: int = Field(default=1, ge=1)
    service_status: str = "available"
    compatible_systems: list[str] = Field(default_factory=list)
    notes: str = ""
    resource_requirements: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _assign_stable_id(self):
        if not self.equipment_id:
            payload = self.model_dump(exclude={"equipment_id", "notes"})
            identity = "|".join(
                str(payload.get(key) or "")
                for key in (
                    "name", "system", "type", "material", "volume_mL", "ID_mm",
                    "wavelength_nm", "gas",
                )
            )
            slug = re.sub(r"[^a-z0-9]+", "_", identity.lower()).strip("_")[:48]
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
            prefix = self.__class__.__name__.replace("Spec", "").lower()
            self.equipment_id = f"{prefix}_{slug or 'item'}_{digest}"
        return self


class PumpSpec(InventoryItemSpec):
    type: str  # "HPLC" / "syringe" / "peristaltic"
    max_pressure_bar: float
    max_flow_rate_mL_min: float
    min_flow_rate_mL_min: float
    flow_rate_increment_mL_min: Optional[float] = Field(default=None, gt=0)
    compatible_materials: list[str] = Field(default_factory=list)
    platform_id: str = ""
    excluded_chemicals: list[str] = Field(default_factory=list)


class TubingSpec(InventoryItemSpec):
    material: str  # "FEP" / "PFA" / "SS" / "PTFE"
    ID_mm: float
    max_pressure_bar: float
    max_temperature_C: float
    transparent: Optional[bool] = None
    length_m: Optional[float] = None


class LightSourceSpec(InventoryItemSpec):
    wavelength_nm: float
    power_W: Optional[float] = None
    compatible_reactor: str  # "coil" / "chip" / "both"
    intensity_mW_cm2: Optional[float] = None
    distance_cm: Optional[float] = None
    min_temperature_C: Optional[float] = None
    max_temperature_C: Optional[float] = None
    allowed_temperatures_C: list[float] = Field(default_factory=list)
    module_id: str = ""
    module_name: str = ""
    max_reactor_volume_mL: Optional[float] = None
    compatible_pump_platforms: list[str] = Field(default_factory=list)


class GasHardwareSpec(InventoryItemSpec):
    """One gas-delivery or gas-liquid-contacting inventory item."""

    type: str = ""
    gas: str = ""
    min_flow_sccm: Optional[float] = None
    max_flow_sccm: Optional[float] = None
    flow_rate_increment_sccm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    max_pressure_bar: Optional[float] = None
    required_outlet_accessory_type: str = ""


class ReactorSpec(InventoryItemSpec):
    type: str
    material: str
    volume_mL: float
    ID_mm: float
    system: str = ""
    photoreactor_module_ids: list[str] = Field(default_factory=list)
    light_source: str = ""
    wavelength_nm: Optional[float] = None
    intensity_mW_cm2: Optional[float] = None
    irradiation: str = ""
    min_temperature_C: Optional[float] = None
    max_temperature_C: Optional[float] = None
    allowed_temperatures_C: list[float] = Field(default_factory=list)
    min_concentration_M: Optional[float] = None
    max_concentration_M: Optional[float] = None
    min_pressure_bar: Optional[float] = None
    max_pressure_bar: Optional[float] = None
    configuration: str = ""
    component_volumes_mL: list[float] = Field(default_factory=list)


class MixerSpec(InventoryItemSpec):
    type: str = "T-mixer"
    material: str = ""
    max_inputs: int = Field(default=2, ge=2)
    min_total_flow_mL_min: Optional[float] = None
    max_total_flow_mL_min: Optional[float] = None
    max_pressure_bar: Optional[float] = None
    supported_ID_mm: list[float] = Field(default_factory=list)


class PressureControllerSpec(InventoryItemSpec):
    type: str = "BPR"
    setpoints_bar: list[float] = Field(default_factory=list)
    min_pressure_bar: Optional[float] = None
    max_pressure_bar: Optional[float] = None


class DegasserSpec(InventoryItemSpec):
    type: str = "inline_degasser"
    method: str = ""
    min_flow_mL_min: Optional[float] = None
    max_flow_mL_min: Optional[float] = None
    max_pressure_bar: Optional[float] = None
    compatible_solvents: list[str] = Field(default_factory=list)


class FilterSpec(InventoryItemSpec):
    type: str = "inline_filter"
    material: str = ""
    pore_size_um: Optional[float] = None
    max_flow_mL_min: Optional[float] = None
    max_pressure_bar: Optional[float] = None


class SeparatorSpec(InventoryItemSpec):
    type: str = "phase_separator"
    supported_phases: list[str] = Field(default_factory=list)
    max_flow_mL_min: Optional[float] = None
    max_pressure_bar: Optional[float] = None


class ConnectorSpec(InventoryItemSpec):
    type: str = "union"
    material: str = ""
    supported_ID_mm: list[float] = Field(default_factory=list)
    max_pressure_bar: Optional[float] = None


class TemperatureControllerSpec(InventoryItemSpec):
    type: str = "heater_chiller"
    min_temperature_C: Optional[float] = None
    max_temperature_C: Optional[float] = None
    allowed_temperatures_C: list[float] = Field(default_factory=list)


class CollectorSpec(InventoryItemSpec):
    type: str = "collection_vessel"
    volume_mL: Optional[float] = None
    max_pressure_bar: Optional[float] = None


class ReactorTrainSpec(InventoryItemSpec):
    configuration: str = "serial"
    component_reactor_ids: list[str] = Field(default_factory=list)
    connector_ids: list[str] = Field(default_factory=list)
    total_volume_mL: Optional[float] = None
    max_pressure_bar: Optional[float] = None


class SafetyAccessorySpec(InventoryItemSpec):
    """Safety hardware or facility capability required by executable controls."""

    type: str = "safety_accessory"
    capabilities: list[str] = Field(default_factory=list)
    max_pressure_bar: Optional[float] = None
    compatible_hazards: list[str] = Field(default_factory=list)
    cracking_pressure_bar: Optional[float] = None


class LabInventory(BaseModel):
    schema_version: str = LEGACY_LAB_INVENTORY_SCHEMA_VERSION
    strict_assignment: bool = False
    resource_capacities: dict[str, int] = Field(default_factory=dict)
    standard_reactor_connectors_available: bool = Field(
        default=True,
        description="Standard interstage connectors are stocked; individual serial reactor pairs need not be predeclared. Ratings remain subject to pre-run verification.",
    )
    capability_status: dict[str, Literal["available", "unavailable", "undocumented"]] = Field(default_factory=dict)
    pumps: list[PumpSpec] = Field(default_factory=list)
    tubing: list[TubingSpec] = Field(default_factory=list)
    BPR_available: list[float] = Field(default_factory=list)
    light_sources: list[LightSourceSpec] = Field(default_factory=list)
    gas_hardware: list[GasHardwareSpec] = Field(default_factory=list)
    reactors: list[ReactorSpec] = Field(default_factory=list)
    mixers: list[MixerSpec] = Field(default_factory=list)
    pressure_controllers: list[PressureControllerSpec] = Field(default_factory=list)
    degassers: list[DegasserSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    separators: list[SeparatorSpec] = Field(default_factory=list)
    connectors: list[ConnectorSpec] = Field(default_factory=list)
    temperature_controllers: list[TemperatureControllerSpec] = Field(default_factory=list)
    collectors: list[CollectorSpec] = Field(default_factory=list)
    reactor_trains: list[ReactorTrainSpec] = Field(default_factory=list)
    safety_accessories: list[SafetyAccessorySpec] = Field(default_factory=list)

    @property
    def allows_standard_reactor_connections(self) -> bool:
        return self.standard_reactor_connectors_available and not any(
            self.capability_status.get(key) == "unavailable"
            for key in ("connectors", "connectors_or_reactor_trains")
        )

    @model_validator(mode="after")
    def _ensure_unique_equipment_ids(self):
        seen: dict[str, int] = {}
        for item in self.all_equipment():
            base = item.equipment_id
            seen[base] = seen.get(base, 0) + 1
            if seen[base] > 1:
                item.equipment_id = f"{base}_{seen[base]}"
        return self

    def all_equipment(self) -> list[InventoryItemSpec]:
        categories = (
            self.pumps, self.tubing, self.light_sources, self.gas_hardware,
            self.reactors, self.mixers, self.pressure_controllers, self.degassers,
            self.filters, self.separators, self.connectors,
            self.temperature_controllers, self.collectors, self.reactor_trains,
            self.safety_accessories,
        )
        return [item for category in categories for item in category]

    @classmethod
    def from_json(cls, path: str) -> LabInventory:
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        return cls(**data)


# ---------------------------------------------------------------------------
# Input — Batch Record
# ---------------------------------------------------------------------------


class BatchRecord(BaseModel):
    reaction_description: str = ""
    photocatalyst: Optional[str] = None
    catalyst_loading_mol_pct: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    reaction_time_h: Optional[float] = None
    concentration_M: Optional[float] = None
    scale_mmol: Optional[float] = None
    yield_pct: Optional[float] = None
    light_source: Optional[str] = None
    wavelength_nm: Optional[float] = None
    additives: Optional[list[str]] = None
    atmosphere: Optional[str] = None
    raw_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Intake — standardized pre-design package
# ---------------------------------------------------------------------------


class IntakeQuestion(BaseModel):
    """One reproducible intake question from the fixed FlowPilot question bank."""

    question_id: str
    section: str
    question: str
    expected_format: str = ""
    required: bool = False
    why_needed: str = ""
    origin: Literal["core", "conditional"] = "core"
    trigger_id: str = ""
    target_path: str = ""
    decision_impact: str = ""
    allow_unavailable: bool = True


class IntakeAnswer(BaseModel):
    """User response to a standardized intake question."""

    question_id: str
    answer: Any = None
    status: Literal["answered", "unavailable"] = "answered"
    source: str = "user"


class DesignInputPackage(BaseModel):
    """Frozen, authority-labeled input package used before design starts."""

    schema_version: str = "flowpilot_intake_v1.1"
    question_bank_version: str = "flowpilot_questions_v1.1"
    raw_protocol: str = ""
    extracted_batch_fields: dict[str, Any] = Field(default_factory=dict)
    objective: str = ""
    screening_priority: Literal["auto", "balanced", "yield_priority", "throughput_priority"] = "auto"
    historical_data: Any = None
    inventory_constraints: Any = None
    hypotheses: list[str] = Field(default_factory=list)
    operating_limits: Any = None
    inventory_profile_snapshot: dict[str, Any] = Field(default_factory=dict)
    inventory_review: dict[str, Any] = Field(default_factory=dict)
    output_preferences: str = ""
    chemistry_identity_confirmation: dict[str, Any] = Field(default_factory=dict)
    active_domains: list[str] = Field(default_factory=list)
    active_question_ids: list[str] = Field(default_factory=list)
    engineering_requirements: dict[str, Any] = Field(default_factory=dict)
    question_set_hash: str = ""
    question_log: list[IntakeQuestion] = Field(default_factory=list)
    answers: list[IntakeAnswer] = Field(default_factory=list)
    missing_question_ids: list[str] = Field(default_factory=list)
    ready_for_design: bool = False
    authority_order: list[str] = Field(
        default_factory=lambda: [
            "measured_evidence",
            "hard_constraints",
            "protocol_facts",
            "chemist_hypotheses",
            "model_inference",
        ]
    )

    def answer_map(self) -> dict[str, IntakeAnswer]:
        """Return the latest answer for each question ID."""

        return {answer.question_id: answer for answer in self.answers}


# ---------------------------------------------------------------------------
# Process Records (from PRISM / paper_knowledge_extractor)
# ---------------------------------------------------------------------------


class ConditionsRecord(BaseModel):
    photocatalyst: Optional[str] = None
    catalyst_loading_mol_pct: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    reaction_time_h: Optional[float] = None
    residence_time_min: Optional[float] = None
    flow_rate_mL_min: Optional[float] = None
    concentration_M: Optional[float] = None
    yield_pct: Optional[float] = None
    light_source: Optional[str] = None
    wavelength_nm: Optional[float] = None


class ProcessDesignRecord(BaseModel):
    reactor_type: Optional[str] = None
    tubing_material: Optional[str] = None
    tubing_ID_mm: Optional[float] = None
    tubing_length_m: Optional[float] = None
    reactor_volume_mL: Optional[float] = None
    pump_type: Optional[str] = None
    mixer_type: Optional[str] = None
    BPR_bar: Optional[float] = None
    BPR_required: Optional[bool] = None
    phase_regime: Optional[str] = None
    light_setup: Optional[str] = None
    deoxygenation_method: Optional[str] = None


class EngineeringLogicRecord(BaseModel):
    batch_limitation: Optional[str] = None
    flow_advantage: Optional[str] = None
    safety_concern: Optional[str] = None
    intensification_factor: Optional[float] = None
    batch_to_flow_yield_delta: Optional[float] = None


class TranslationRecord(BaseModel):
    batch_baseline: Optional[ConditionsRecord] = None
    flow_optimized: Optional[ConditionsRecord] = None
    flow_process_design: Optional[ProcessDesignRecord] = None
    reasoning: Optional[str] = None


class ProcessRecord(BaseModel):
    record_id: str = ""
    doi: str = ""
    title: str = ""
    year: int = 0
    chemistry_class: str = ""
    mechanism_type: str = ""        # "radical" | "SET" | "EnT" | "HAT" | "ionic" | ""
    phase_regime: str = "single_phase_liquid"  # "single_phase_liquid" | "gas_liquid" | "liquid_liquid" | "solid_liquid"
    process_mode: str = ""  # "batch" / "flow" / "both"
    conditions: ConditionsRecord = Field(default_factory=ConditionsRecord)
    process_design: Optional[ProcessDesignRecord] = None
    engineering_logic: Optional[EngineeringLogicRecord] = None
    translation_record: Optional[TranslationRecord] = None
    confidence: int = 1
    embedding_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Chemistry Plan (Layer 1 — generated before retrieval and translation)
# ---------------------------------------------------------------------------


class ReagentRole(BaseModel):
    """A single chemical species and its role in the reaction."""
    name: str = ""                       # e.g. "Ir(ppy)3", "sulfenamide 1a"
    role: str = ""                       # "photocatalyst" | "substrate" | "oxidant" | "base" | "additive" | "solvent" | "sensitizer" | "co-catalyst" | "quencher"
    equiv_or_loading: str = ""           # "1.0 equiv" or "2 mol%" — free text
    smiles: Optional[str] = None         # if known
    notes: str = ""                      # e.g. "light-sensitive", "moisture-sensitive"


class MechanismStep(BaseModel):
    """One elementary step in the proposed mechanism."""
    step_number: int = 0
    description: str = ""                # e.g. "Photoexcitation of Ir(III) to *Ir(III)"
    species_involved: list[str] = Field(default_factory=list)
    is_photon_dependent: bool = False    # does this step require light?
    is_rate_limiting: bool = False


RequirementAuthority = Literal[
    "measured_evidence",
    "hard_constraint",
    "protocol_fact",
    "chemist_hypothesis",
    "model_inference",
    "deterministic_derivation",
]


class CanonicalOperationRequirement(BaseModel):
    """One provenance-bearing operation in the immutable reaction contract."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    operation_type: Literal[
        "liquid_feed",
        "gas_feed",
        "mixer",
        "reactor",
        "light_source",
        "pressure_control",
        "interstage_operation",
    ]
    stage_number: int = Field(default=1, ge=1)
    required: bool = True
    accepted_requirement: bool = True
    authority: RequirementAuthority = "deterministic_derivation"
    source_evidence: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    feed_group: str = ""
    rationale: str = ""


class CanonicalReactionContract(BaseModel):
    """Protocol-anchored process facts that downstream models cannot expand."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "flowpilot_canonical_reaction_contract_v1.0"
    protocol_sha256: str = ""
    reaction_name: str = ""
    n_stages: int = Field(default=1, ge=1)
    required_reagent_gases: list[str] = Field(default_factory=list)
    light_required_stages: list[int] = Field(default_factory=list)
    operations: list[CanonicalOperationRequirement] = Field(default_factory=list)
    authority_order: list[str] = Field(
        default_factory=lambda: [
            "measured_evidence",
            "hard_constraint",
            "protocol_fact",
            "chemist_hypothesis",
            "model_inference",
        ]
    )


class StreamLogic(BaseModel):
    """Chemistry-driven reasoning for which reagents go in which stream."""
    stream_label: str = ""               # "A", "B"
    reagents: list[str] = Field(default_factory=list)   # names of reagents in this stream
    reasoning: str = ""                  # WHY these go together
    molar_equiv: float = 1.0             # stoichiometric equivalents relative to limiting reagent
    molar_equiv_basis: str = ""
    concentration_M: Optional[float] = None  # concentration of this stream (for Q calculation)
    phase: str = ""                      # "liquid" | "gas" | "solid" | ""
    gas_flow_sccm: Optional[float] = None
    gas_flow_actual_mL_min: Optional[float] = None
    gas_reagent_mole_fraction: Optional[float] = Field(default=None, gt=0, le=1)
    introduction_stage: Optional[int] = Field(default=None, ge=1)
    delivery_mode: Literal["new_feed", "carried_from_previous"] = "new_feed"
    requirement_authority: RequirementAuthority = "model_inference"
    source_evidence: list[str] = Field(default_factory=list)
    accepted_requirement: bool = True
    separate_feed_required: bool = False
    feed_group: str = ""

    @model_validator(mode="after")
    def _normalize_phase(self):
        self.phase = normalized_stream_phase(
            self.phase,
            self.reagents,
            gas_flow_sccm=self.gas_flow_sccm,
            gas_flow_actual_mL_min=self.gas_flow_actual_mL_min,
        )
        return self


class IntensificationMandate(BaseModel):
    """Why this batch reaction should be translated to flow."""
    tau_reduction_target: Optional[float] = 2.0
    minimum_flow_advantage: str = "productivity"
    required_mixing_regime: str = "laminar_acceptable"
    flow_justification_basis: str = ""


class ProcessStage(BaseModel):
    """One stage of a multi-step flow process.

    Each stage = one reactor zone with its own feeds, conditions, and outlet.
    Stages are connected in sequence: stage N outlet feeds into stage N+1
    (possibly via a quench mixer or solvent switch point).
    """
    stage_number: int = 1
    stage_name: str = ""                  # e.g. "Grignard formation", "Reduction"
    reaction_type: str = ""               # e.g. "nucleophilic addition", "photoredox"
    reactor_type: str = "coil"            # "coil" | "packed_bed" | "chip" | "CSTR"
    temperature_C: Optional[float] = None
    requires_light: bool = False
    wavelength_nm: Optional[float] = None
    batch_time_h: Optional[float] = None       # batch duration for this stage, if stated

    # What feeds into this stage
    feed_streams: list[StreamLogic] = Field(default_factory=list)
    # What comes from the previous stage (empty for stage 1)
    inlet_from_previous: str = ""         # e.g. "crude ArMgBr in THF"

    # Conditions and sensitivities for THIS stage
    solvent: str = ""
    atmosphere: str = ""                  # "N2" | "Ar" | "air" | "H2"
    oxygen_sensitive: bool = False
    moisture_sensitive: bool = False
    deoxygenation_required: bool = False

    # What happens between THIS stage and the NEXT
    post_stage_action: str = ""           # "quench with NH4Cl" | "solvent switch to DMF" | "inline filter" | ""
    post_stage_reasoning: str = ""


class ChemistryPlan(BaseModel):
    """Pure chemistry analysis — no hardware decisions.

    Generated by the Chemistry Reasoning Agent (Layer 1) before
    retrieval or hardware translation.

    For multi-step reactions, the `stages` list describes the process
    as an ordered sequence of stages, each with its own feeds, reactor
    type, conditions, and inter-stage actions. The flat fields
    (stream_logic, deoxygenation_required, etc.) describe the OVERALL
    process or the first stage for backward compatibility.
    """
    scientific_context: dict = Field(default_factory=dict)
    # Reaction identity
    reaction_name: str = ""
    reaction_class: str = ""
    mechanism_type: str = ""
    bond_formed: str = ""
    bond_broken: str = ""

    # Multi-step stages (empty for single-step — backward compatible)
    stages: list[ProcessStage] = Field(default_factory=list)
    n_stages: int = 1

    # All species (across all stages)
    reagents: list[ReagentRole] = Field(default_factory=list)

    # Mechanism
    mechanism_steps: list[MechanismStep] = Field(default_factory=list)
    key_intermediate: str = ""
    excited_state_type: str | None = ""
    energy_transfer_or_redox: str | None = ""

    # Sensitivity and constraints (overall)
    oxygen_sensitive: bool = False        # reaction is INHIBITED by O2 → needs degassing/inert blanket
    o2_is_reagent: bool = False           # O2 (or air) is a REAGENT → needs MFC + BPR + gas-liquid handling
    moisture_sensitive: bool = False
    temperature_sensitive: bool = False
    light_sensitive_reagents: list[str] = Field(default_factory=list)

    # Stream separation logic (overall / stage-1 for backward compat)
    stream_logic: list[StreamLogic] = Field(default_factory=list)
    mixing_order_reasoning: str = ""
    incompatible_pairs: list[list[str]] = Field(default_factory=list)

    # Pre/post reactor chemistry (overall / stage-1)
    deoxygenation_required: bool = False
    deoxygenation_reasoning: str = ""
    quench_required: bool = False
    quench_reagent: str = ""
    quench_reasoning: str = ""

    # Retrieval hints
    retrieval_keywords: list[str] = Field(default_factory=list)
    similar_reaction_classes: list[str] = Field(default_factory=list)

    # Batch rate-limiting bottlenecks that flow CAN remove. Controlled vocabulary:
    #   "mass_transfer_gas_liquid"  — kLa-limited (gas dissolution / contact)
    #   "photon_penetration"        — Beer-Lambert / path-length-limited
    #   "heat_removal"              — exotherm; thermal runaway risk
    #   "stirring_diffusion"        — bulk mixing or molecular diffusion in liquid
    #   "thermodynamic_equilibrium" — fundamentally equilibrium-limited (cannot intensify)
    #   "kinetic"                   — intrinsic rate at given T
    # Drives intensification target — populated from chemistry plan reasoning.
    batch_limitations: list[str] = Field(default_factory=list)
    batch_limitations_reasoning: str = ""

    # Wavelength recommendation
    recommended_wavelength_nm: Optional[float] = None
    wavelength_reasoning: str = ""

    # Confidence
    confidence_notes: str = ""

    # Explicit process-intensification target for downstream design.
    intensification_mandate: IntensificationMandate = Field(default_factory=IntensificationMandate)

    # Frozen after protocol/chemist reconciliation. Downstream stages may use
    # this contract but may not create new required operations outside it.
    canonical_contract: Optional[CanonicalReactionContract] = None
    reconciliation_log: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_stage_semantics(self):
        self.n_stages = max(int(self.n_stages or 1), len(self.stages or []), 1)
        global_streams = {
            str(feed.stream_label or "").strip().upper(): feed
            for feed in self.stream_logic or []
            if str(feed.stream_label or "").strip()
        }
        for stage in self.stages or []:
            for feed in stage.feed_streams or []:
                authoritative = global_streams.get(
                    str(feed.stream_label or "").strip().upper()
                )
                if authoritative is None:
                    continue
                feed.phase = authoritative.phase
                if authoritative.concentration_M is not None:
                    feed.concentration_M = authoritative.concentration_M
                if authoritative.molar_equiv is not None:
                    feed.molar_equiv = authoritative.molar_equiv
                    feed.molar_equiv_basis = authoritative.molar_equiv_basis
                feed.introduction_stage = (
                    authoritative.introduction_stage
                    or feed.introduction_stage
                    or stage.stage_number
                )
        feeds = [
            *list(self.stream_logic or []),
            *[
                feed
                for stage in self.stages or []
                for feed in stage.feed_streams or []
            ],
        ]
        oxygen_reagent_found = False
        for feed in feeds:
            if normalized_stream_phase(
                feed.phase,
                feed.reagents,
                gas_flow_sccm=feed.gas_flow_sccm,
                gas_flow_actual_mL_min=feed.gas_flow_actual_mL_min,
            ) != "gas":
                continue
            identity = " ".join(str(item or "") for item in feed.reagents)
            if re.search(r"(?:^|[^a-z0-9])(?:o2|oxygen|air)(?:[^a-z0-9]|$)", identity, re.I):
                oxygen_reagent_found = True
        if feeds:
            self.o2_is_reagent = oxygen_reagent_found
        return self


# ---------------------------------------------------------------------------
# Output — Flow Proposal
# ---------------------------------------------------------------------------


class StreamAssignment(BaseModel):
    """Describes what goes into a single pump/stream."""
    stream_label: str = ""              # "A", "B", "C"
    pump_role: str = ""                 # e.g. "substrate + photocatalyst", "oxidant"
    contents: list[str] = Field(default_factory=list)  # e.g. ["sulfenamide (0.1M)", "Ir(ppy)3 (1 mol%)"]
    solvent: str = ""
    concentration_M: Optional[float] = None
    concentration_basis: str = ""       # protocol_fact | chemistry_plan | screening_assumption
    flow_rate_mL_min: Optional[float] = None
    phase: str = ""                      # "liquid" | "gas" | ""
    gas_flow_sccm: Optional[float] = None # MFC set point at STP for gas streams
    gas_flow_actual_mL_min: Optional[float] = None # gas volume flow at reactor T/P
    gas_reagent_mole_fraction: Optional[float] = Field(default=None, gt=0, le=1)
    gas_active_species: str = ""
    gas_total_molar_flow_mmol_min: Optional[float] = None
    gas_active_molar_flow_mmol_min: Optional[float] = None
    gas_active_equivalents: Optional[float] = None
    molar_equiv: float = 1.0            # stoichiometric equivalents relative to limiting reagent (substrate=1.0)
    molar_equiv_basis: str = ""          # protocol_fact | chemistry_plan | screening_assumption
    pump_equipment_id: Optional[str] = None  # deterministic inventory assignment
    introduction_stage: int = Field(default=1, ge=1)
    reasoning: str = ""                 # why these go together
    requirement_authority: RequirementAuthority = "model_inference"
    source_evidence: list[str] = Field(default_factory=list)
    accepted_requirement: bool = True
    separate_feed_required: bool = False
    feed_group: str = ""

    @field_validator(
        "stream_label",
        "pump_role",
        "solvent",
        "phase",
        "reasoning",
        "concentration_basis",
        "molar_equiv_basis",
        mode="before",
    )
    @classmethod
    def _none_to_empty_string(cls, value):
        return "" if value is None else value

    @field_validator("contents", mode="before")
    @classmethod
    def _none_to_empty_contents(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _normalize_phase(self):
        self.phase = normalized_stream_phase(
            self.phase,
            self.contents,
            gas_flow_sccm=self.gas_flow_sccm,
            gas_flow_actual_mL_min=self.gas_flow_actual_mL_min,
        )
        return self


class FlowProposal(BaseModel):
    scientific_design: dict = Field(default_factory=dict)
    # Proposed flow conditions
    residence_time_min: float = 0
    flow_rate_mL_min: float = 0
    temperature_C: float = 25
    concentration_M: float = 0.1
    BPR_bar: float = 0
    BPR_basis: str = "gauge"
    pressure_absolute_bar: Optional[float] = None
    reactor_type: str = "coil"
    tubing_material: str = "FEP"
    tubing_ID_mm: float = 1.0
    reactor_volume_mL: float = 0
    residence_time_inlet_min: Optional[float] = None
    residence_time_in_channel_min: Optional[float] = None
    residence_time_basis: str = ""
    light_setup: str = ""
    wavelength_nm: Optional[float] = None
    deoxygenation_method: Optional[str] = None

    # Chemistry-aware stream assignments
    streams: list[StreamAssignment] = Field(default_factory=list)
    mixer_type: str = "T-mixer"
    mixing_order_reasoning: str = ""     # why this mixing order matters
    pre_reactor_steps: list[str] = Field(default_factory=list)  # e.g. ["degas stream A with N2"]
    post_reactor_steps: list[str] = Field(default_factory=list)  # e.g. ["inline quench with Na2S2O3"]
    chemistry_notes: str = ""            # mechanism-specific design notes

    # Per-stage parameters (populated by council Chief for multi-step processes)
    # Each dict: {stage_number, tau_fraction, d_mm, Q_inlet_mL_min, V_R_mL}
    stage_parameters: list[dict] = Field(default_factory=list)

    # Deterministic engineering annotations populated after final calculation.
    multiphase_metrics: dict = Field(default_factory=dict)
    heat_transfer_metrics: dict = Field(default_factory=dict)
    inventory_selection: dict = Field(default_factory=dict)
    inventory_constraints: dict = Field(default_factory=dict)
    evidence_calibration: dict = Field(default_factory=dict)

    # Reasoning
    reasoning_per_field: dict[str, str] = Field(default_factory=dict)
    decision_rationale: dict = Field(default_factory=dict)
    literature_analogies: list[str] = Field(default_factory=list)

    # Status
    engine_validated: bool = False
    safety_flags: list[str] = Field(default_factory=list)
    # Preserve model-supplied bands or numeric confidence as display metadata;
    # confidence never overrides deterministic feasibility gates.
    confidence: str = Field(default="LOW", coerce_numbers_to_str=True)

    @field_validator("pre_reactor_steps", "post_reactor_steps", "literature_analogies", "safety_flags", mode="before")
    @classmethod
    def normalize_text_metadata(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            value = value["items"]
        if isinstance(value, (str, dict)):
            value = [value]
        if isinstance(value, list):
            # Lossless serialization: retain equipment, severity, citations,
            # and instructions, not just the first field of a structured note.
            return [item if isinstance(item, str) else json.dumps(item, sort_keys=True, ensure_ascii=True)
                    for item in value if item is not None]
        return value

    @field_validator("reasoning_per_field", mode="before")
    @classmethod
    def normalize_reasoning_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): item if isinstance(item, str) else json.dumps(item, sort_keys=True, ensure_ascii=True)
                    for key, item in value.items()}
        return value


# ---------------------------------------------------------------------------
# FLORA-Design — Chemistry Features, Unit Operations, Topology
# ---------------------------------------------------------------------------


class ChemFeatures(BaseModel):
    """Structured chemistry features extracted from user goal text."""
    reaction_class: str = "unknown"
    photocatalyst: Optional[str] = None
    photocatalyst_class: Optional[str] = None
    wavelength_nm: Optional[float] = None
    base: Optional[str] = None
    solvent: Optional[str] = None
    temperature_C: Optional[float] = None
    concentration_M: Optional[float] = None
    scale: str = "lab"
    phase_regime: str = "single_phase_liquid"
    generates_gas: bool = False
    has_solid_catalyst: bool = False
    generates_precipitate: bool = False
    O2_sensitive: bool = True
    light_sensitive_product: bool = False
    hazard_level: str = "low"
    exothermic: bool = False
    multi_step: bool = False
    number_of_steps: int = 1
    classifier_confidence: float = 0.0
    ambiguous_fields: list[str] = Field(default_factory=list)


class UnitOperation(BaseModel):
    """A single unit operation in the process."""
    op_id: str = ""
    op_type: str = ""
    label: str = ""
    parameters: dict = Field(default_factory=dict)
    required: bool = True
    rationale: str = ""
    inventory_item_id: Optional[str] = None
    inventory_item_ids: list[str] = Field(default_factory=list)
    instrument_name: str = ""
    inventory_category: str = ""
    assignment_status: str = "unassigned"
    capability_checks: dict[str, bool] = Field(default_factory=dict)
    requirement_authority: RequirementAuthority = "model_inference"
    source_evidence: list[str] = Field(default_factory=list)
    accepted_requirement: bool = True


class StreamConnection(BaseModel):
    """A stream connecting two unit operations."""
    stream_id: str = ""
    from_op: str = ""
    to_op: str = ""
    stream_type: str = "liquid"
    connection_type: Literal["process", "control", "utility", "waste"] = "process"
    label: Optional[str] = None


class ProcessTopology(BaseModel):
    """Complete ordered process topology from inlet to outlet."""
    topology_id: str = ""
    unit_operations: list[UnitOperation] = Field(default_factory=list)
    streams: list[StreamConnection] = Field(default_factory=list)
    total_flow_rate_mL_min: float = 0
    total_flow_rate_basis: str = "sum of liquid feeds at the outlet; reagent gas is reported separately"
    residence_time_min: float = 0
    reactor_volume_mL: float = 0
    pid_description: str = ""
    literature_support: list[str] = Field(default_factory=list)
    topology_confidence: str = "MEDIUM"
    compilation_status: str = "abstract"
    inventory_schema_version: str = ""
    inventory_sha256: str = ""
    instrument_manifest: list[dict[str, Any]] = Field(default_factory=list)


class ExecutableFeed(BaseModel):
    """One inventory-assigned feed edge in the executable process graph."""

    operation_id: str = Field(min_length=1)
    equipment_id: str = Field(min_length=1)
    stream_label: str = Field(min_length=1)
    phase: Literal["liquid", "gas", "solid", "solid_liquid", "liquid_liquid"]
    contents: list[str] = Field(default_factory=list)
    solvent: str = ""
    flow_rate_mL_min: Optional[float] = Field(default=None, gt=0)
    gas_flow_sccm: Optional[float] = Field(default=None, gt=0)
    gas_flow_actual_mL_min: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_phase_flow(self):
        if self.phase == "gas":
            if self.gas_flow_sccm is None and self.gas_flow_actual_mL_min is None:
                raise ValueError("gas feeds require STP or in-channel gas flow")
        elif self.flow_rate_mL_min is None:
            raise ValueError("non-gas feeds require volumetric flow")
        return self


class ExecutableStage(BaseModel):
    """One inventory-assigned reactor stage with a closed flow-time-volume basis."""

    stage_number: int = Field(ge=1)
    reactor_operation_id: str = Field(min_length=1)
    reactor_equipment_id: str = Field(min_length=1)
    mixer_operation_id: Optional[str] = None
    mixer_equipment_id: Optional[str] = None
    temperature_C: float
    reactor_volume_mL: float = Field(gt=0)
    residence_flow_mL_min: float = Field(gt=0)
    residence_time_min: float = Field(gt=0)
    residence_time_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def _close_residence_time(self):
        expected = self.reactor_volume_mL / self.residence_flow_mL_min
        scale = max(abs(expected), abs(self.residence_time_min), 1e-9)
        if abs(expected - self.residence_time_min) / scale > 0.02:
            raise ValueError(
                "stage residence time does not close with reactor volume and flow"
            )
        return self


class ExecutableProcessGraph(BaseModel):
    """Typed, authoritative process graph published as an executable design."""

    schema_version: Literal["flowpilot_executable_process_v1.0"] = (
        "flowpilot_executable_process_v1.0"
    )
    topology: ProcessTopology
    feeds: list[ExecutableFeed] = Field(min_length=1)
    stages: list[ExecutableStage] = Field(min_length=1)
    total_liquid_flow_mL_min: float = Field(gt=0)
    total_reactor_volume_mL: float = Field(gt=0)
    total_residence_time_min: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_graph(self):
        if self.topology.compilation_status not in {
            "inventory_assigned",
            "inventory_assigned_with_accessories",
        }:
            raise ValueError("process graph is not inventory compiled")
        operation_ids = {item.op_id for item in self.topology.unit_operations}
        if len(operation_ids) != len(self.topology.unit_operations):
            raise ValueError("process graph contains duplicate operation IDs")
        for connection in self.topology.streams:
            if connection.from_op not in operation_ids or connection.to_op not in operation_ids:
                raise ValueError("process graph contains a dangling stream endpoint")

        feed_ids = {item.operation_id for item in self.feeds}
        stage_ids = {item.reactor_operation_id for item in self.stages}
        if not feed_ids <= operation_ids or not stage_ids <= operation_ids:
            raise ValueError("typed feeds or stages are absent from the topology")
        if len({item.stage_number for item in self.stages}) != len(self.stages):
            raise ValueError("process graph contains duplicate stage numbers")

        stage_volume = sum(item.reactor_volume_mL for item in self.stages)
        volume_scale = max(stage_volume, self.total_reactor_volume_mL, 1e-9)
        if abs(stage_volume - self.total_reactor_volume_mL) / volume_scale > 0.02:
            raise ValueError("stage volumes do not close with total reactor volume")

        stage_time = sum(item.residence_time_min for item in self.stages)
        time_scale = max(stage_time, self.total_residence_time_min, 1e-9)
        if abs(stage_time - self.total_residence_time_min) / time_scale > 0.02:
            raise ValueError("stage times do not close with total residence time")
        return self


class CanonicalStreamComponent(BaseModel):
    """One chemical component with execution-relevant quantitative provenance."""

    stream_label: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = "unknown"
    source_text: str = ""
    concentration_M: Optional[float] = Field(default=None, gt=0)
    molar_equiv: Optional[float] = Field(default=None, gt=0)
    loading_mol_pct: Optional[float] = Field(default=None, gt=0)
    quantification_required: bool = True
    quantified: bool = False
    provenance: list[str] = Field(default_factory=list)


class ExecutableChemistryIdentity(BaseModel):
    """Frozen chemistry identity that downstream agents may not rewrite."""

    reaction_name: str = ""
    reaction_class: str = ""
    transformation_family: str = "unknown"
    protocol_transformation_family: str = "unknown"
    bond_formed: str = ""
    bond_broken: str = ""
    protocol_sha256: str = Field(min_length=64, max_length=64)
    chemistry_plan_sha256: str = Field(min_length=64, max_length=64)
    authority_source: Literal[
        "protocol_fact", "chemist_confirmed", "model_inference", "legacy_unavailable"
    ] = "model_inference"
    confirmed: bool = False


class ExecutableSafetyControl(BaseModel):
    control_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool = True
    satisfied: bool = False
    evidence: list[str] = Field(default_factory=list)
    equipment_ids: list[str] = Field(default_factory=list)


class ExecutableSafetyContract(BaseModel):
    hazards: list[str] = Field(default_factory=list)
    controls: list[ExecutableSafetyControl] = Field(default_factory=list)
    complete: bool = False
    missing_control_ids: list[str] = Field(default_factory=list)


class ExecutableProcedureStep(BaseModel):
    step_id: str = Field(min_length=1)
    section: Literal[
        "preparation", "setup", "startup", "steady_state", "collection",
        "shutdown", "emergency", "waste",
    ]
    instruction: str = Field(min_length=1)
    required: bool = True
    equipment_ids: list[str] = Field(default_factory=list)
    parameter_bindings: dict[str, Any] = Field(default_factory=dict)


class ExecutableValidationExperiment(BaseModel):
    experiment_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    procedure: str = Field(min_length=1)
    equipment_ids: list[str] = Field(default_factory=list)
    parameter_bindings: dict[str, Any] = Field(default_factory=dict)


class ExecutableArtifactBundle(BaseModel):
    """Non-numeric executable artifacts compiled from one realized design."""

    schema_version: Literal["flowpilot_executable_artifacts_v1.0"] = (
        "flowpilot_executable_artifacts_v1.0"
    )
    chemistry_identity: ExecutableChemistryIdentity
    stream_components: list[CanonicalStreamComponent] = Field(default_factory=list)
    safety: ExecutableSafetyContract
    operating_procedure: list[ExecutableProcedureStep] = Field(default_factory=list)
    validation_experiments: list[ExecutableValidationExperiment] = Field(
        default_factory=list
    )


class DesignResult(BaseModel):
    """Final output of FLORA-Design."""
    goal_text: str = ""
    chem_features: ChemFeatures = Field(default_factory=ChemFeatures)
    topology: ProcessTopology = Field(default_factory=ProcessTopology)
    design_candidate: Optional[DesignCandidate] = None
    svg_path: str = ""
    png_path: str = ""
    diagram_artifacts: dict = Field(default_factory=dict)
    diagram_render_manifest: dict = Field(default_factory=dict)
    inventory_allocation: dict = Field(default_factory=dict)
    instrument_manifest: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    retrieved_records: list[str] = Field(default_factory=list)
    alternatives: list[ProcessTopology] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ENGINE — Council Messages (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------


class CouncilMessage(BaseModel):
    agent: str
    status: str  # "ACCEPT" / "WARNING" / "REJECT"
    field: str = ""
    value: str = ""
    concern: str = ""
    revision_required: bool = False
    suggested_revision: Optional[str] = None


# ---------------------------------------------------------------------------
# ENGINE — Deliberation Log (new: LLM-powered multi-agent council)
# ---------------------------------------------------------------------------


class FieldProposal(BaseModel):
    """A single concrete, machine-applicable design change proposed by an agent."""
    field: str = ""                       # FlowProposal field name, e.g. "residence_time_min"
    value: str = ""                       # Target value as string, e.g. "15.0"
    reason: str = ""                      # Why this change is needed


class AgentDeliberation(BaseModel):
    """One agent's contribution in a single deliberation round.

    Agents READ the DesignCalculator output (authoritative physics) and
    INTERPRET it — they do not re-derive Re, ΔP, Da, etc.  Their role
    is to assess whether the design is adequate in their domain and
    propose concrete field-level changes when it is not.
    """
    agent: str = ""                      # "KineticsSpecialist", etc.
    agent_display_name: str = ""         # "Dr. Kinetics", etc.
    round: int = 1
    chain_of_thought: str = ""           # Full reasoning (shown to user)
    values_referenced: list[str] = Field(default_factory=list)  # Calculator values cited
    findings: list[str] = Field(default_factory=list)       # Bullet-point findings
    proposals: list[FieldProposal] = Field(default_factory=list)  # Structured changes
    concerns: list[str] = Field(default_factory=list)       # Issues raised
    status: str = "ACCEPT"               # "ACCEPT" / "WARNING" / "REVISE"
    had_error: bool = False              # True if agent LLM call failed — blocks convergence
    references_to_agents: list[str] = Field(default_factory=list)
    rules_cited: list[str] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)  # [{"tool": name, "input": {...}, "result": {...}}, ...]


class SanityCheckResult(BaseModel):
    """Central orchestrator's cross-agent consistency check."""
    round: int = 1
    consistent: bool = True
    chain_of_thought: str = ""
    conflicts_found: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)
    # Only simple numeric/string FlowProposal fields — no lists, no nested objects
    final_changes: dict[str, str] = Field(default_factory=dict)


class DeliberationLog(BaseModel):
    """Complete record of the multi-agent deliberation process."""
    rounds: list[list[AgentDeliberation]] = Field(default_factory=list)
    sanity_checks: list[SanityCheckResult] = Field(default_factory=list)
    total_rounds: int = 0
    consensus_reached: bool = False
    # Cumulative record of all field changes applied across all rounds
    all_changes_applied: dict[str, str] = Field(default_factory=dict)
    summary: str = ""                    # Human-readable summary of deliberation
    trade_off_summary: str = ""   # Skeptic's cross-pick comparative narrative
    trade_off_matrix: str = ""    # Pre-computed surviving-picks comparison table (Chief input)


class DesignCandidate(BaseModel):
    proposal: FlowProposal
    chemistry_plan: Optional[ChemistryPlan] = None
    council_messages: list[CouncilMessage] = Field(default_factory=list)
    council_rounds: int = 0
    safety_report: dict = Field(default_factory=dict)
    unit_operations: list[str] = Field(default_factory=list)
    pid_description: str = ""
    human_explanation: str = ""
    deliberation_log: Optional[DeliberationLog] = None
