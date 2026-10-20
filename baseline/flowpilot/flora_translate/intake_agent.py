"""Standardized FlowPilot intake before batch-to-flow design.

The intake layer turns a chemist's free-form protocol plus follow-up answers
into a frozen ``DesignInputPackage``. Question selection is deterministic and
uses stable IDs from the fixed question bank so benchmarks and GUI sessions are
reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import flora_translate.config as cfg
from flora_translate.config import PROMPTS_DIR
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.executable_artifacts import transformation_family
from flora_translate.schemas import (
    ChemistryPlan,
    DesignInputPackage,
    IntakeAnswer,
    IntakeQuestion,
    StreamLogic,
)

logger = logging.getLogger("flora.intake")


QUESTION_BANK: dict[str, IntakeQuestion] = {
    "Q-BATCH-001": IntakeQuestion(
        question_id="Q-BATCH-001",
        section="batch_protocol",
        question=(
            "Please provide or confirm the batch reaction protocol, including "
            "substrate, product, solvent, concentration, scale, reagents, "
            "equivalents, catalyst/additives, gas atmosphere, temperature, "
            "time, irradiation/light source if any, and reported yield."
        ),
        expected_format="Free-text protocol or structured JSON.",
        required=True,
        why_needed="The design pipeline needs protocol facts before translation.",
        allow_unavailable=False,
    ),
    "Q-OBJ-001": IntakeQuestion(
        question_id="Q-OBJ-001",
        section="objective",
        question=(
            "What is the main goal for the first flow design? Examples: "
            "first safe screening design, reproduce batch yield, improve "
            "throughput, maximize conversion, minimize residence time, or "
            "test a specific hypothesis."
        ),
        expected_format="Short text objective.",
        required=True,
        why_needed="The council ranks candidate designs against this objective.",
        allow_unavailable=False,
    ),
    "Q-CHEM-001": IntakeQuestion(
        question_id="Q-CHEM-001",
        section="chemistry_identity",
        question=(
            "Please confirm the intended reaction transformation. State the "
            "transformation family and, where known, the substrate, product, "
            "bond formed, and bond broken."
        ),
        expected_format=(
            "Short text such as 'hydrogenolysis/debenzylation of A to B; C-N "
            "protecting-group bond broken'."
        ),
        required=True,
        why_needed=(
            "A model-only chemistry identity cannot authorize an executable design."
        ),
        allow_unavailable=False,
    ),
    "Q-HIST-001": IntakeQuestion(
        question_id="Q-HIST-001",
        section="historical_data",
        question=(
            "Do you have historical experimental data or previous flow attempts "
            "for this reaction? If yes, provide a table with conditions and "
            "results."
        ),
        expected_format=(
            "Table/CSV/text with entry, T, concentration, pressure, flows, "
            "residence times, reactor volume, tubing ID, product/yield/"
            "conversion/selectivity, and notes; or mark unavailable."
        ),
        required=True,
        why_needed="Measured evidence has highest design authority.",
    ),
    "Q-INV-001": IntakeQuestion(
        question_id="Q-INV-001",
        section="inventory",
        question=(
            "Do you have inventory constraints that the design must use? "
            "Provide available reactor volumes, tubing IDs/materials, pumps, "
            "pressure/BPR options, gas MFC limits, light sources, and "
            "temperature ranges."
        ),
        expected_format="LabInventory JSON preferred; free text accepted; or mark unavailable.",
        required=True,
        why_needed="Hard hardware constraints must be enforced before final design.",
    ),
    "Q-CONSTR-001": IntakeQuestion(
        question_id="Q-CONSTR-001",
        section="operating_limits",
        question=(
            "Are there operating limits that must not be exceeded? Examples: "
            "max pressure, max temperature, max/min flow rate, max residence "
            "time, forbidden solvents, gas safety limits, oxygen limits, or "
            "material compatibility limits."
        ),
        expected_format="Free text or JSON; or mark unavailable.",
        required=True,
        why_needed="Safety and operating limits constrain feasible candidates.",
    ),
    "Q-HYP-001": IntakeQuestion(
        question_id="Q-HYP-001",
        section="hypotheses",
        question=(
            "Do you have chemical or engineering hypotheses the agent should "
            "consider? Examples: oxygen mass-transfer limitation, photon "
            "limitation, slow kinetics, degradation, gas-liquid ratio, "
            "residence-time sensitivity, or light intensity importance."
        ),
        expected_format="Bullet list/free text; or mark unavailable.",
        required=True,
        why_needed="Hypotheses guide experiment design but are not treated as facts.",
    ),
    "Q-PREF-001": IntakeQuestion(
        question_id="Q-PREF-001",
        section="output_preferences",
        question=(
            "What output do you want? Examples: one best design, short/"
            "intermediate/long residence-time screen, inventory-constrained "
            "design only, benchmark comparison, or design with calculations "
            "and uncertainty."
        ),
        expected_format="Short text preference; optional.",
        required=False,
        why_needed="Output preferences tune presentation and candidate selection.",
    ),
    "Q-GAS-001": IntakeQuestion(
        question_id="Q-GAS-001",
        section="gas_identity",
        question=(
            "Which reagent-gas feed will be used? State the physical feed gas "
            "and active reagent mole fraction, for example pure O2 (1.00), "
            "air (0.21 O2), or H2 (1.00)."
        ),
        expected_format="Gas name and active mole fraction, for example: O2, 1.00.",
        required=True,
        why_needed=(
            "The physical feed composition is required to convert reagent "
            "equivalents into an inlet/STP MFC setpoint."
        ),
        origin="conditional",
        trigger_id="gas_liquid_reagent_detected",
        target_path="engineering_requirements.gas",
        decision_impact="Sets the gas identity and reagent mole fraction used by the MFC calculation.",
        allow_unavailable=False,
    ),
    "Q-GAS-002": IntakeQuestion(
        question_id="Q-GAS-002",
        section="gas_stoichiometry",
        question=(
            "How many molar equivalents of active reagent gas should be delivered "
            "relative to the limiting substrate? FlowPilot will calculate this "
            "from the inlet/STP MFC flow, not the compressed in-channel flow."
        ),
        expected_format="Positive number in equivalents, for example: 2.0 equiv.",
        required=True,
        why_needed=(
            "A target equivalent is required for stoichiometric inlet/STP gas-flow closure."
        ),
        origin="conditional",
        trigger_id="gas_liquid_reagent_detected_without_explicit_equivalents",
        target_path="engineering_requirements.gas.target_equiv_inlet_stp",
        decision_impact="Directly changes the inlet/STP MFC setpoint and both reported apparent residence times.",
        allow_unavailable=True,
    ),
    "Q-GAS-003": IntakeQuestion(
        question_id="Q-GAS-003",
        section="gas_introduction_stage",
        question=(
            "At which reaction stage is the reagent gas first introduced? "
            "Identify the stage number or describe the stage operation."
        ),
        expected_format="Stage number and short description, for example: Stage 2, aerobic oxidation.",
        required=True,
        why_needed="The gas feed must be attached to exactly one stage in a multistep topology.",
        origin="conditional",
        trigger_id="multistep_gas_liquid_reaction",
        target_path="engineering_requirements.gas.introduction_stage",
        decision_impact="Determines the mixer, MFC connection, stage temperature, and stage residence-time calculation.",
        allow_unavailable=False,
    ),
    "Q-PHOTO-001": IntakeQuestion(
        question_id="Q-PHOTO-001",
        section="photochemistry",
        question=(
            "What irradiation wavelength should the flow design target? If the "
            "batch wavelength is unknown, provide the catalyst absorption target "
            "or mark it unavailable so inventory matching can remain provisional."
        ),
        expected_format="Wavelength in nm, for example: 450 nm; or mark unavailable.",
        required=True,
        why_needed="Wavelength is required to match the chemistry to an available light source.",
        origin="conditional",
        trigger_id="photochemical_reaction_without_explicit_wavelength",
        target_path="engineering_requirements.photochemistry.wavelength_nm",
        decision_impact="Restricts compatible photoreactors and light sources.",
        allow_unavailable=True,
    ),
    "Q-MULTI-001": IntakeQuestion(
        question_id="Q-MULTI-001",
        section="multistep_definition",
        question=(
            "Confirm the ordered reaction stages and the operation separating "
            "them. Include each stage's transformation, temperature, time, new "
            "feeds, and any interstage mixing, quench, filtration, or solvent change."
        ),
        expected_format="Numbered stage list with conditions and interstage operation.",
        required=True,
        why_needed="A multistep topology cannot be authorized from an ambiguous stage boundary.",
        origin="conditional",
        trigger_id="multistep_reaction_detected",
        target_path="engineering_requirements.multistep.stage_definition",
        decision_impact="Controls stage count, feed introduction, equipment topology, and per-stage timing.",
        allow_unavailable=False,
    ),
}

REQUIRED_READINESS_IDS = {
    "Q-BATCH-001",
    "Q-OBJ-001",
    "Q-CHEM-001",
    "Q-HIST-001",
    "Q-INV-001",
    "Q-CONSTR-001",
    "Q-HYP-001",
}

QUESTION_BANK_VERSION = "flowpilot_questions_v1.1"
CORE_QUESTION_ORDER = (
    "Q-BATCH-001",
    "Q-OBJ-001",
    "Q-CHEM-001",
    "Q-HIST-001",
    "Q-INV-001",
    "Q-CONSTR-001",
    "Q-HYP-001",
)
CONDITIONAL_QUESTION_ORDER = (
    "Q-GAS-001",
    "Q-GAS-002",
    "Q-GAS-003",
    "Q-PHOTO-001",
    "Q-MULTI-001",
)


def get_question(question_id: str) -> IntakeQuestion:
    return QUESTION_BANK[question_id]


def intake_context_block(package: DesignInputPackage | dict | None) -> str:
    """Format the frozen intake package for upstream/council prompts."""

    if not package:
        return "## FlowPilot Intake Context\nNo standardized intake package was provided."
    pkg = _coerce_package(package)
    # The complete inventory remains frozen on the package for deterministic
    # allocation. Prompts receive only capability-defining fields so the same
    # profile is not repeated in Q-INV-001 and the answer log.
    inventory_context = _compact_inventory_for_prompt(pkg.inventory_constraints)
    answer_log = [
        {
            "question_id": answer.question_id,
            "status": answer.status,
            "source": answer.source,
            "answer": (
                "See hard constraints and operating limits above."
                if answer.question_id in {"Q-INV-001", "Q-CONSTR-001"}
                else answer.answer
            ),
        }
        for answer in pkg.answer_map().values()
    ]
    return (
        "## FlowPilot Intake Context - authority labeled\n"
        "Authority order: measured evidence > hard constraints > protocol facts > "
        "chemist hypotheses > model inference.\n"
        f"Schema: {pkg.schema_version}\n"
        f"Ready for design: {pkg.ready_for_design}\n\n"
        "### Design objective\n"
        f"{pkg.objective or '(not provided)'}\n\n"
        f"Screening priority selection: {pkg.screening_priority}\n\n"
        "### Protocol facts\n"
        f"{json.dumps(pkg.extracted_batch_fields or {}, indent=2, default=str)}\n\n"
        "### Frozen chemistry identity\n"
        f"{json.dumps(pkg.chemistry_identity_confirmation or {}, indent=2, default=str)}\n\n"
        "### Measured evidence / historical data\n"
        f"{_format_any(pkg.historical_data)}\n\n"
        "### Hard constraints / inventory\n"
        f"{_format_any(inventory_context)}\n\n"
        "### Safety and operating limits\n"
        f"{_format_any(pkg.operating_limits)}\n\n"
        "### Deterministic engineering requirements\n"
        f"{json.dumps(pkg.engineering_requirements, indent=2, default=str)}\n\n"
        "### Chemist hypotheses\n"
        f"{json.dumps(pkg.hypotheses, indent=2, default=str)}\n\n"
        "### Intake questions and answers\n"
        "Current effective answers only (latest answer per question ID). Superseded answers remain in the saved package history and are not active design instructions.\n"
        f"{json.dumps(answer_log, indent=2, default=str)}"
    )


_PROMPT_EQUIPMENT_FIELDS = (
    "equipment_id", "name", "quantity", "service_status", "type", "material",
    "volume_mL", "ID_mm", "configuration", "component_volumes_mL", "gas",
    "setpoints_bar", "min_pressure_bar", "max_pressure_bar", "min_temperature_C",
    "max_temperature_C", "allowed_temperatures_C", "min_flow_rate_mL_min", "flow_rate_increment_mL_min",
    "max_flow_rate_mL_min", "min_flow_sccm", "max_flow_sccm", "wavelength_nm",
    "power_W", "intensity_mW_cm2", "compatible_reactor",
    "platform_id", "resource_requirements", "excluded_chemicals", "module_id",
    "module_name", "max_reactor_volume_mL", "compatible_pump_platforms",
    "photoreactor_module_ids",
    "required_outlet_accessory_type", "cracking_pressure_bar",
)


def _compact_inventory_for_prompt(value: Any) -> Any:
    """Keep design limits while removing notes and repeated profile metadata."""

    if not isinstance(value, dict):
        return value
    compact: dict[str, Any] = {}
    for category, items in value.items():
        if isinstance(items, list):
            compact[category] = [
                {
                    key: item[key]
                    for key in _PROMPT_EQUIPMENT_FIELDS
                    if key in item and item[key] not in (None, "", [], {})
                }
                if isinstance(item, dict)
                else item
                for item in items
            ]
        elif items not in (None, "", [], {}):
            compact[category] = items
    return compact


def batch_input_from_package(package: DesignInputPackage | dict) -> str | dict:
    """Return the best batch input payload for the existing parser."""

    pkg = _coerce_package(package)
    fields = dict(pkg.extracted_batch_fields or {})
    if fields:
        fields.setdefault("raw_text", pkg.raw_protocol)
        if pkg.raw_protocol and not fields.get("reaction_description"):
            fields["reaction_description"] = pkg.raw_protocol[:500]
        return fields
    return pkg.raw_protocol


def historical_text_from_package(package: DesignInputPackage | dict | None) -> str:
    """Return measured feedback text for closed-loop extraction."""

    if not package:
        return ""
    pkg = _coerce_package(package)
    return _format_any(pkg.historical_data)


def apply_intake_requirements_to_chemistry_plan(
    chemistry_plan: ChemistryPlan,
    package: DesignInputPackage | dict | None,
) -> tuple[ChemistryPlan, list[dict[str, Any]]]:
    """Apply chemist-confirmed conditional answers as structured authority."""

    if not package:
        return chemistry_plan, []
    pkg = _coerce_package(package)
    plan = chemistry_plan.model_copy(deep=True)
    requirements = pkg.engineering_requirements or {}
    decisions: list[dict[str, Any]] = []

    gas = requirements.get("gas") if isinstance(requirements, dict) else None
    if isinstance(gas, dict) and gas.get("species"):
        plan.scientific_context["gas_delivery"] = dict(gas)
        species = str(gas["species"])
        reagent_fraction = _positive_number(gas.get("reagent_mole_fraction")) or 1.0
        delegated = gas.get("feed_selection_mode") == "agent_to_propose"
        target_equiv = None if delegated else _positive_number(gas.get("target_equiv_inlet_stp")) or 1.0
        target_stage = _positive_integer(gas.get("introduction_stage"))
        if target_stage is None:
            target_stage = max(
                (int(stage.stage_number or 1) for stage in plan.stages or []),
                default=1,
            )
        gas_feeds = [feed for feed in plan.stream_logic or [] if feed.phase == "gas"]
        if gas.get("minimum_equiv_inlet_stp") and not delegated:
            target_equiv = max([target_equiv, *[feed.molar_equiv or 0 for feed in gas_feeds]])
        if not gas_feeds:
            gas_feeds = [
                StreamLogic(
                    stream_label="G",
                    reagents=[species],
                    phase="gas",
                    introduction_stage=target_stage,
                    separate_feed_required=True,
                    accepted_requirement=True,
                )
            ]
            plan.stream_logic.append(gas_feeds[0])
        for feed in gas_feeds:
            feed.reagents = [species]
            feed.phase = "gas"
            feed.gas_reagent_mole_fraction = reagent_fraction
            if not delegated:
                feed.molar_equiv = target_equiv
            feed.molar_equiv_basis = "model_proposed_inlet_stp_requires_justification" if delegated else "chemist_confirmed_intake_inlet_stp"
            feed.introduction_stage = target_stage
            feed.requirement_authority = "model_inference" if delegated else "hard_constraint"
            feed.source_evidence = [
                "Structured standardized-intake gas requirement."
            ]
            feed.accepted_requirement = True
            feed.separate_feed_required = True
            feed.reasoning = (
                "Chemist delegated the inlet/STP equivalent selection. The proposed equivalent is not measured or chemist-confirmed and requires scientific justification. "
                + (feed.reasoning or "")
            ) if delegated else (
                f"Chemist-confirmed {species} feed at {target_equiv:g} equiv; "
                "equivalents and MFC flow are defined at inlet/STP."
            )
        for stage in plan.stages or []:
            retained = [
                feed
                for feed in stage.feed_streams or []
                if feed.phase != "gas"
                or feed.stream_label not in {item.stream_label for item in gas_feeds}
            ]
            if int(stage.stage_number or 1) == target_stage:
                retained.extend(item.model_copy(deep=True) for item in gas_feeds)
            stage.feed_streams = retained
        plan.o2_is_reagent = species.lower() in {"o2", "oxygen", "air"}
        decisions.append(
            {
                "decision": "apply_intake_gas_requirement",
                "species": species,
                "reagent_mole_fraction": reagent_fraction,
                "target_equiv_inlet_stp": target_equiv,
                "feed_selection_mode": gas.get("feed_selection_mode", "fixed"),
                "introduction_stage": target_stage,
                "authority": "chemist_answer_or_protocol_fact",
            }
        )

    photo = requirements.get("photochemistry") if isinstance(requirements, dict) else None
    if isinstance(photo, dict):
        wavelength = _positive_number(photo.get("wavelength_nm"))
        if wavelength is not None:
            plan.recommended_wavelength_nm = wavelength
            plan.wavelength_reasoning = (
                "Target wavelength fixed by the standardized intake package."
            )
            light_stages = [stage for stage in plan.stages or [] if stage.requires_light]
            if not light_stages and plan.stages:
                light_stages = [plan.stages[0]]
                light_stages[0].requires_light = True
            for stage in light_stages:
                stage.wavelength_nm = wavelength
            decisions.append(
                {
                    "decision": "apply_intake_wavelength",
                    "wavelength_nm": wavelength,
                    "authority": photo.get("wavelength_source") or "protocol_fact",
                }
            )

    return plan, decisions


def _active_domains(protocol: str) -> list[str]:
    text = str(protocol or "")
    domains: list[str] = []
    if _protocol_has_reagent_gas(text):
        domains.append("gas_liquid")
    if re.search(r"(?i)\b(?:photo(?:redox|chemical|chemistry)?|irradiat\w*|LED)\b|\d{3,4}\s*nm", text):
        domains.append("photochemistry")
    if re.search(r"(?i)\b(?:two[- ]stage|multi[- ]step|step\s*[12]|stage\s*[12]|followed by|sequential(?:ly)?)\b", text):
        domains.append("multistep")
    return domains


def _conditional_question_ids(protocol: str) -> list[str]:
    domains = set(_active_domains(protocol))
    facts = _protocol_engineering_requirements(protocol)
    selected: list[str] = []
    gas = facts.get("gas") or {}
    if "gas_liquid" in domains:
        if not gas.get("species"):
            selected.append("Q-GAS-001")
        if not _positive_number(gas.get("target_equiv_inlet_stp")):
            selected.append("Q-GAS-002")
        if "multistep" in domains and not _positive_integer(gas.get("introduction_stage")):
            selected.append("Q-GAS-003")
    photo = facts.get("photochemistry") or {}
    if "photochemistry" in domains and not _positive_number(photo.get("wavelength_nm")):
        selected.append("Q-PHOTO-001")
    multistep = facts.get("multistep") or {}
    if "multistep" in domains and not multistep.get("stage_definition"):
        selected.append("Q-MULTI-001")
    return [qid for qid in CONDITIONAL_QUESTION_ORDER if qid in selected]


def _protocol_engineering_requirements(protocol: str) -> dict[str, Any]:
    text = str(protocol or "")
    requirements: dict[str, Any] = {}
    if _protocol_has_reagent_gas(text):
        gas: dict[str, Any] = {
            "calculation_basis": "inlet_stp",
            "stp_temperature_K": 273.15,
            "stp_pressure_bar": 1.01325,
        }
        identity = _gas_identity_from_text(text)
        if identity:
            species, fraction = identity
            gas.update(
                {
                    "species": species,
                    "reagent_mole_fraction": fraction,
                    "identity_source": "protocol_fact",
                }
            )
        equiv = _protocol_gas_equiv(text)
        if equiv is not None:
            gas["target_equiv_inlet_stp"] = equiv
            gas["equiv_source"] = "protocol_fact"
        stage = _protocol_gas_stage(text)
        if stage is not None:
            gas["introduction_stage"] = stage
            gas["introduction_stage_source"] = "protocol_fact"
        requirements["gas"] = gas

    wavelength = re.search(r"(?i)(\d{3,4}(?:\.\d+)?)\s*nm\b", text)
    if wavelength:
        requirements["photochemistry"] = {
            "wavelength_nm": float(wavelength.group(1)),
            "wavelength_source": "protocol_fact",
        }

    if "multistep" in _active_domains(text) and re.search(
        r"(?is)\b(?:step|stage)\s*1\b.+\b(?:step|stage)\s*2\b",
        text,
    ):
        requirements["multistep"] = {
            "stage_definition": text,
            "stage_definition_source": "protocol_fact",
        }
    return requirements


def _protocol_has_reagent_gas(protocol: str) -> bool:
    text = str(protocol or "").lower().replace("₂", "2").replace("₃", "3")
    for negative in (
        "oxygen-sensitive", "oxygen sensitive", "o2-sensitive", "o2 sensitive",
        "oxygen-free", "oxygen free", "o2-free", "deoxygenated",
    ):
        text = text.replace(negative, "")
    gas = r"(?:oxygen|o2|air|hydrogen|h2|carbon dioxide|co2|carbon monoxide|syngas|ozone|o3|chlorine|cl2|ammonia|nh3|hydrogen chloride|hcl gas|sulfur dioxide|so2)"
    context = r"(?:equiv|atm|bar|balloon|bubbl\w*|sparg\w*|feed|mfc|oxidant|reagent|introduced|exposed|under)"
    return bool(
        re.search(rf"\b{gas}\b[^\n]{{0,100}}\b{context}\b", text)
        or re.search(rf"\b{context}\b[^\n]{{0,100}}\b{gas}\b", text)
        or re.search(r"\b(?:hydrogenation|carbonylation|aerobic oxidation)\b", text)
    )


def _gas_identity_from_text(text: str) -> tuple[str, float] | None:
    lower = str(text or "").lower().replace("₂", "2").replace("₃", "3")
    identities = (
        ("air", 0.21, r"\bair\b"),
        ("O2", 1.0, r"\b(?:oxygen|o2)\b"),
        ("H2", 1.0, r"\b(?:hydrogen|h2)\b"),
        ("CO2", 1.0, r"\b(?:carbon dioxide|co2)\b"),
        ("CO", 1.0, r"\bcarbon monoxide\b"),
        ("syngas", 0.5, r"\bsyngas\b"),
        ("O3", 1.0, r"\b(?:ozone|o3)\b"),
        ("Cl2", 1.0, r"\b(?:chlorine|cl2)\b"),
        ("NH3", 1.0, r"\b(?:ammonia|nh3)\b"),
        ("HCl", 1.0, r"\b(?:hydrogen chloride|hcl gas)\b"),
        ("SO2", 1.0, r"\b(?:sulfur dioxide|so2)\b"),
    )
    for species, fraction, pattern in identities:
        if re.search(pattern, lower):
            return species, fraction
    return None


def _protocol_gas_equiv(text: str) -> float | None:
    normalized = str(text or "").replace("₂", "2").replace("₃", "3")
    gas = r"(?:oxygen|o2|air|hydrogen|h2|carbon dioxide|co2|carbon monoxide|syngas|ozone|o3|chlorine|cl2|ammonia|nh3|hydrogen chloride|hcl gas|sulfur dioxide|so2)"
    for pattern in (
        rf"\b{gas}\b[^\n]{{0,100}}?(\d+(?:\.\d+)?)\s*(?:mol\s*)?equiv",
        rf"(\d+(?:\.\d+)?)\s*(?:mol\s*)?equiv[^\n]{{0,100}}?\b{gas}\b",
    ):
        match = re.search(pattern, normalized, re.I)
        if match:
            value = _positive_number(match.group(1))
            if value is not None:
                return value
    return None


def _protocol_gas_stage(text: str) -> int | None:
    normalized = str(text or "").replace("₂", "2").replace("₃", "3")
    gas_pattern = r"\b(?:oxygen|o2|air|hydrogen|h2|co2|carbon monoxide|syngas|ozone|o3|chlorine|cl2|ammonia|nh3|hcl gas|so2)\b"
    for match in re.finditer(r"(?i)\b(?:step|stage)\s*(\d+)\b([^\n.]{0,240})", normalized):
        segment = match.group(2).lower()
        for absent_gas_phrase in (
            "oxygen-free", "oxygen free", "o2-free", "o2 free",
            "oxygen-sensitive", "oxygen sensitive", "o2-sensitive",
            "deoxygenated", "without oxygen", "absence of oxygen",
        ):
            segment = segment.replace(absent_gas_phrase, "")
        if re.search(gas_pattern, segment, re.I):
            return int(match.group(1))
    return None


def _parse_gas_identity_and_fraction(value: Any) -> dict[str, Any] | None:
    text = str(value or "").replace("₂", "2").replace("₃", "3")
    # Excluded alternatives in an answer are not the requested feed identity.
    gas_names = (r"air|oxygen|o2|hydrogen chloride|hydrogen|h2|carbon dioxide|co2|"
                 r"carbon monoxide|syngas|ozone|o3|chlorine|cl2|ammonia|nh3|"
                 r"hcl gas|sulfur dioxide|so2")
    text = re.sub(
        rf"\b(?:do\s+not|don't|must\s+not|never|not|without|rather\s+than|instead\s+of)"
        rf"\s+(?:(?:use|using|feed|introduce|supply|select|choose)\s+)?"
        rf"(?:(?:pure|ambient|compressed)\s+)?(?:{gas_names})\b",
        "", text, flags=re.I,
    )
    identity = _gas_identity_from_text(text)
    if not identity:
        return None
    species, default_fraction = identity
    percent = re.search(r"(\d+(?:\.\d+)?)\s*(?:mol\s*)?%", text, re.I)
    decimal = re.search(r"(?:fraction|mole fraction|purity)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, re.I)
    if not decimal:
        decimal = re.search(
            r"(0(?:\.\d+)?|1(?:\.0+)?)\s*(?:mole\s*)?(?:fraction|purity)",
            text,
            re.I,
        )
    if percent:
        fraction = float(percent.group(1)) / 100.0
    elif decimal:
        fraction = float(decimal.group(1))
    else:
        trailing = re.search(r"[,;]\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$", text)
        fraction = float(trailing.group(1)) if trailing else default_fraction
    if not 0 < fraction <= 1:
        return None
    return {"species": species, "reagent_mole_fraction": fraction}


def _positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        match = re.search(r"(?:^|[^\d.])(\d+(?:\.\d+)?)", str(value or ""))
        if not match:
            return None
        number = float(match.group(1))
    return number if number > 0 else None


def _positive_integer(value: Any) -> int | None:
    number = _positive_number(value)
    return int(number) if number is not None and float(number).is_integer() else None


def _gas_introduction_stage(value: Any) -> int | None:
    text = str(value or "")
    exclusions = set(re.findall(r"stage\s*(\d+)\s+(?:must\s+)?(?:remain|be|is|stays?)\s+oxygen[- ]free", text, re.I))
    if exclusions:
        positive = set()
        for sentence in re.split(r"[.;\n]", text):
            if re.search(r"oxygen[- ]free|\bnot\b|\bnever\b", sentence, re.I):
                continue
            positive.update(re.findall(r"stage\s*(\d+)", sentence, re.I))
        if len(positive) == 1 and positive.isdisjoint(exclusions):
            return int(next(iter(positive)))
        return None
    if re.search(r"\b(?:not|never)\b", text, re.I):
        return None
    # A numbered effluent is the source, not the destination of an addition.
    transfer = re.search(r"stage\s*(\d+)\s+effluent\s+enters\b", text, re.I)
    if transfer:
        destination = re.search(r"enters\s+(?:the\s+)?stage\s*(\d+)", text, re.I)
        return int(destination[1]) if destination else int(transfer[1]) + 1
    stages = set(re.findall(r"stage\s*(\d+)", text, re.I))
    if len(stages) > 1 or re.search(r"\b(?:not|never)\b", text, re.I):
        return None
    return _positive_integer(value)


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _question_set_hash(question_ids: list[str]) -> str:
    payload = {
        "question_bank_version": QUESTION_BANK_VERSION,
        "questions": [QUESTION_BANK[qid].model_dump() for qid in question_ids],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _conditional_answer_resolved(
    package: DesignInputPackage,
    question_id: str,
) -> bool:
    requirements = package.engineering_requirements or {}
    gas = requirements.get("gas") or {}
    photo = requirements.get("photochemistry") or {}
    multistep = requirements.get("multistep") or {}
    if question_id == "Q-GAS-001":
        return bool(gas.get("species")) and bool(
            _positive_number(gas.get("reagent_mole_fraction"))
        )
    if question_id == "Q-GAS-002":
        return bool(_positive_number(gas.get("target_equiv_inlet_stp"))) or gas.get("feed_selection_mode") == "agent_to_propose"
    if question_id == "Q-GAS-003":
        return bool(_positive_integer(gas.get("introduction_stage")))
    if question_id == "Q-PHOTO-001":
        return bool(_positive_number(photo.get("wavelength_nm")))
    if question_id == "Q-MULTI-001":
        return bool(str(multistep.get("stage_definition") or "").strip())
    return True


class IntakeAgent:
    """Build and update a standardized DesignInputPackage."""

    def analyze(
        self,
        raw_protocol: str = "",
        *,
        existing_package: DesignInputPackage | dict | None = None,
        answers: list[IntakeAnswer | dict] | None = None,
        use_llm: bool = True,
    ) -> DesignInputPackage:
        # An answer revision must not mutate the prior frozen package through
        # shared nested requirements, inventory or answer objects.
        pkg = _coerce_package(existing_package).model_copy(deep=True) if existing_package else DesignInputPackage()
        raw_protocol = raw_protocol or pkg.raw_protocol or ""
        protocol_changed = bool(pkg.raw_protocol and raw_protocol != pkg.raw_protocol)
        # Answers are valid only for the exact protocol/question set. Reusing
        # them after a protocol edit can falsely close unrelated questions.
        merged_answers = [] if protocol_changed else list(pkg.answers)
        for answer in answers or []:
            merged_answers.append(_coerce_answer(answer))

        extracted = {} if protocol_changed else dict(pkg.extracted_batch_fields or {})
        if raw_protocol and not extracted:
            extracted = self._extract_batch_fields(raw_protocol, use_llm=use_llm)

        protocol_requirements = _protocol_engineering_requirements(raw_protocol)
        existing_requirements = (
            {} if protocol_changed else dict(pkg.engineering_requirements or {})
        )
        engineering_requirements = _merge_nested(
            protocol_requirements,
            existing_requirements,
        )
        active_domains = _active_domains(raw_protocol)
        active_question_ids = [
            *CORE_QUESTION_ORDER,
            *_conditional_question_ids(raw_protocol),
        ]

        updated = DesignInputPackage(
            schema_version="flowpilot_intake_v1.1",
            question_bank_version=QUESTION_BANK_VERSION,
            raw_protocol=raw_protocol,
            extracted_batch_fields=extracted,
            objective=pkg.objective,
            screening_priority=pkg.screening_priority,
            historical_data=pkg.historical_data,
            inventory_constraints=pkg.inventory_constraints,
            hypotheses=list(pkg.hypotheses or []),
            operating_limits=pkg.operating_limits,
            inventory_profile_snapshot=dict(pkg.inventory_profile_snapshot or {}),
            output_preferences=pkg.output_preferences,
            chemistry_identity_confirmation=dict(
                pkg.chemistry_identity_confirmation or {}
            ),
            active_domains=active_domains,
            active_question_ids=active_question_ids,
            engineering_requirements=engineering_requirements,
            question_set_hash=_question_set_hash(active_question_ids),
            question_log=[] if protocol_changed else list(pkg.question_log or []),
            answers=merged_answers,
        )
        updated = self._apply_answers(updated)
        if not updated.chemistry_identity_confirmation:
            family = transformation_family(updated.raw_protocol)
            multistep = bool(
                re.search(
                    r"(?i)\b(two[- ]stage|multi[- ]step|sequential(?:ly)?|followed by)\b",
                    updated.raw_protocol,
                )
            )
            if family != "unknown" and not multistep:
                updated.chemistry_identity_confirmation = {
                    "transformation_family": family,
                    "confirmed": True,
                    "source": "protocol_fact",
                }
        missing = self._missing_required_ids(updated)
        logged = {q.question_id for q in updated.question_log}
        question_log = list(updated.question_log)
        for qid in active_question_ids:
            if qid not in logged:
                question_log.append(QUESTION_BANK[qid])
        updated.question_log = question_log
        updated.missing_question_ids = missing
        from flora_translate.inventory_resolution import review_inventory

        updated.inventory_review = review_inventory(updated)
        updated.ready_for_design = not missing and updated.inventory_review["ready"]
        return updated

    def pending_questions(self, package: DesignInputPackage | dict) -> list[IntakeQuestion]:
        pkg = _coerce_package(package)
        return [QUESTION_BANK[qid] for qid in pkg.missing_question_ids if qid in QUESTION_BANK]

    def _extract_batch_fields(self, raw_protocol: str, *, use_llm: bool) -> dict[str, Any]:
        if use_llm:
            try:
                return self._llm_extract(raw_protocol)
            except Exception as exc:
                logger.warning("Intake LLM extraction failed; using fallback extraction: %s", exc)
        return _fallback_extract(raw_protocol)

    def _llm_extract(self, raw_protocol: str) -> dict[str, Any]:
        system = (PROMPTS_DIR / "intake_system.txt").read_text()
        user_template = (PROMPTS_DIR / "intake_user.txt").read_text()
        result = call_model_text(
            model=cfg.MODEL_INPUT_PARSER,
            api_name="intake_agent",
            max_tokens=1200,
            system=system,
            user_content=user_template.replace("{raw_protocol}", raw_protocol),
        )
        data = _parse_json(result.text)
        fields = data.get("extracted_batch_fields", data)
        if not isinstance(fields, dict):
            return _fallback_extract(raw_protocol)
        fields.setdefault("raw_text", raw_protocol)
        if raw_protocol and not fields.get("reaction_description"):
            fields["reaction_description"] = raw_protocol[:500]
        return fields

    def _apply_answers(self, package: DesignInputPackage) -> DesignInputPackage:
        answer_map = package.answer_map()

        batch_answer = answer_map.get("Q-BATCH-001")
        if batch_answer and batch_answer.status == "answered" and batch_answer.answer:
            package.raw_protocol = str(batch_answer.answer)
            if not package.extracted_batch_fields:
                package.extracted_batch_fields = _fallback_extract(package.raw_protocol)

        obj_answer = answer_map.get("Q-OBJ-001")
        if obj_answer and obj_answer.status == "answered":
            package.objective = str(obj_answer.answer or "").strip()

        chemistry_answer = answer_map.get("Q-CHEM-001")
        if chemistry_answer and chemistry_answer.status == "answered":
            answer_text = str(chemistry_answer.answer or "").strip()
            family = transformation_family(
                " ".join(part for part in (answer_text, package.raw_protocol) if part)
            )
            package.chemistry_identity_confirmation = {
                "transformation_family": family,
                "chemist_description": answer_text,
                "confirmed": bool(answer_text),
                "source": "chemist_confirmed",
            }

        hist_answer = answer_map.get("Q-HIST-001")
        if hist_answer:
            package.historical_data = None if hist_answer.status == "unavailable" else hist_answer.answer

        inv_answer = answer_map.get("Q-INV-001")
        if inv_answer:
            package.inventory_constraints = (
                None if inv_answer.status == "unavailable" else _parse_json_or_text(inv_answer.answer)
            )

        constr_answer = answer_map.get("Q-CONSTR-001")
        if constr_answer:
            package.operating_limits = (
                None if constr_answer.status == "unavailable" else _parse_json_or_text(constr_answer.answer)
            )

        hyp_answer = answer_map.get("Q-HYP-001")
        if hyp_answer:
            package.hypotheses = [] if hyp_answer.status == "unavailable" else _split_hypotheses(hyp_answer.answer)

        pref_answer = answer_map.get("Q-PREF-001")
        if pref_answer and pref_answer.status == "answered":
            package.output_preferences = str(pref_answer.answer or "").strip()

        gas_identity = answer_map.get("Q-GAS-001")
        if gas_identity and gas_identity.status == "answered":
            parsed = _parse_gas_identity_and_fraction(gas_identity.answer)
            if parsed:
                gas = package.engineering_requirements.setdefault("gas", {})
                gas.update(parsed)
                gas.setdefault("calculation_basis", "inlet_stp")
                gas["identity_source"] = "chemist_answer"

        gas_equiv = answer_map.get("Q-GAS-002")
        if gas_equiv:
            gas = package.engineering_requirements.setdefault("gas", {})
            if gas_equiv.status == "unavailable":
                gas.pop("feed_selection_mode", None)
                gas["target_equiv_inlet_stp"] = 1.0
                gas["equiv_source"] = "deterministic_screening_assumption"
            else:
                delegated = re.search(r"\bno fixed value\b", str(gas_equiv.answer), re.I)
                value = _positive_number(gas_equiv.answer)
                if delegated:
                    gas.pop("target_equiv_inlet_stp", None)
                    gas["feed_selection_mode"] = "agent_to_propose"
                    gas["equiv_source"] = "chemist_delegated_with_justification_required"
                elif value is not None:
                    gas.pop("feed_selection_mode", None)
                    gas["target_equiv_inlet_stp"] = value
                    gas["equiv_source"] = "chemist_answer"
                    if re.search(r"at least|minimum|no less than", str(gas_equiv.answer), re.I):
                        gas["minimum_equiv_inlet_stp"] = value
            gas.setdefault("calculation_basis", "inlet_stp")

        gas_stage = answer_map.get("Q-GAS-003")
        if gas_stage and gas_stage.status == "answered":
            stage = _gas_introduction_stage(gas_stage.answer)
            if stage is not None:
                gas = package.engineering_requirements.setdefault("gas", {})
                gas["introduction_stage"] = stage
                gas["introduction_stage_source"] = "chemist_answer"
            else:
                gas = package.engineering_requirements.setdefault("gas", {})
                gas.pop("introduction_stage", None)
                gas.pop("introduction_stage_source", None)

        photo = answer_map.get("Q-PHOTO-001")
        if photo and photo.status == "answered":
            wavelength = _positive_number(photo.answer)
            if wavelength is not None:
                requirements = package.engineering_requirements.setdefault(
                    "photochemistry", {}
                )
                requirements["wavelength_nm"] = wavelength
                requirements["wavelength_source"] = "chemist_answer"

        multistep = answer_map.get("Q-MULTI-001")
        if multistep and multistep.status == "answered":
            definition = str(multistep.answer or "").strip()
            if definition:
                requirements = package.engineering_requirements.setdefault(
                    "multistep", {}
                )
                requirements["stage_definition"] = definition
                requirements["stage_definition_source"] = "chemist_answer"

        return package

    def _missing_required_ids(self, package: DesignInputPackage) -> list[str]:
        answer_map = package.answer_map()
        missing: list[str] = []

        if not _has_protocol(package):
            missing.append("Q-BATCH-001")

        if not package.objective.strip():
            missing.append("Q-OBJ-001")

        identity = package.chemistry_identity_confirmation or {}
        if not identity.get("confirmed"):
            missing.append("Q-CHEM-001")

        for qid, attr in (
            ("Q-HIST-001", "historical_data"),
            ("Q-INV-001", "inventory_constraints"),
            ("Q-CONSTR-001", "operating_limits"),
            ("Q-HYP-001", "hypotheses"),
        ):
            answer = answer_map.get(qid)
            value = getattr(package, attr)
            if answer and answer.status in {"answered", "unavailable"}:
                continue
            if _has_value(value):
                continue
            missing.append(qid)

        for qid in package.active_question_ids:
            if qid not in CONDITIONAL_QUESTION_ORDER:
                continue
            answer = answer_map.get(qid)
            question = QUESTION_BANK[qid]
            if answer and answer.status == "unavailable" and question.allow_unavailable:
                continue
            if not _conditional_answer_resolved(package, qid):
                missing.append(qid)

        return missing


def _coerce_package(package: DesignInputPackage | dict) -> DesignInputPackage:
    if isinstance(package, DesignInputPackage):
        return package
    return DesignInputPackage.model_validate(package)


def _coerce_answer(answer: IntakeAnswer | dict) -> IntakeAnswer:
    if isinstance(answer, IntakeAnswer):
        return answer
    return IntakeAnswer.model_validate(answer)


def _has_protocol(package: DesignInputPackage) -> bool:
    if package.raw_protocol.strip():
        return True
    desc = str((package.extracted_batch_fields or {}).get("reaction_description") or "")
    return bool(desc.strip())


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _fallback_extract(raw_protocol: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "reaction_description": raw_protocol[:500],
        "raw_text": raw_protocol,
    }
    temp = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|C\b)", raw_protocol, re.I)
    if temp:
        fields["temperature_C"] = float(temp.group(1))
    conc = re.search(r"(\d+(?:\.\d+)?)\s*M\b", raw_protocol, re.I)
    if conc:
        fields["concentration_M"] = float(conc.group(1))
    wave = re.search(r"(\d+(?:\.\d+)?)\s*nm\b", raw_protocol, re.I)
    if wave:
        fields["wavelength_nm"] = float(wave.group(1))
    time_h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", raw_protocol, re.I)
    if time_h:
        fields["reaction_time_h"] = float(time_h.group(1))
    yield_pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:isolated\s*)?yield", raw_protocol, re.I)
    if yield_pct:
        fields["yield_pct"] = float(yield_pct.group(1))
    photocatalyst = re.search(
        r"\bphotocatalyst\s*:\s*(.+?)(?=,\s*\d+(?:\.\d+)?\s*mol\s*%|[.;\n])",
        raw_protocol,
        re.I,
    )
    if photocatalyst:
        fields["photocatalyst"] = photocatalyst.group(1).strip()
    loading = re.search(
        r"\bphotocatalyst\s*:.{0,250}?,\s*(\d+(?:\.\d+)?)\s*mol\s*%",
        raw_protocol,
        re.I,
    )
    if loading:
        fields["catalyst_loading_mol_pct"] = float(loading.group(1))
    solvent = re.search(
        r"\bsolvent\s*:\s*(.+?)(?=\.\s*(?:Concentration|Step|Stage)\b|[;\n]|$)",
        raw_protocol,
        re.I,
    )
    if solvent:
        fields["solvent"] = solvent.group(1).strip().rstrip(".")
    if re.search(r"\b(?:under|maintain under)\s+argon\b", raw_protocol, re.I):
        fields["atmosphere"] = "argon"
    if wave:
        fields["light_source"] = f"{float(wave.group(1)):g} nm light"
    return fields


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _parse_json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _split_hypotheses(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    pieces = re.split(r"(?:\n|;)+", text)
    return [p.strip(" -\t") for p in pieces if p.strip(" -\t")]


def _format_any(value: Any) -> str:
    if value is None:
        return "(unavailable)"
    if isinstance(value, str):
        return value.strip() or "(unavailable)"
    return json.dumps(value, indent=2, default=str)
