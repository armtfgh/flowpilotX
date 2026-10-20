"""FLORA-Translate — Main entry point.

Pipeline:
  BatchRecord
    → ChemistryAgent          (Layer 1 — pure chemistry analysis)
    → plan-aware Retrieval    (Layer 2 — better semantic search)
    → TranslationLLM          (Layer 2 — proposal grounded in plan + analogies)
    → ENGINE + ChemValidator  (Layer 3 — engineering + chemistry validation)
    → ProcessTopology builder (converts validated proposal → chemistry-aware diagram)
    → FlowsheetBuilder        (SVG/PNG diagram with actual chemical names)
    → OutputFormatter
"""

import json
import logging
import re
import sys
from pathlib import Path

from flora_translate.analogy_selector import AnalogySelector
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.config import LAB_INVENTORY_PATH, RECORDS_DIR
from flora_translate.design_calculator import GAS_LIQUID_MIN_BPR_BAR
from flora_translate.design_realizer import realize_executable_design
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.engine.council_v4 import CouncilV4
from flora_translate.intake_agent import (
    apply_intake_requirements_to_chemistry_plan,
    batch_input_from_package,
    historical_text_from_package,
    intake_context_block,
)
from flora_translate.final_design_validator import finalize_design
from flora_translate.final_design_contract import (
    build_final_design_contract,
    publish_final_design_artifacts,
)
from flora_translate.inventory_constraints import available_pressure_settings
from flora_translate.lightweight_upstream import analyze_batch_chemistry, parse_batch_input
from flora_translate.multistage_inventory import reconcile_multistage_inventory
from flora_translate.output_formatter import OutputFormatter
from flora_translate.pipeline_runtime import (
    PipelineRuntimeOptions,
    merged_hard_constraints,
    with_runtime_model_routing,
)
from flora_translate.prompt_builder import TranslationPromptBuilder
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    IN_CHANNEL_BASIS,
    LIQUID_ONLY_BASIS,
    normalize_residence_time_basis,
)
from flora_translate.retriever import VectorRetriever
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    DesignInputPackage,
    FlowProposal,
    LabInventory,
    LightSourceSpec,
    ProcessStage,
    ProcessTopology,
    ReactorSpec,
    StreamConnection,
    UnitOperation,
)
from flora_translate.translation_llm import TranslationLLM
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.topology_semantics import normalize_topology_semantics
from flora_translate.topology_preflight import analyze_topology_requirements
from flora_translate.vector_store import VectorStore

logger = logging.getLogger("flora.translate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = Path("outputs")
ROUTINE_GAS_LIQUID_BPR_MAX_BAR = 10.0
TOPOLOGY_DEFAULT_GAS_LIQUID_BPR_BAR = 5.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_intake_package(intake_package) -> DesignInputPackage | None:
    if intake_package is None:
        return None
    if isinstance(intake_package, DesignInputPackage):
        package = intake_package
    else:
        package = DesignInputPackage.model_validate(intake_package)
    if not package.ready_for_design:
        missing = ", ".join(package.missing_question_ids or [])
        raise ValueError(
            "DesignInputPackage is not ready for design"
            + (f"; missing intake questions: {missing}" if missing else "")
        )
    return package


def _inventory_from_intake_or_path(
    intake_package: DesignInputPackage | None,
    inventory_path: str,
) -> LabInventory:
    inventory_payload = None
    if intake_package is not None:
        inventory_payload = intake_package.inventory_constraints
        if isinstance(inventory_payload, str):
            stripped = inventory_payload.strip()
            if stripped.startswith("{"):
                try:
                    inventory_payload = json.loads(stripped)
                except json.JSONDecodeError:
                    inventory_payload = None
            else:
                parsed = _lab_inventory_from_text(stripped)
                if parsed and parsed.reactors:
                    return parsed
                inventory_payload = None
    if isinstance(inventory_payload, dict):
        try:
            return LabInventory.model_validate(inventory_payload)
        except Exception as exc:
            logger.warning("Intake inventory was not valid LabInventory JSON; using path inventory: %s", exc)
    return LabInventory.from_json(inventory_path)


def _lab_inventory_from_text(text: str) -> LabInventory | None:
    """Parse the standardized intake inventory text into LabInventory.

    This intentionally handles the compact human-readable inventory block
    emitted by ``inventory_prompt_block`` so hard constraints are not lost when
    a user pastes inventory as text instead of JSON.
    """

    text = str(text or "")
    if not text.strip():
        return None

    reactors: list[ReactorSpec] = []
    light_sources: list[LightSourceSpec] = []
    bpr_available: list[float] = []

    bpr_match = re.search(r"Available BPR/pressure settings:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if bpr_match:
        bpr_available = [
            float(v)
            for v in re.findall(r"[-+]?\d+(?:\.\d+)?", bpr_match.group(1))
        ]

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()

        if re.search(r"\b\d+(?:\.\d+)?\s*mL\b", item, flags=re.IGNORECASE) and re.search(
            r"\b\d+(?:\.\d+)?\s*mm\s*ID\b", item, flags=re.IGNORECASE
        ):
            reactors.append(_parse_inventory_reactor_line(item))
            continue

        if re.search(r"\b\d+(?:\.\d+)?\s*nm\b", item, flags=re.IGNORECASE):
            src = _parse_inventory_light_line(item)
            if src:
                light_sources.append(src)

    if not reactors and not light_sources and not bpr_available:
        return None
    return LabInventory(
        reactors=reactors,
        light_sources=light_sources,
        BPR_available=bpr_available,
    )


def _parse_inventory_reactor_line(item: str) -> ReactorSpec:
    header, _, rest = item.partition(":")
    body = rest or item
    full_name = header.strip() if rest else item.strip()

    system = ""
    name = full_name
    for prefix in ("Vapourtec System", "Manual Setup"):
        if full_name.startswith(prefix):
            system = prefix
            name = full_name[len(prefix):].strip() or full_name
            break

    volume = _safe_float(_first_match(body, r"(\d+(?:\.\d+)?)\s*mL\b"), 0.0)
    id_mm = _safe_float(_first_match(body, r"(\d+(?:\.\d+)?)\s*mm\s*ID\b"), 1.0)
    material_match = re.search(r"\b(FEP|PFA|PTFE|SS|stainless steel)\b", body, flags=re.IGNORECASE)
    material = (material_match.group(1).upper() if material_match else "FEP").replace("STAINLESS STEEL", "SS")

    constraints = item[item.find("(") + 1:item.rfind(")")] if "(" in item and ")" in item else item
    temp_allowed = [
        float(v)
        for v in re.findall(r"T\s+in\s+\[([^\]]+)\]", constraints, flags=re.IGNORECASE)
        for v in re.findall(r"[-+]?\d+(?:\.\d+)?", v)
    ]
    temp_range = _range_from_text(constraints, r"T\s+")
    conc_range = _range_from_text(constraints, r"C\s+")
    pressure_range = _range_from_text(constraints, r"P\s+")
    wavelength = _safe_float(_first_match(constraints, r"lambda\s+(\d+(?:\.\d+)?)\s*nm"), 0.0) or None
    intensity = _safe_float(_first_match(constraints, r"(\d+(?:\.\d+)?)\s*mW/cm2"), 0.0) or None

    light_source = ""
    if "vapourtec" in system.lower() and wavelength:
        light_source = f"Vapourtec mirrored LED bar {wavelength:g} nm"
    elif "manual" in system.lower() and wavelength:
        light_source = f"Manual strip LED {wavelength:g} nm"

    irradiation = ""
    lower = item.lower()
    if "mirrored" in lower:
        irradiation = "bilateral 360 irradiation with mirrored photoreactor"
    elif "360" in lower:
        irradiation = "standard 360 irradiation"

    return ReactorSpec(
        name=name,
        system=system,
        type="coil",
        material=material,
        volume_mL=volume,
        ID_mm=id_mm,
        light_source=light_source,
        wavelength_nm=wavelength,
        intensity_mW_cm2=intensity,
        irradiation=irradiation,
        min_temperature_C=temp_range[0],
        max_temperature_C=temp_range[1],
        allowed_temperatures_C=temp_allowed,
        min_concentration_M=conc_range[0],
        max_concentration_M=conc_range[1],
        min_pressure_bar=pressure_range[0],
        max_pressure_bar=pressure_range[1],
    )


def _parse_inventory_light_line(item: str) -> LightSourceSpec | None:
    header, _, rest = item.partition(":")
    source_text = rest or item
    wavelength = _safe_float(_first_match(source_text, r"(\d+(?:\.\d+)?)\s*nm"), 0.0)
    if wavelength <= 0:
        return None
    intensity = _safe_float(_first_match(source_text, r"(\d+(?:\.\d+)?)\s*mW/cm2"), 0.0) or None
    return LightSourceSpec(
        name=header.strip() or f"LED {wavelength:g} nm",
        wavelength_nm=wavelength,
        power_W=0.0,
        compatible_reactor="coil",
        intensity_mW_cm2=intensity,
    )


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _range_from_text(text: str, prefix_pattern: str) -> tuple[float | None, float | None]:
    match = re.search(prefix_pattern + r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _reconcile_final_bpr(
    result: dict,
    design_candidate=None,
    inventory: LabInventory | None = None,
) -> None:
    """Keep final BPR tied to the validated winner, not stale pre-council math."""
    calc = result.get("design_calculations") or {}
    proposal = result.get("proposal") or {}
    if not calc.get("is_gas_liquid"):
        return

    proposal_bpr = _safe_float(proposal.get("BPR_bar"))
    calc_bpr = _safe_float(calc.get("bpr_pressure_bar"))
    final_bpr = max(proposal_bpr, GAS_LIQUID_MIN_BPR_BAR)

    # The gas-liquid minimum is an engineering requirement, not a physical
    # BPR setpoint. Resolve it upward to an actually declared inventory
    # setting instead of silently producing an unavailable value such as
    # 3.0 bar when the lab owns only 2.5 and 7.0 bar cartridges.
    settings = available_pressure_settings(inventory)
    eligible = [
        value
        for value in settings
        if value + 1e-12 >= final_bpr
        and value <= ROUTINE_GAS_LIQUID_BPR_MAX_BAR + 1e-12
    ]
    if eligible:
        final_bpr = min(eligible)

    note = ""
    if calc_bpr > final_bpr + 0.1:
        note = (
            f"Calculator BPR {calc_bpr:.1f} bar exceeds proposal BPR "
            f"{final_bpr:.1f} bar; this is treated as a validation warning, "
            "not silently promoted to the final proposal."
        )
    if final_bpr > ROUTINE_GAS_LIQUID_BPR_MAX_BAR:
        note = (
            f"Final BPR {final_bpr:.1f} bar exceeds the "
            f"{ROUTINE_GAS_LIQUID_BPR_MAX_BAR:.0f} bar routine gas-liquid ceiling. "
            "This design requires geometry redesign or manual hardware review."
        )
        proposal["engine_validated"] = False
        flags = proposal.setdefault("safety_flags", [])
        if isinstance(flags, list):
            flags.append("SCREEN_REQUIRED: BPR exceeds routine gas-liquid hardware ceiling")

    proposal["BPR_bar"] = round(final_bpr, 1)
    calc["bpr_required"] = True
    calc["bpr_pressure_bar"] = round(final_bpr, 1)
    if note:
        calc["bpr_reconciliation_note"] = note
    try:
        design_candidate.proposal.BPR_bar = round(final_bpr, 1)
    except Exception:
        pass


def _is_quench_stream_dict(stream: dict) -> bool:
    role = str(stream.get("pump_role") or "").lower()
    return any(k in role for k in ("quench", "neutraliz", "neutralis", "workup", "post-reactor"))


def _stream_dict_is_gas(stream: dict) -> bool:
    from types import SimpleNamespace

    return _stream_is_gas(SimpleNamespace(**stream))


def _sync_final_stream_flowrates(result: dict, design_candidate=None) -> None:
    """Make per-stream pump rates match the final proposal/calculator.

    The Chief may initially derive stream rates from molar flow and feed
    concentrations. Later deterministic gas-liquid sizing can change the final
    authoritative liquid flow. Without this pass, Summary/Engineering can show
    the final Q while Stream Assignments and topology still show stale pump Q.
    """
    proposal = result.get("proposal") or {}
    streams = proposal.get("streams") or []
    if not streams:
        return

    calc = result.get("design_calculations") or {}
    target_liquid_q = _safe_float(
        calc.get("liquid_flow_rate_mL_min"),
        _safe_float(proposal.get("flow_rate_mL_min")),
    )
    if target_liquid_q <= 0:
        target_liquid_q = _safe_float(proposal.get("flow_rate_mL_min"))

    liquid_streams = [
        s for s in streams
        if not _stream_dict_is_gas(s) and not _is_quench_stream_dict(s)
    ]
    if target_liquid_q > 0 and liquid_streams:
        current_sum = sum(_safe_float(s.get("flow_rate_mL_min")) for s in liquid_streams)
        if len(liquid_streams) == 1:
            assignments = {id(liquid_streams[0]): round(target_liquid_q, 5)}
        elif current_sum > 0:
            scale = target_liquid_q / current_sum
            assignments = {
                id(s): round(_safe_float(s.get("flow_rate_mL_min")) * scale, 5)
                for s in liquid_streams
            }
        else:
            equal = round(target_liquid_q / len(liquid_streams), 5)
            assignments = {id(s): equal for s in liquid_streams}

        for s in liquid_streams:
            old_rate = _safe_float(s.get("flow_rate_mL_min"))
            new_rate = assignments[id(s)]
            if abs(old_rate - new_rate) > 1e-5:
                s["flow_rate_mL_min"] = new_rate
                note = (
                    f"Final synchronization: liquid pump rate set to {new_rate:.5f} "
                    f"mL/min so Σ liquid reactor feeds = final Q_liquid "
                    f"{target_liquid_q:.5f} mL/min."
                )
                existing = str(s.get("reasoning") or "").strip()
                s["reasoning"] = f"{existing}\n{note}".strip() if existing else note

    gas_sccm = calc.get("gas_flow_sccm")
    gas_actual = calc.get("gas_flow_actual_mL_min")
    target_gas_equiv = _safe_float(calc.get("target_gas_equiv_inlet"))
    supplied_gas_equiv = _safe_float(
        calc.get("gas_equiv_supplied"),
        _safe_float(calc.get("o2_equiv_supplied")),
    )
    for s in streams:
        if not _stream_dict_is_gas(s):
            continue
        s["phase"] = "gas"
        if gas_sccm is not None:
            s["gas_flow_sccm"] = round(_safe_float(gas_sccm), 4)
        if gas_actual is not None:
            s["gas_flow_actual_mL_min"] = round(_safe_float(gas_actual), 5)
            s["flow_rate_mL_min"] = round(_safe_float(gas_actual), 5)
        if target_gas_equiv > 0:
            s["molar_equiv"] = round(target_gas_equiv, 4)
            s["reasoning"] = (
                "Deterministic gas stoichiometry: inlet/STP MFC flow was "
                f"recomputed from target gas equivalents ({target_gas_equiv:.2f} equiv), "
                f"liquid flow {target_liquid_q:.5f} mL/min, and substrate concentration. "
                f"Pressure-corrected in-channel gas flow is reported separately; "
                f"supplied gas equiv = {supplied_gas_equiv:.2f}."
            )

    # Mirror the synchronized dict streams back into the Pydantic proposal used
    # by topology generation.
    try:
        from flora_translate.schemas import StreamAssignment

        design_candidate.proposal.streams = [StreamAssignment(**s) for s in streams]
    except Exception:
        pass


def _apply_final_design_guards(result: dict, design_candidate=None) -> None:
    """Reject final JSON that violates measured evidence or O2 stoichiometry."""

    proposal = result.get("proposal") or {}
    calc = result.get("design_calculations") or {}
    flags = proposal.setdefault("safety_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
        proposal["safety_flags"] = flags

    calibration = proposal.get("evidence_calibration") or {}
    if calibration:
        basis_label = str(calibration.get("primary_residence_time_basis") or "").lower()
        best_response = _safe_float(calibration.get("best_response_pct"))
        target_response = _safe_float(calibration.get("target_response_pct"))
        if "inlet" in basis_label or "stp" in basis_label:
            final_tau = _safe_float(
                proposal.get("residence_time_inlet_min"),
                _safe_float(proposal.get("residence_time_min")),
            )
            best_tau = _safe_float(
                calibration.get("best_tau_inlet_min"),
                _safe_float(calibration.get("best_tau_min")),
            )
        elif "channel" in basis_label:
            final_tau = _safe_float(
                proposal.get("residence_time_in_channel_min"),
                _safe_float(proposal.get("residence_time_min")),
            )
            best_tau = _safe_float(
                calibration.get("best_tau_in_channel_min"),
                _safe_float(calibration.get("best_tau_min")),
            )
        else:
            final_tau = _safe_float(proposal.get("residence_time_min"))
            best_tau = _safe_float(calibration.get("best_tau_min"))
        if best_tau > 0 and final_tau > 0 and best_response < target_response and final_tau < best_tau * 0.999:
            proposal["engine_validated"] = False
            flags.append(
                "BLOCKED: final residence time is below the best measured evidence anchor "
                f"({final_tau:.2f} < {best_tau:.2f} min) while response is still below target."
            )

    target_equiv = _safe_float(calc.get("target_gas_equiv_inlet"))
    supplied_equiv = _safe_float(
        calc.get("gas_equiv_supplied"),
        _safe_float(calc.get("o2_equiv_supplied")),
    )
    if calc.get("is_gas_liquid") and target_equiv > 0:
        if supplied_equiv <= 0 or supplied_equiv < target_equiv * 0.95:
            proposal["engine_validated"] = False
            flags.append(
                "BLOCKED: final reagent-gas feed does not meet the target inlet/STP equivalents "
                f"({supplied_equiv:.2f} supplied vs {target_equiv:.2f} target)."
            )
        proposal.setdefault("multiphase_metrics", {})["target_gas_equiv_inlet"] = target_equiv
        proposal.setdefault("multiphase_metrics", {})["gas_equiv_supplied"] = supplied_equiv
        proposal.setdefault("multiphase_metrics", {})["o2_equiv_supplied"] = supplied_equiv

    try:
        design_candidate.proposal = FlowProposal(**proposal)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chemistry-aware topology builder
# ---------------------------------------------------------------------------

def _build_translate_topology(
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    batch_record: BatchRecord,
    inventory: LabInventory | None = None,
) -> ProcessTopology:
    """Convert a FlowProposal + ChemistryPlan into a ProcessTopology.

    Handles both single-step and multi-step processes:
    - Single-step: pumps → mixer → [degas] → reactor → [BPR] → [quench] → collector
    - Multi-step: each stage gets its own reactor zone with mid-stream
      injection points, inter-stage quench/filter/solvent-switch operations.
    """
    # Dispatch: multi-step if ChemistryPlan has stages
    if chemistry_plan and chemistry_plan.stages and chemistry_plan.n_stages > 1:
        topology = _build_multistep_topology(
            proposal, chemistry_plan, batch_record, inventory=inventory
        )
    else:
        topology = _build_singlestep_topology(
            proposal, chemistry_plan, batch_record, inventory=inventory
        )
    from flora_translate.executable_artifacts import _stream_components
    feed_map = {s.stream_label: s for s in proposal.streams}
    components = _stream_components(
        {"chemistry_plan": chemistry_plan.model_dump() if chemistry_plan else {}},
        [s.model_dump() for s in proposal.streams],
    )
    for operation in topology.unit_operations:
        if operation.op_type not in {"pump", "mfc"}:
            continue
        feed = feed_map.get(operation.parameters.get("stream"))
        if feed is None:
            continue
        operation.parameters["concentration_M"] = feed.concentration_M
        operation.parameters["solvent"] = feed.solvent
        operation.parameters["gas_reagent_mole_fraction"] = feed.gas_reagent_mole_fraction
        if feed.phase == "gas":
            continue
        labels = []
        for component in components:
            if component.stream_label != feed.stream_label or component.role == "solvent":
                continue
            details = []
            if component.concentration_M is not None:
                details.append(f"{component.concentration_M:.6g} M")
            if component.molar_equiv is not None:
                details.append(f"{component.molar_equiv:.6g} equiv")
            elif component.loading_mol_pct is not None:
                details.append(f"{component.loading_mol_pct:.6g} mol%")
            labels.append(component.name + (" (" + ", ".join(details) + ")" if details else ""))
        operation.parameters["contents"] = labels or list(feed.contents)
    return normalize_topology_semantics(topology)


def _connect(ops, streams, counter, from_op, to_op, label=""):
    """Helper: add a stream connection."""
    counter[0] += 1
    streams.append(StreamConnection(
        stream_id=f"s{counter[0]}", from_op=from_op,
        to_op=to_op, stream_type="liquid", label=label
    ))


_GAS_IDENTITIES = {
    "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂", "h2", "h₂",
    "hydrogen", "ar", "argon", "he", "helium", "air", "compressed air",
    "co", "carbon monoxide", "cl2", "chlorine", "nh3", "ammonia",
    "so2", "sulfur dioxide", "n2o", "nitrous oxide",
}
_GAS_PHASE_WORDS = {"gas", "gaseous", "vapor", "vapour", "mfc"}
_LIQUID_PHASE_WORDS = {
    "liquid", "solution", "solvent", "dissolved", "degassed", "sparged",
    "reagent", "substrate",
}


def _normalized_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9₂]+", " ", str(text).lower()).split()


def _has_explicit_gas_identity(text: str) -> bool:
    normalized = " " + " ".join(_normalized_words(text)) + " "
    for kw in _GAS_IDENTITIES:
        key = re.sub(r"[^a-z0-9₂]+", " ", kw.lower()).strip()
        if key and f" {key} " in normalized:
            return True
    return False


def _is_gas_phase_value(value) -> bool:
    words = set(_normalized_words(value))
    return bool(words & _GAS_PHASE_WORDS) and not bool(words & _LIQUID_PHASE_WORDS)


def _has_real_solvent(value) -> bool:
    if value is None:
        return False
    normalized = " ".join(_normalized_words(value))
    return normalized not in {"", "none", "no solvent", "na", "n a", "null", "gas"}


def _is_pure_gas_text(text: str) -> bool:
    words = set(_normalized_words(text))
    if not _has_explicit_gas_identity(text):
        return False
    if words & _LIQUID_PHASE_WORDS:
        return False
    return True


def _is_gas_contents(contents) -> bool:
    if isinstance(contents, str):
        contents = [contents]
    items = [str(c).strip() for c in (contents or []) if str(c).strip()]
    return bool(items) and all(_is_pure_gas_text(item) for item in items)

def _stream_is_gas(stream) -> bool:
    """True only for genuine gas feed streams (not degassed liquid solutions)."""
    phase = getattr(stream, "phase", None) or getattr(stream, "state", None)
    if phase and _is_gas_phase_value(phase):
        return True
    if phase and set(_normalized_words(phase)) & _LIQUID_PHASE_WORDS:
        return False

    role = getattr(stream, "pump_role", None) or ""
    role_words = set(_normalized_words(role))
    if role_words & {"quench", "neutralization", "neutralisation", "workup"}:
        return False
    if _is_gas_phase_value(role):
        return True
    if _is_pure_gas_text(role):
        return True

    solvent = getattr(stream, "solvent", None)
    if _has_real_solvent(solvent):
        return False

    contents = (
        getattr(stream, "contents", None)
        or getattr(stream, "reagents", None)
        or []
    )
    return _is_gas_contents(contents)


def _enforce_gas_delivery_hardware(ops: list[UnitOperation]) -> None:
    """Final topology pass: gas delivery streams must use MFC nodes."""
    for op in ops:
        if op.op_type not in ("pump", "mfc"):
            continue
        p = op.parameters or {}
        phase = p.get("phase") or p.get("state") or p.get("stream_type")
        has_solvent = bool(p.get("solvent"))
        contents = p.get("contents") or p.get("components") or []

        is_gas = False
        if phase and _is_gas_phase_value(phase):
            is_gas = True
        elif phase and set(_normalized_words(phase)) & _LIQUID_PHASE_WORDS:
            is_gas = False
        elif not has_solvent and _is_gas_contents(contents):
            is_gas = True
        elif not has_solvent and not contents and _is_pure_gas_text(op.label or ""):
            is_gas = True

        if not is_gas:
            continue
        op.op_type = "mfc"
        if not str(op.label or "").lower().startswith("mfc"):
            label = str(op.label or "").replace("Pump", "MFC", 1)
            op.label = label if label.startswith("MFC") else f"MFC — {label}"
        p["delivery_hardware"] = "MFC"
        op.parameters = p


def _add_pump(ops, label_char, role, contents, solvent, flow_rate, reasoning,
              is_gas: bool = False, gas_flow_sccm=None, gas_flow_actual_mL_min=None,
              inventory_equipment_id=None, molar_equiv=None):
    """Helper: create a pump (or MFC for gas) UnitOperation."""
    op_id = f"pump_{label_char.lower()}"
    ops.append(UnitOperation(
        op_id=op_id, op_type="mfc" if is_gas else "pump",
        label=f"{'MFC' if is_gas else 'Pump'} {label_char} — {role}",
        parameters={
            "stream": label_char,
            "contents": contents,
            "solvent": solvent,
            "flow_rate_mL_min": flow_rate,
            "phase": "gas" if is_gas else "liquid",
            "gas_flow_sccm": gas_flow_sccm,
            "gas_flow_actual_mL_min": gas_flow_actual_mL_min,
            "molar_equiv": molar_equiv,
            "inventory_equipment_id": inventory_equipment_id,
        },
        required=True, rationale=reasoning,
    ))
    return op_id


def _stoich_flow_rates(streams_list, total_Q: float) -> list[float]:
    """Compute per-stream flow rates from molar_equiv and concentration.

    weight_i = equiv_i / conc_i  (volume of stream i needed per unit substrate volume)
    Q_i = total_Q × weight_i / Σweights

    Falls back to equal split if all equivs are 1.0 and no concentrations differ.
    Gas streams are excluded from the calculation (they get zero Q from this function).
    """
    liquid = [s for s in streams_list if not _stream_is_gas(s)]
    if not liquid:
        return [0.0] * len(streams_list)

    equivs = [max(getattr(s, "molar_equiv", 1.0) or 1.0, 1e-9) for s in liquid]
    concs  = [max(getattr(s, "concentration_M", None) or 1.0, 1e-9) for s in liquid]
    weights = [e / c for e, c in zip(equivs, concs)]
    total_w = sum(weights)
    qs_liquid = [round(total_Q * w / total_w, 4) for w in weights]

    # Re-expand to full list (gas streams get 0 here — handled separately)
    liq_iter = iter(qs_liquid)
    return [0.0 if _stream_is_gas(s) else next(liq_iter) for s in streams_list]


def _is_quench_stream(s, chemistry_plan) -> bool:
    """Classify a stream as a quench/workup stream.

    Quench streams are injected AFTER the main reactor at a separate mixer.
    They must never feed into the main reactor's T-mixer, or Pump C ends up
    plumbed to two places at once.
    """
    role = (getattr(s, "pump_role", "") or "").lower()
    if any(kw in role for kw in ("quench", "neutraliz", "workup", "post-reactor")):
        return True
    # Also match if the stream contents mention the plan's quench_reagent
    qr = (getattr(chemistry_plan, "quench_reagent", "") or "").lower() if chemistry_plan else ""
    if qr:
        contents = getattr(s, "contents", None) or []
        for c in contents:
            if qr in str(c).lower():
                return True
    return False


def _uses_offline_deoxygenation(proposal: FlowProposal) -> bool:
    """Return whether oxygen exclusion is performed before the flow setup.

    Offline sparging or pre-degassed feed preparation belongs in the operating
    procedure and feed-reservoir labels, not as an inline unit operation.
    """

    text = " ".join(
        [
            str(proposal.deoxygenation_method or ""),
            *[str(step) for step in proposal.pre_reactor_steps or []],
        ]
    ).lower()
    return any(
        marker in text
        for marker in (
            "deoxygenation not required",
            "deoxygenation is not required",
            "deoxygenation not needed",
            "no deoxygenation required",
            "offline argon sparg",
            "offline ar sparg",
            "offline nitrogen sparg",
            "offline n2 sparg",
            "pre-degassed",
            "predegassed",
            "pre-deoxygenated",
            "no inline degasser",
            "inline degasser is unavailable",
            "inline degasser unavailable",
            "offline nitrogen purge",
            "offline n2 purge",
        )
    )


def _build_singlestep_topology(
    proposal, chemistry_plan, batch_record, inventory: LabInventory | None = None
):
    """Linear single-step topology.

    Design contract — ONE SOURCE OF TRUTH:
      Every flow rate shown in the diagram comes from proposal.streams
      (populated by the Chief Engineer). This function does not recompute
      pump rates from stoichiometry when proposal.streams already carries
      Chief-normalised values.

    Stream classification:
      • reactor_feeds → pumps feed the main T-mixer → reactor
      • quench_streams → injected at a post-reactor Quench T-mixer
      A stream is NEVER both; this prevents Pump C appearing twice.

    Q conservation:
      Q_reactor_inlet = Σ Q_i (reactor_feeds)
      Q_quench_inlet  = Q_reactor_inlet + Σ Q_j (quench_streams)
    """
    import math

    ops: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    sc = [0]

    is_photochem = proposal.wavelength_nm is not None

    # ── Classify streams ───────────────────────────────────────────────────
    all_streams = list(proposal.streams or [])
    reactor_feeds = [s for s in all_streams if not _is_quench_stream(s, chemistry_plan)]
    quench_streams = [s for s in all_streams if _is_quench_stream(s, chemistry_plan)]

    # ── Pumps for reactor feeds (only) ─────────────────────────────────────
    pump_ids = []
    if reactor_feeds:
        total_Q = proposal.flow_rate_mL_min or 0.5
        # Only compute stoichiometric split as a fallback when a stream
        # lacks an explicit flow_rate_mL_min.
        qs_fallback = _stoich_flow_rates(reactor_feeds, total_Q)
        for s, fr_fallback in zip(reactor_feeds, qs_fallback):
            fr = s.flow_rate_mL_min if s.flow_rate_mL_min else fr_fallback
            gas_sccm = getattr(s, "gas_flow_sccm", None)
            gas_actual = getattr(s, "gas_flow_actual_mL_min", None)
            if _stream_is_gas(s) and not gas_sccm:
                gas_sccm = (proposal.multiphase_metrics or {}).get("gas_flow_sccm")
                gas_actual = (proposal.multiphase_metrics or {}).get("gas_flow_actual_mL_min")
                fr = gas_actual if gas_actual is not None else fr
            pid = _add_pump(ops, s.stream_label, s.pump_role,
                            s.contents, s.solvent, fr, s.reasoning or "",
                            is_gas=_stream_is_gas(s),
                            gas_flow_sccm=gas_sccm,
                            gas_flow_actual_mL_min=gas_actual,
                            inventory_equipment_id=s.pump_equipment_id,
                            molar_equiv=s.molar_equiv)
            pump_ids.append(pid)
    else:
        default_Q = round((proposal.flow_rate_mL_min or 0.5) / 2, 4)
        for lbl in ("A", "B"):
            pid = _add_pump(ops, lbl, "reagent", [], "", default_Q, "")
            pump_ids.append(pid)

    # ── Main T-mixer: reactor feeds only ───────────────────────────────────
    ops.append(UnitOperation(op_id="mixer_1", op_type="mixer",
        label=proposal.mixer_type or "T-Mixer",
        parameters={"type": proposal.mixer_type or "T-mixer", "material": "not specified"},
        required=True, rationale=proposal.mixing_order_reasoning or "Combine reactor feeds"))
    for pid in pump_ids:
        _connect(ops, streams, sc, pid, "mixer_1")
    prev = "mixer_1"

    # ── Deoxygenation ──────────────────────────────────────────────────────
    deoxy = proposal.deoxygenation_method
    if not deoxy and chemistry_plan and chemistry_plan.deoxygenation_required:
        deoxy = "N2 sparging"
    if deoxy and not _uses_offline_deoxygenation(proposal):
        ops.append(UnitOperation(op_id="deoxy_1", op_type="deoxygenation_unit",
            label="Inline Deoxygenation", parameters={"method": deoxy},
            required=True, rationale=chemistry_plan.deoxygenation_reasoning if chemistry_plan else ""))
        _connect(ops, streams, sc, prev, "deoxy_1")
        prev = "deoxy_1"

    # ── Main reactor ───────────────────────────────────────────────────────
    mat = proposal.tubing_material
    reactor_type = (proposal.reactor_type or "coil").lower()
    if "packed" in reactor_type or "bed" in reactor_type:
        reactor_op_type = "packed_bed_reactor"
        reactor_label = f"{mat} Packed-Bed Reactor"
    elif any(token in reactor_type for token in ("microchannel", "microreactor", "chip")):
        reactor_op_type = "chip_reactor"
        reactor_label = f"{mat} Microchannel Reactor"
    else:
        reactor_op_type = "photoreactor" if is_photochem else "coil_reactor"
        reactor_label = f"{mat} Photoreactor Coil" if is_photochem else f"{mat} Flow Reactor Coil"
    vol = proposal.reactor_volume_mL
    id_mm = proposal.tubing_ID_mm
    # Q entering the main reactor = sum of reactor_feed pump rates (gas excluded)
    Q_reactor_inlet = sum(
        (s.flow_rate_mL_min or 0.0)
        for s in reactor_feeds
        if not _stream_is_gas(s)
    ) or proposal.flow_rate_mL_min or 0.0
    ops.append(UnitOperation(op_id="reactor_1", op_type=reactor_op_type,
        label=reactor_label, parameters={
            "material": mat, "ID_mm": id_mm, "volume_mL": vol,
            "Q_inlet_mL_min": Q_reactor_inlet,
            "Q_liquid_mL_min": Q_reactor_inlet,
            "Q_gas_sccm": sum(float(s.gas_flow_sccm or 0.0) for s in reactor_feeds if _stream_is_gas(s)),
            "Q_gas_actual_mL_min": sum(float(s.gas_flow_actual_mL_min or 0.0) for s in reactor_feeds if _stream_is_gas(s)),
            "temperature_C": proposal.temperature_C, "wavelength_nm": proposal.wavelength_nm,
            "residence_time_min": proposal.residence_time_min,
            "residence_time_basis": proposal.residence_time_basis,
            "residence_time_inlet_min": proposal.residence_time_inlet_min,
            "residence_time_in_channel_min": proposal.residence_time_in_channel_min,
            "reactor_type": proposal.reactor_type,
        }, required=True, rationale="Flow reactor"))
    _connect(ops, streams, sc, prev, "reactor_1")
    prev = "reactor_1"

    if inventory and inventory.temperature_controllers:
        ops.append(UnitOperation(
            op_id="heater_1",
            op_type="heater",
            label=f"Temperature Control {proposal.temperature_C:g} deg C",
            parameters={"temperature_C": proposal.temperature_C},
            required=True,
            rationale="Active reactor temperature control",
        ))

    # ── LED ────────────────────────────────────────────────────────────────
    if is_photochem:
        ops.append(UnitOperation(op_id="led_1", op_type="led_module",
            label=f"LED {proposal.wavelength_nm:.0f} nm",
            parameters={"wavelength_nm": proposal.wavelength_nm},
            required=True, rationale="Photoexcitation"))

    # ── BPR ────────────────────────────────────────────────────────────────
    if proposal.BPR_bar and proposal.BPR_bar > 0:
        ops.append(UnitOperation(op_id="bpr_1", op_type="bpr",
            label="BPR", parameters={"pressure_bar": proposal.BPR_bar},
            required=True, rationale="Maintain liquid phase"))
        _connect(ops, streams, sc, prev, "bpr_1")
        prev = "bpr_1"

    # ── Quench: T-mixer + short contact coil ───────────────────────────────
    # Only runs if plan says quench_required OR a quench stream exists.
    # An unspecified quench reagent is an offline/collection instruction, not
    # permission to invent a quench pump and mixer. Inline quench hardware is
    # created only when an explicit serialized quench feed exists.
    needs_quench = bool(quench_streams)
    if needs_quench:
        # Create pumps for each quench stream. If the plan flags quench_required
        # but no quench stream exists in proposal.streams, synthesise a default.
        quench_pump_ids: list[str] = []
        Q_quench_total = 0.0
        if quench_streams:
            for s in quench_streams:
                fr = s.flow_rate_mL_min or 0.0
                Q_quench_total += fr
                pid = _add_pump(
                    ops, s.stream_label or "Q", s.pump_role or "Inline quench",
                    s.contents, s.solvent, fr, s.reasoning or "",
                    is_gas=_stream_is_gas(s),
                    gas_flow_sccm=getattr(s, "gas_flow_sccm", None),
                    gas_flow_actual_mL_min=getattr(s, "gas_flow_actual_mL_min", None),
                    inventory_equipment_id=s.pump_equipment_id,
                )
                quench_pump_ids.append(pid)
        # Q_inlet to the quench coil = reactor outlet + ALL quench pump rates
        Q_quench_inlet = round(Q_reactor_inlet + Q_quench_total, 4)

        # Quench T-mixer: reactor outlet + quench pumps
        ops.append(UnitOperation(op_id="quench_mixer", op_type="mixer",
            label="Quench T-Mixer",
            parameters={
                "type": "T-mixer",
                "material": "not specified",
                "Q_inlet_mL_min": Q_quench_inlet,
                "quench_flow_rate_mL_min": round(Q_quench_total, 4),
                "reactor_outlet_flow_rate_mL_min": round(Q_reactor_inlet, 4),
                "role": "inline safety neutralization, not a reaction stage",
            },
            required=True, rationale="Mix reactor outlet with quench stream(s)"))
        _connect(ops, streams, sc, prev, "quench_mixer")
        for qpid in quench_pump_ids:
            _connect(ops, streams, sc, qpid, "quench_mixer")
        prev = "quench_mixer"
        final_outlet_Q = Q_quench_inlet
    else:
        final_outlet_Q = Q_reactor_inlet

    # Material/phase separation explicitly requested by the chemistry or
    # operating procedure is represented only when declared inventory exists.
    separator_text = " ".join(
        [
            batch_record.reaction_description or "",
            batch_record.raw_text or "",
            " ".join(proposal.post_reactor_steps or []),
            getattr(chemistry_plan, "reaction_class", "") if chemistry_plan else "",
        ]
    ).lower()
    if inventory and inventory.separators and any(
        token in separator_text
        for token in (
            "phase separat", "separator", "separate", "separation", "liquid-liquid",
            "liquid liquid", "biphas", "spent-acid",
        )
    ):
        separator_phases = (
            ["gas", "liquid"]
            if any(_stream_is_gas(stream) for stream in proposal.streams)
            else ["liquid", "liquid"]
        )
        ops.append(UnitOperation(
            op_id="separator_1",
            op_type="phase_separator",
            label="Phase Separator",
            parameters={"phases": separator_phases},
            required=True,
            rationale=(
                "Separate gas from the product-containing liquid."
                if separator_phases[0] == "gas"
                else "Separate product-containing and spent reagent phases."
            ),
        ))
        _connect(ops, streams, sc, prev, "separator_1")
        prev = "separator_1"

    # ── Collector ──────────────────────────────────────────────────────────
    ops.append(UnitOperation(op_id="collector_1", op_type="collector",
        label="Product Collection", parameters={}, required=True, rationale="Outlet"))
    _connect(ops, streams, sc, prev, "collector_1")

    _enforce_gas_delivery_hardware(ops)

    pid_parts = [o.label for o in ops if o.op_type != "led_module"]
    return ProcessTopology(
        topology_id="translate", unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=final_outlet_Q,
        residence_time_min=proposal.residence_time_min,
        reactor_volume_mL=proposal.reactor_volume_mL,
        pid_description=" → ".join(pid_parts),
        topology_confidence=proposal.confidence,
    )


def _build_multistep_topology(
    proposal, chemistry_plan, batch_record, inventory: LabInventory | None = None
):
    """Graph-based multi-step topology.

    Each stage in chemistry_plan.stages becomes:
      [new feed pumps] → [mixer with previous outlet] → [pre-stage ops] →
      [reactor for this stage] → [post-stage ops (quench/filter/solvent switch)]
      → feeds into next stage

    Key correctness rules:
    - Q_inlet_i = Q_from_previous_stage + sum(Q_new_feeds_i)
    - New feed Qs are derived from molar_equiv and concentration (stoichiometric split)
    - V_R_i = τ_i × Q_inlet_i  (not the global Q)
    - τ_i comes from proposal.stage_parameters (council decision) or equal split
    - d_mm per stage also comes from stage_parameters if the council set it
    """
    import math

    ops: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    sc = [0]
    pump_char_counter = [0]

    def next_pump_char():
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c = chars[pump_char_counter[0] % 26]
        pump_char_counter[0] += 1
        return c

    n_stages = len(chemistry_plan.stages)
    total_Q   = proposal.flow_rate_mL_min or 0.5
    selected_components = [
        float(value)
        for value in (
            (proposal.inventory_selection or {}).get("component_volumes_mL")
            or []
        )
        if _safe_float(value) > 0
    ]
    use_inventory_components = len(selected_components) >= n_stages

    # Build a lookup for council-provided per-stage diameter overrides
    stage_params_by_sn: dict[int, dict] = {
        p["stage_number"]: p
        for p in (proposal.stage_parameters or [])
        if isinstance(p, dict) and "stage_number" in p
    }

    def _planned_stage_residence_times() -> dict[int, float]:
        """Return complete per-stage tau values for topology construction.

        The council currently selects one global residence time.  If per-stage
        tau values are incomplete, treat that council value as total process
        residence time and allocate it across stages.  Never let later stages
        silently fall back to batch/IF kinetics after council selection.
        """
        explicit: dict[int, float] = {}
        for sn, sp in stage_params_by_sn.items():
            try:
                tau = float(sp.get("residence_time_min") or 0.0)
            except (TypeError, ValueError):
                tau = 0.0
            if tau > 0:
                explicit[int(sn)] = round(tau, 2)

        stage_numbers = [int(s.stage_number) for s in chemistry_plan.stages]
        if stage_numbers and all(sn in explicit for sn in stage_numbers):
            return explicit

        total_tau = float(proposal.residence_time_min or 0.0)
        if total_tau <= 0:
            return explicit

        weights = []
        for stage in chemistry_plan.stages:
            try:
                stage_batch_h = float(getattr(stage, "batch_time_h", None) or 0.0)
            except (TypeError, ValueError):
                stage_batch_h = 0.0
            weights.append(stage_batch_h if stage_batch_h > 0 else 0.0)

        if not weights or sum(weights) <= 0:
            weights = [1.0 for _ in chemistry_plan.stages]

        total_weight = sum(weights)
        return {
            int(stage.stage_number): round(total_tau * weight / total_weight, 2)
            for stage, weight in zip(chemistry_plan.stages, weights)
        }

    stage_tau_by_sn = _planned_stage_residence_times()
    proposal_rate_by_label: dict[str, float] = {
        (stream.stream_label or "").upper(): float(stream.flow_rate_mL_min)
        for stream in (proposal.streams or [])
        if (
            stream.stream_label
            and stream.flow_rate_mL_min is not None
            and not _stream_is_gas(stream)
        )
    }
    proposal_liquid_labels = set(proposal_rate_by_label)
    proposal_gas_by_label: dict[str, object] = {
        (s.stream_label or "").upper(): s
        for s in (proposal.streams or [])
        if (s.stream_label or "") and _stream_is_gas(s)
    }
    proposal_gas_streams = list(proposal_gas_by_label.values())

    def _proposal_gas_for_feed(feed):
        matched = proposal_gas_by_label.get((feed.stream_label or "").upper())
        if matched is None and _stream_is_gas(feed) and len(proposal_gas_streams) == 1:
            return proposal_gas_streams[0]
        return matched

    prev_op: str | None = None
    Q_prev_outlet: float = 0.0  # cumulative Q leaving the previous reactor
    total_reactor_volume_mL: float = 0.0

    # Reference Q and C for stoichiometric new-feed sizing in stages 2+
    Q_reference: float = total_Q
    C_reference: float = proposal.concentration_M or 0.1

    # Resolve each feed to its declared injection stage. Provider-specific
    # plans sometimes repeat a Stage 1 feed under Stage 2 while retaining
    # introduction_stage=1. The declaration, not list position, owns the pump.
    _label_injection_stage: dict[str, int] = {}
    for _stg in chemistry_plan.stages:
        for _feed in _stg.feed_streams:
            lbl = (_feed.stream_label or "").upper()
            if lbl and _feed.delivery_mode != "carried_from_previous":
                declared_stage = int(
                    _feed.introduction_stage or _stg.stage_number
                )
                _label_injection_stage[lbl] = declared_stage

    for stage in chemistry_plan.stages:
        sn = stage.stage_number
        prefix = f"st{sn}"
        sp = stage_params_by_sn.get(sn, {})

        # Classify the stage itself, not arbitrary prose attached to its feeds
        # or the operation that follows it. Feed reasoning commonly mentions
        # a later aqueous workup, and ``post_stage_action`` describes what
        # happens after this stage's reactor. Neither should remove the current
        # reaction stage from the executable topology.
        stage_identity_text = " ".join([
            stage.stage_name or "",
            stage.reaction_type or "",
        ]).lower()
        explicitly_nonreaction_stage = any(
            token in stage_identity_text
            for token in (
                "inline quench", "quench stage", "acid quench", "aqueous quench",
                "neutraliz", "protonat", "destroy residual", "reductive workup",
                "workup only", "not a reaction",
            )
        )
        finalized_reactor_allocation = bool(
            sp.get("reactor_equipment_id")
            or _safe_float(sp.get("reactor_volume_mL") or sp.get("V_R_mL")) > 0
            or _safe_float(
                sp.get("residence_time_min")
                or sp.get("residence_time_inlet_min")
            ) > 0
        )
        is_quench_stage = (
            explicitly_nonreaction_stage and not finalized_reactor_allocation
        )
        if explicitly_nonreaction_stage and finalized_reactor_allocation:
            logger.warning(
                "Stage %s (%s) contains non-reaction wording but has a finalized "
                "reactor allocation; preserving the canonical reactor assignment.",
                sn,
                stage.stage_name,
            )

        # ── Per-stage τ ─────────────────────────────────────────────────────
        # Council-approved topology uses the complete per-stage allocation.
        # If the council did not provide all stage taus, the allocation above
        # splits the global council tau across stages. This prevents an
        # unreviewed Stage 2 from reverting to batch/IF timing.
        if stage_tau_by_sn.get(sn):
            stage_rt = stage_tau_by_sn[sn]
        else:
            stage_rt = round((proposal.residence_time_min or 5.0) / max(n_stages, 1), 2)

        # ── Per-stage d_mm (council override or proposal default) ──────────
        stage_id_mm = float(sp.get("d_mm") or proposal.tubing_ID_mm or 1.0)

        # ── New feed pump flow rates ────────────────────────────────────────
        # A physical feed is created once, at its declared introduction stage.
        active_feeds = [
            f for f in stage.feed_streams
            if f.delivery_mode != "carried_from_previous"
            and _label_injection_stage.get(
                (f.stream_label or "").upper(),
                int(f.introduction_stage or sn),
            ) == sn
        ]

        stage_pump_ids: list[str] = []

        if sn == 1:
            # Only use Chief-derived rates if every active liquid feed is covered.
            active_liquid_labels = {
                (f.stream_label or "").upper()
                for f in active_feeds
                if not _stream_is_gas(f)
            }
            if proposal_rate_by_label and active_liquid_labels.issubset(
                proposal_rate_by_label.keys()
            ):
                new_feed_qs = [
                    0.0 if _stream_is_gas(f)
                    else proposal_rate_by_label[(f.stream_label or "").upper()]
                    for f in active_feeds
                ]
                liquid_sum = sum(q for f, q in zip(active_feeds, new_feed_qs) if not _stream_is_gas(f))
                # ``proposal.flow_rate_mL_min`` is the final outlet flow. It is
                # also the Stage 1 flow only when every serialized liquid feed
                # enters Stage 1. If a feed is injected downstream, scaling the
                # Stage 1 subset to the final total double-counts that feed in
                # all later reactors.
                all_liquids_enter_stage_1 = proposal_liquid_labels.issubset(
                    active_liquid_labels
                )
                if (
                    all_liquids_enter_stage_1
                    and total_Q > 0
                    and liquid_sum > 0
                    and abs(liquid_sum - total_Q) / total_Q > 0.02
                ):
                    scale = total_Q / liquid_sum
                    new_feed_qs = [
                        q if _stream_is_gas(f) else round(q * scale, 4)
                        for f, q in zip(active_feeds, new_feed_qs)
                    ]
            else:
                new_feed_qs = _stoich_flow_rates(active_feeds, total_Q)
        else:
            # Frozen proposal stream rates are authoritative. Derive a rate
            # from stoichiometry only when the proposal has no rate for this
            # downstream feed.
            new_feed_qs = []
            for feed in active_feeds:
                if _stream_is_gas(feed):
                    new_feed_qs.append(0.0)
                    continue
                label = (feed.stream_label or "").upper()
                if label in proposal_rate_by_label:
                    new_feed_qs.append(proposal_rate_by_label[label])
                    continue
                equiv = max(getattr(feed, "molar_equiv", 1.0) or 1.0, 1e-9)
                conc  = max(getattr(feed, "concentration_M", None) or C_reference, 1e-9)
                q = round(Q_reference * equiv * C_reference / conc, 4)
                new_feed_qs.append(q)

        for feed, q in zip(active_feeds, new_feed_qs):
            char = feed.stream_label or next_pump_char()
            contents = feed.reagents if feed.reagents else []
            matched_proposal_stream = next(
                (
                    item for item in proposal.streams or []
                    if (item.stream_label or "").upper() == (char or "").upper()
                ),
                None,
            )
            matched_gas = _proposal_gas_for_feed(feed)
            if _stream_is_gas(feed) and getattr(matched_gas, "stream_label", None):
                char = matched_gas.stream_label
                contents = list(getattr(matched_gas, "contents", None) or contents)
            pump_role = (
                getattr(matched_gas, "pump_role", None)
                if _stream_is_gas(feed) and matched_gas is not None
                else None
            ) or feed.reasoning or f"Stage {sn} feed"
            gas_sccm = getattr(matched_gas, "gas_flow_sccm", None)
            gas_actual = getattr(matched_gas, "gas_flow_actual_mL_min", None)
            if _stream_is_gas(feed) and not gas_sccm:
                gas_sccm = (proposal.multiphase_metrics or {}).get("gas_flow_sccm")
                gas_actual = (proposal.multiphase_metrics or {}).get("gas_flow_actual_mL_min")
                q = gas_actual if gas_actual is not None else q
            pid = _add_pump(
                ops, char,
                pump_role,
                contents, stage.solvent, q,
                pump_role,
                is_gas=_stream_is_gas(feed),
                gas_flow_sccm=gas_sccm,
                gas_flow_actual_mL_min=gas_actual,
                molar_equiv=getattr(matched_proposal_stream, "molar_equiv", None),
                inventory_equipment_id=(
                    matched_proposal_stream.pump_equipment_id
                    if matched_proposal_stream is not None
                    else None
                ),
            )
            stage_pump_ids.append(pid)

        # Track reference Q/C from stage 1 for use in stages 2+
        if sn == 1 and active_feeds:
            liquid_feeds = [f for f in active_feeds if not _stream_is_gas(f)]
            if liquid_feeds:
                ref_feed = liquid_feeds[0]
                Q_reference = new_feed_qs[active_feeds.index(ref_feed)]
                C_reference = getattr(ref_feed, "concentration_M", None) or C_reference

        # ── Q entering this reactor ─────────────────────────────────────────
        Q_new_feeds = sum(
            q for feed, q in zip(active_feeds, new_feed_qs)
            if q > 0 and not _stream_is_gas(feed)
        )
        Q_inlet = float(sp["Q_liquid_mL_min"]) if sp.get("Q_liquid_mL_min") is not None else round(Q_prev_outlet + Q_new_feeds, 6)

        # ── Reactor volume V_R = τ_i × Q_inlet_i ───────────────────────────
        gas_feeds_active = [f for f in active_feeds if _stream_is_gas(f)]
        gas_holdup = 0.0
        if gas_feeds_active and proposal.multiphase_metrics:
            gas_holdup = float(proposal.multiphase_metrics.get("gas_holdup") or 0.0)
        stage_gas_actual = sum(
            _safe_float(
                getattr(
                    _proposal_gas_for_feed(feed),
                    "gas_flow_actual_mL_min",
                    0.0,
                )
            )
            for feed in gas_feeds_active
        )
        stage_gas_sccm = sum(
            _safe_float(
                getattr(
                    _proposal_gas_for_feed(feed),
                    "gas_flow_sccm",
                    0.0,
                )
            )
            for feed in gas_feeds_active
        )
        # Finalized stage balances include feeds carried from earlier reactors;
        # counting only this stage's new gas pumps would drop that gas at export.
        if sp.get("Q_gas_actual_mL_min") is not None:
            stage_gas_actual = float(sp["Q_gas_actual_mL_min"])
        if sp.get("Q_gas_sccm") is not None:
            stage_gas_sccm = float(sp["Q_gas_sccm"])
        stage_residence_basis = "liquid-only"
        stage_inventory_volume = _safe_float(
            sp.get("reactor_volume_mL") or sp.get("V_R_mL")
        )
        if stage_inventory_volume > 0 or use_inventory_components:
            stage_vol = round(
                stage_inventory_volume
                if stage_inventory_volume > 0
                else selected_components[sn - 1],
                4,
            )
            liquid_stage_vol = stage_vol * max(1.0 - gas_holdup, 0.0)
            stage_residence_basis = str(
                sp.get("residence_time_basis") or "nominal liquid contact time"
            )
            stage_basis_code = normalize_residence_time_basis(
                stage_residence_basis
            )
            if stage_basis_code == INLET_STP_BASIS:
                stage_rt = round(
                    _safe_float(sp.get("residence_time_inlet_min"))
                    or stage_vol / max(Q_inlet + stage_gas_sccm, 1e-9),
                    2,
                )
            elif stage_basis_code == IN_CHANNEL_BASIS:
                stage_rt = round(
                    _safe_float(sp.get("residence_time_in_channel_min"))
                    or stage_vol / max(Q_inlet + stage_gas_actual, 1e-9),
                    2,
                )
            else:
                # The canonical primary reaction time is liquid-contact time.
                # STP and pressure-corrected gas-inclusive values remain
                # separate diagnostics and must not silently replace it.
                stage_residence_basis = (
                    stage_residence_basis
                    if stage_basis_code == LIQUID_ONLY_BASIS
                    else "nominal liquid contact time"
                )
                stage_rt = round(
                    _safe_float(sp.get("residence_time_min"))
                    or stage_vol / max(Q_inlet, 1e-9),
                    2,
                )
        else:
            liquid_stage_vol = stage_rt * Q_inlet
            stage_vol = round(
                liquid_stage_vol / max(1.0 - gas_holdup, 1e-9),
                4,
            )
            if gas_feeds_active:
                stage_residence_basis = "in-channel pressure-corrected"
        stage_length = (
            round((stage_vol * 1e-6) / (math.pi * (stage_id_mm * 5e-4) ** 2), 2)
            if stage_vol and stage_id_mm else None
        )

        # ── Mixer: combine new feeds + previous stage outlet ───────────────
        mixer_id = f"{prefix}_mixer"
        mixer_inputs = stage_pump_ids[:]
        if prev_op:
            mixer_inputs.append(prev_op)

        if len(mixer_inputs) > 1:
            ops.append(UnitOperation(
                op_id=mixer_id, op_type="mixer",
                label=f"Mixer — Stage {sn}",
                parameters={"type": "T-mixer", "material": "not specified"},
                required=True,
                rationale=f"Combine feeds for {stage.stage_name}",
            ))
            for mid in mixer_inputs:
                _connect(ops, streams, sc, mid, mixer_id, "")
            prev_op = mixer_id
        elif len(mixer_inputs) == 1:
            prev_op = mixer_inputs[0]

        if is_quench_stage:
            Q_prev_outlet = Q_inlet
            logger.info(
                "Skipping reactor for stage %s (%s) — classified as inline quench/workup.",
                sn, stage.stage_name,
            )
            continue

        total_reactor_volume_mL += stage_vol

        # ── Pre-stage: deoxygenation if needed ─────────────────────────────
        if stage.deoxygenation_required and not _uses_offline_deoxygenation(proposal):
            deoxy_id = f"{prefix}_deoxy"
            ops.append(UnitOperation(
                op_id=deoxy_id, op_type="deoxygenation_unit",
                label=f"Degas — Stage {sn}",
                parameters={"method": "N2 sparging"},
                required=True,
                rationale=f"O2-sensitive: {stage.stage_name}",
            ))
            _connect(ops, streams, sc, prev_op, deoxy_id)
            prev_op = deoxy_id

        # ── Reactor for this stage ─────────────────────────────────────────
        reactor_id = f"{prefix}_reactor"
        rtype = stage.reactor_type or "coil"
        op_type_map = {
            "coil": "coil_reactor", "packed_bed": "packed_bed_reactor",
            "chip": "chip_reactor", "CSTR": "coil_reactor",
        }
        is_photo = stage.requires_light
        mat = str(
            sp.get("material")
            or (proposal.inventory_selection or {}).get("material")
            or proposal.tubing_material
            or "not specified"
        )
        stage_temperature_C = float(
            sp.get("temperature_C")
            or proposal.temperature_C
            or stage.temperature_C
            or 25.0
        )
        stage_wavelength_nm = (
            sp.get("wavelength_nm")
            or proposal.wavelength_nm
            or stage.wavelength_nm
        )
        rlabel = f"{'Photo' if is_photo else ''}{rtype.replace('_', ' ').title()} — {stage.stage_name}"

        ops.append(UnitOperation(
            op_id=reactor_id,
            op_type=op_type_map.get(rtype, "coil_reactor"),
            label=rlabel,
            parameters={
                "material": mat,
                "ID_mm": stage_id_mm,
                "volume_mL": stage_vol,
                "liquid_holdup_volume_mL": round(liquid_stage_vol, 4),
                "gas_holdup": round(gas_holdup, 4),
                "Q_inlet_mL_min": Q_inlet,
                "Q_liquid_mL_min": Q_inlet,
                "Q_gas_actual_mL_min": (
                    stage_gas_actual if stage_gas_actual > 0 else None
                ),
                "Q_gas_sccm": (
                    stage_gas_sccm if stage_gas_sccm > 0 else None
                ),
                "temperature_C": stage_temperature_C,
                "wavelength_nm": stage_wavelength_nm if is_photo else None,
                "residence_time_min": stage_rt,
                "residence_time_basis": stage_residence_basis,
                "reactor_type": rtype,
                "inventory_equipment_id": sp.get("reactor_equipment_id"),
                "inventory_equipment_name": sp.get("reactor_name"),
                "residence_time_inlet_min": sp.get("residence_time_inlet_min"),
                "residence_time_in_channel_min": sp.get("residence_time_in_channel_min"),
            },
            required=True,
            rationale=f"Reactor for {stage.stage_name} — τ={stage_rt} min, Q_inlet={Q_inlet} mL/min, V_R={stage_vol} mL",
        ))
        if not prev_op:
            raise ValueError(
                f"Topology invariant failed: reaction stage {sn} has no material inlet. "
                "Chemistry reconciliation must provide an accepted liquid feed."
            )
        _connect(ops, streams, sc, prev_op, reactor_id)
        prev_op = reactor_id

        if inventory and inventory.temperature_controllers:
            ops.append(UnitOperation(
                op_id=f"{prefix}_heater",
                op_type="heater",
                label=f"Temperature Control — Stage {sn}",
                parameters={"temperature_C": stage_temperature_C},
                required=True,
                rationale=f"Active temperature control for {stage.stage_name}",
            ))

        # Q leaving this reactor = Q_inlet (incompressible flow)
        Q_prev_outlet = Q_inlet

        # LED if photochemical stage
        if is_photo and stage_wavelength_nm:
            led_id = f"{prefix}_led"
            ops.append(UnitOperation(
                op_id=led_id, op_type="led_module",
                label=f"LED {stage_wavelength_nm:.0f} nm",
                parameters={
                    "wavelength_nm": stage_wavelength_nm,
                    "temperature_C": stage_temperature_C,
                    "inventory_equipment_id": sp.get("light_equipment_id"),
                    "inventory_equipment_name": sp.get("light_name"),
                },
                required=True, rationale=f"Light for {stage.stage_name}",
            ))

        # ── Post-stage action (quench, filter, solvent switch, separator, BPR) ─
        if stage.post_stage_action:
            action = stage.post_stage_action.lower()
            post_id = f"{prefix}_post"
            n_ops_before = len(ops)  # track whether a node was actually added

            if "filter" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="inline_filter",
                    label="Filter",
                    parameters={"pore_size_um": 10},
                    required=True,
                    rationale=stage.post_stage_reasoning or stage.post_stage_action,
                ))
            elif "quench" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="quench_mixer",
                    label="Quench",
                    parameters={"reagent": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            elif "solvent" in action or "switch" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label="Solvent Switch",
                    parameters={"type": "solvent_switch"},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            elif any(k in action for k in ("segment", "gas-liquid", "gas_liquid",
                                            "separator", "l-l", "liquid-liquid",
                                            "extraction", "phase sep")):
                # Only insert a separator when TWO IMMISCIBLE PHASES are genuinely
                # present AND the reasoning is explicit. Prevents hallucinated
                # separators for atmosphere changes or gas switching.
                reasoning = stage.post_stage_reasoning or ""
                if reasoning:
                    ops.append(UnitOperation(
                        op_id=post_id, op_type="liq_liq_extraction",
                        label="L-L Separator",
                        parameters={},
                        required=True,
                        rationale=reasoning,
                    ))
                else:
                    logger.warning(
                        "Skipping L-L separator for stage %d — "
                        "no post_stage_reasoning provided (not chemistry-justified).",
                        stage.stage_number,
                    )
            elif any(k in action for k in ("bpr", "back pressure", "back-pressure",
                                            "backpressure")):
                ops.append(UnitOperation(
                    op_id=post_id, op_type="bpr",
                    label="BPR",
                    parameters={"pressure_bar": proposal.BPR_bar or 5},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            else:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label=stage.post_stage_action[:25],
                    parameters={"details": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))

            # Only wire into the stream graph if a node was actually appended
            if len(ops) > n_ops_before:
                _connect(ops, streams, sc, prev_op, post_id)
                prev_op = post_id

    # ── Final BPR ─────────────────────────────────────────────────────────
    # Gas-liquid proposals must always retain a BPR even if the final LLM/Chief
    # text omitted BPR_bar. The calculator uses this for gas solubility and
    # controlled degassing; the topology must not silently drop it.
    bpr_already_in_ops = any(o.op_type == "bpr" for o in ops)
    bpr_pressure = float(proposal.BPR_bar or 0.0)
    if any(_stream_is_gas(stream) for stream in proposal.streams or []) and bpr_pressure <= 0:
        bpr_pressure = TOPOLOGY_DEFAULT_GAS_LIQUID_BPR_BAR
    if bpr_pressure > 0 and not bpr_already_in_ops:
        ops.append(UnitOperation(op_id="bpr_final", op_type="bpr",
            label="BPR", parameters={"pressure_bar": bpr_pressure},
            required=True, rationale="Back-pressure regulation"))
        _connect(ops, streams, sc, prev_op, "bpr_final")
        prev_op = "bpr_final"

    # ── Collector ──────────────────────────────────────────────────────────
    ops.append(UnitOperation(op_id="collector_1", op_type="collector",
        label="Product Collection", parameters={}, required=True, rationale="Outlet"))
    _connect(ops, streams, sc, prev_op, "collector_1")

    _enforce_gas_delivery_hardware(ops)

    # PID description
    pid_parts = [o.label for o in ops if o.op_type != "led_module"]
    # Q_prev_outlet is the true final outlet Q — Stage 1 Q plus any additions
    # from downstream feeds (e.g. quench). Use it for total_flow_rate so the
    # process diagram and summary agree on the outlet stream.
    final_outlet_Q = Q_prev_outlet or proposal.flow_rate_mL_min
    return ProcessTopology(
        topology_id="translate_multistep",
        unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=final_outlet_Q,
        residence_time_min=proposal.residence_time_min,
        reactor_volume_mL=round(total_reactor_volume_mL, 4) or proposal.reactor_volume_mL,
        pid_description=" → ".join(pid_parts),
        topology_confidence=proposal.confidence,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _topology_matches_serialized_proposal(
    topology: ProcessTopology,
    proposal: FlowProposal,
) -> bool:
    """Verify that topology nodes are a faithful projection of final values."""

    def close(left, right, *, absolute=0.02, relative=0.01):
        left = _safe_float(left)
        right = _safe_float(right)
        if left <= 0 or right <= 0:
            return False
        return abs(left - right) <= max(absolute, relative * max(left, right))

    expected_volume = float(proposal.reactor_volume_mL or 0.0)
    if not close(topology.reactor_volume_mL, expected_volume):
        return False
    if not close(topology.residence_time_min, proposal.residence_time_min):
        return False
    if proposal.flow_rate_mL_min and not close(
        topology.total_flow_rate_mL_min,
        proposal.flow_rate_mL_min,
        absolute=0.0001,
        relative=0.02,
    ):
        return False

    reactor_types = {
        "coil_reactor", "reactor", "heated_coil", "photoreactor",
        "chip_reactor", "packed_bed", "packed_bed_reactor",
    }
    reactors = [
        operation
        for operation in topology.unit_operations
        if operation.op_type in reactor_types
    ]
    if not reactors:
        return False
    reactor_volume = sum(
        _safe_float((operation.parameters or {}).get("volume_mL"))
        for operation in reactors
    )
    if not close(reactor_volume, expected_volume):
        return False

    stage_parameters = sorted(
        [item for item in proposal.stage_parameters or [] if isinstance(item, dict)],
        key=lambda item: _safe_float(item.get("stage_number")),
    )
    if stage_parameters:
        if len(stage_parameters) != len(reactors):
            return False
        for stage, operation in zip(stage_parameters, reactors):
            parameters = operation.parameters or {}
            if not close(
                parameters.get("volume_mL"),
                stage.get("reactor_volume_mL") or stage.get("V_R_mL"),
            ):
                return False
            for field in (
                "residence_time_inlet_min",
                "residence_time_in_channel_min",
            ):
                expected = stage.get(field)
                if expected is not None and not close(
                    parameters.get(field), expected, absolute=0.05, relative=0.01
                ):
                    return False
        inlet_total = sum(
            _safe_float(item.get("residence_time_inlet_min") or item.get("residence_time_min"))
            for item in stage_parameters
        )
        channel_total = sum(
            _safe_float(item.get("residence_time_in_channel_min") or item.get("residence_time_min"))
            for item in stage_parameters
        )
        if not close(
            inlet_total,
            proposal.residence_time_inlet_min or proposal.residence_time_min,
            absolute=0.05,
        ):
            return False
        if not close(
            channel_total,
            proposal.residence_time_in_channel_min or proposal.residence_time_min,
            absolute=0.05,
        ):
            return False
    else:
        node_tau_total = sum(
            _safe_float((operation.parameters or {}).get("residence_time_min"))
            for operation in reactors
        )
        if not close(node_tau_total, proposal.residence_time_min, absolute=0.05):
            return False
    return True


def _store_process_topology(result: dict, topology: ProcessTopology) -> None:
    """Persist topology diagnostics without mutating the validated proposal."""

    result["process_topology"] = topology.model_dump()
    result.setdefault("design_calculations", {})[
        "topology_total_reactor_volume_mL"
    ] = topology.reactor_volume_mL


def _sync_stage_hardware_from_compiled_topology(
    proposal: FlowProposal,
    topology: ProcessTopology,
) -> bool:
    """Replace model-written stage hardware IDs with allocator assignments."""

    if not proposal.stage_parameters:
        return False
    operations = {operation.op_id: operation for operation in topology.unit_operations}
    incoming: dict[str, list[str]] = {}
    for stream in topology.streams:
        incoming.setdefault(stream.to_op, []).append(stream.from_op)
    bpr_ids = [
        operation.inventory_item_id
        for operation in topology.unit_operations
        if operation.op_type.lower() == "bpr" and operation.inventory_item_id
    ]
    changed = False
    last_stage = max(
        int(stage.get("stage_number") or index + 1)
        for index, stage in enumerate(proposal.stage_parameters)
    )
    for index, stage in enumerate(proposal.stage_parameters):
        stage_number = int(stage.get("stage_number") or index + 1)
        reactor_id = f"st{stage_number}_reactor"
        reactor = operations.get(reactor_id)
        light = operations.get(f"st{stage_number}_led")
        reachable: list[Any] = []
        stack = list(incoming.get(reactor_id, []))
        seen: set[str] = set()
        while stack:
            operation_id = stack.pop()
            if operation_id in seen:
                continue
            seen.add(operation_id)
            operation = operations.get(operation_id)
            if operation is None:
                continue
            if operation.op_type.lower() in {
                "coil_reactor", "photoreactor", "reactor", "heated_coil",
            }:
                continue
            reachable.append(operation)
            stack.extend(incoming.get(operation_id, []))

        liquid_ids = [
            operation.inventory_item_id
            for operation in reachable
            if operation.op_type.lower() == "pump" and operation.inventory_item_id
        ]
        gas_ids = [
            operation.inventory_item_id
            for operation in reachable
            if operation.op_type.lower() == "mfc" and operation.inventory_item_id
        ]
        mixer = next(
            (operation for operation in reachable if operation.op_type.lower() in {"mixer", "t_mixer", "y_mixer"}),
            None,
        )
        for key in (
            "pump_equipment_id", "pump_equipment_ids", "gas_equipment_id",
            "gas_equipment_ids", "mixer_equipment_id", "BPR_equipment_id",
            "bpr_equipment_id",
        ):
            stage.pop(key, None)
        if reactor and reactor.inventory_item_id:
            stage["reactor_equipment_id"] = reactor.inventory_item_id
        if light and light.inventory_item_id:
            stage["light_equipment_id"] = light.inventory_item_id
        if liquid_ids:
            stage["pump_equipment_ids"] = liquid_ids
            if len(liquid_ids) == 1:
                stage["pump_equipment_id"] = liquid_ids[0]
        if gas_ids:
            stage["gas_equipment_ids"] = gas_ids
            if len(gas_ids) == 1:
                stage["gas_equipment_id"] = gas_ids[0]
        if mixer is not None:
            stage["mixer_assignment_status"] = mixer.assignment_status
            stage["mixer_name"] = mixer.instrument_name or mixer.label
            if mixer.inventory_item_id:
                stage["mixer_equipment_id"] = mixer.inventory_item_id
        if stage_number == last_stage and bpr_ids:
            stage["BPR_equipment_id"] = bpr_ids[0]
        changed = True
    return changed


def _store_blocked_topology(result: dict, disposition) -> None:
    failures = [
        {
            "finding_id": finding.finding_id,
            "message": finding.message,
        }
        for finding in disposition.hard_failures
    ]
    result["svg_path"] = ""
    result["png_path"] = ""
    result["process_topology"] = {
        "topology_id": "blocked",
        "generation_status": "blocked",
        "unit_operations": [],
        "streams": [],
        "pid_description": "",
        "blocking_reasons": failures,
    }


def _inventory_preflight_result(
    *,
    batch_record: BatchRecord,
    chemistry_plan: ChemistryPlan,
    intake: DesignInputPackage | None,
    intake_block: str,
    topology: ProcessTopology,
    preflight: dict,
) -> dict:
    """Return a non-numerical requirements result before downstream design."""

    unresolved = list(preflight.get("unresolved_requirements") or [])
    confirmation_required = preflight.get("status") == "needs_confirmation"
    result = {
        "proposal": {},
        "chemistry_plan": chemistry_plan.model_dump(exclude_none=True),
        "confidence": "NOT_ASSESSED",
        "design_status": (
            "inventory_confirmation_required"
            if confirmation_required
            else "inventory_infeasible"
        ),
        "recommended_disposition": "BLOCK",
        "reported_disposition": "BLOCK",
        "disposition_rationale": (
            "Inventory capability confirmation is required before numerical design."
            if confirmation_required
            else "A required process capability is explicitly unavailable."
        ),
        "design_disposition": {
            "recommended_disposition": "BLOCK",
            "rationale": (
                "Inventory capability confirmation is required before numerical design."
                if confirmation_required
                else "A required process capability is explicitly unavailable."
            ),
            "hard_failures": [
                {
                    "finding_id": item["requirement_id"],
                    "message": item["reason"],
                }
                for item in unresolved
            ],
        },
        "inventory_preflight": preflight,
        "inventory_allocation": {
            "schema_version": "flowpilot_inventory_allocation_v1.0",
            "status": "confirmation_required" if confirmation_required else "incomplete",
            "strict_assignment": True,
            "checks": {"all_required_operations_assigned": False},
            "assignments": [],
            "instrument_manifest": [],
            "unresolved_requirements": unresolved,
        },
        "final_validation": {
            "schema_version": "flowpilot_final_validation_v1.0",
            "status": "blocked",
            "checks": {"topology_capability_preflight_complete": False},
            "unresolved_reasons": [item["requirement_id"] for item in unresolved],
        },
        "process_requirements_topology": topology.model_dump(),
        "diagnostic_topology": topology.model_dump(),
        "process_topology": {
            "topology_id": "blocked_before_numerical_design",
            "generation_status": "blocked",
            "unit_operations": [],
            "streams": [],
            "pid_description": "",
        },
        "svg_path": "",
        "png_path": "",
        "explanation": (
            "The chemistry-derived process topology was checked before retrieval, "
            "engineering calculation, downstream translation, and council. No "
            "numerical run parameters were generated."
        ),
        "batch_record": batch_record.model_dump(exclude_none=True),
    }
    if intake:
        from flora_translate.inventory_resolution import review_inventory
        intake.inventory_review = review_inventory(intake, chemistry_plan)
        intake.ready_for_design = False
        result["intake_package"] = intake.model_dump()
        result["intake_context"] = intake_block
    try:
        artifacts = render_topology_artifacts(
            topology,
            title="REQUIREMENTS TOPOLOGY - NOT EXECUTABLE",
        )
        result["diagnostic_svg_path"] = artifacts.get("svg_path", "")
        result["diagnostic_png_path"] = artifacts.get("png_path", "")
        result["diagnostic_diagram_artifacts"] = {
            key: value for key, value in artifacts.items() if key != "manifest"
        }
        result["diagnostic_diagram_render_manifest"] = artifacts.get("manifest", {})
    except Exception as exc:
        logger.warning("Requirements topology rendering failed: %s", exc)
        result["diagnostic_svg_path"] = ""
        result["diagnostic_png_path"] = ""
        result["diagnostic_diagram_artifacts"] = {
            "render_status": "failed",
            "warnings": [str(exc)],
        }
    result["final_design"] = build_final_design_contract(result)
    return result


@with_runtime_model_routing
def translate(
    batch_input: str | dict,
    inventory_path: str = str(LAB_INVENTORY_PATH),
    intake_package: DesignInputPackage | dict | None = None,
    runtime_options: PipelineRuntimeOptions | dict | None = None,
) -> dict:
    """Full FLORA-Translate pipeline.

    Returns:
        Dict with proposal, chemistry_plan, explanation, safety report,
        council messages, svg_path, png_path.
    """
    runtime = PipelineRuntimeOptions.coerce(runtime_options)
    scientific = runtime.design_policy == "scientific_v2"
    scientific_archive = None
    if scientific:
        from pathlib import Path
        from datetime import datetime
        from uuid import uuid4
        scientific_archive = Path("outputs/scientific_pipeline") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        scientific_archive.mkdir(parents=True, exist_ok=False)
    def scientific_snapshot(name, data):
        if scientific_archive:
            (scientific_archive / (name + ".json")).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    intake = _coerce_intake_package(intake_package)
    intake_block = intake_context_block(intake)
    effective_batch_input = batch_input_from_package(intake) if intake else batch_input
    objectives = (
        runtime.objective_override
        or (intake.objective if intake and intake.objective else "balanced")
    )

    # 1. Parse input
    logger.info("Step 1: Parsing batch input")
    batch_record = parse_batch_input(effective_batch_input)
    logger.info(f"  Parsed: {batch_record.reaction_description[:80]}...")

    # 2. Chemistry Reasoning — Layer 1
    logger.info("Step 2: Chemistry analysis (Layer 1)")
    chemistry_plan = analyze_batch_chemistry(batch_record, intake_package=intake, **({"scientific": True} if scientific else {}))
    scientific_snapshot("upstream", {"batch_record": batch_record.model_dump(), "chemistry_plan": chemistry_plan.model_dump()})
    chemistry_plan, chemistry_reconciliation = reconcile_chemistry_plan(
        batch_record,
        chemistry_plan,
        scientific=scientific,
        hard_constraints={
            "inventory_constraints": intake.inventory_constraints if intake else None,
            "operating_limits": intake.operating_limits if intake else None,
            "runtime_hard_constraints": list(runtime.hard_constraints),
            "gas_introduction_requirement": (intake.engineering_requirements.get("gas", {}) if intake else {}),
        },
    )
    chemistry_plan, intake_requirement_decisions = (
        apply_intake_requirements_to_chemistry_plan(chemistry_plan, intake)
    )
    if scientific:
        chemistry_plan.scientific_context["operating_limits"] = intake.operating_limits if intake else None
        chemistry_plan.scientific_context["hard_constraints"] = list(runtime.hard_constraints)
    if intake_requirement_decisions:
        chemistry_reconciliation["intake_requirement_decisions"] = (
            intake_requirement_decisions
        )
    logger.info(f"  Mechanism: {chemistry_plan.mechanism_type}")
    scientific_snapshot("chemistry_contract", chemistry_plan.model_dump())
    logger.info(f"  Streams: {len(chemistry_plan.stream_logic)}  O2: {chemistry_plan.oxygen_sensitive}")
    logger.info(f"  Upstream mode: {getattr(chemistry_plan, '_upstream_mode', 'full')}")

    # Resolve structural inventory capability before any numerical design or
    # downstream model call. Missing categories are confirmation requests, not
    # fabricated equipment and not proof that the lab lacks the capability.
    inventory = _inventory_from_intake_or_path(intake, inventory_path)
    if inventory.strict_assignment:
        requirements_topology, inventory_preflight = analyze_topology_requirements(
            chemistry_plan, inventory
        )
        logger.info("Step 2b: Inventory topology preflight %s", inventory_preflight["status"])
        if not str(inventory_preflight["status"]).startswith("ready"):
            blocked_result = _inventory_preflight_result(
                batch_record=batch_record,
                chemistry_plan=chemistry_plan,
                intake=intake,
                intake_block=intake_block,
                topology=requirements_topology,
                preflight=inventory_preflight,
            )
            blocked_result["pipeline_runtime"] = runtime.provenance()
            blocked_result["chemistry_reconciliation"] = chemistry_reconciliation
            return blocked_result

    # 3. Plan-aware retrieval — Layer 2
    logger.info("Step 3: Retrieving literature analogies (plan-aware)")
    store = VectorStore()
    retriever = VectorRetriever(store=store)
    raw_analogies = retriever.retrieve(
        batch_record,
        top_k=3,
        chemistry_plan=chemistry_plan,
        exclude_record_ids=set(runtime.exclude_record_ids),
        retrieval_mode=runtime.retrieval_mode,
    )
    analogies = AnalogySelector(records_dir=RECORDS_DIR).select(raw_analogies)
    logger.info(f"  Found {len(analogies)} analogies")

    # 3b. Pre-compute engineering calculations (9-step design calculator)
    logger.info("Step 3b: Running 9-step design calculator")
    from flora_translate.design_calculator import DesignCalculator
    calculations = DesignCalculator().run(
        batch_record,
        chemistry_plan=chemistry_plan,
        inventory=inventory,
        analogies=analogies,
    )
    logger.info(
        f"  τ = {calculations.residence_time_min:.1f} min "
        f"(range {calculations.residence_time_range_min}), "
        f"method = {calculations.kinetics_method}, "
        f"Re = {calculations.reynolds_number:.0f}, "
        f"Da = {calculations.damkohler_mass}, "
        f"ΔP = {calculations.pressure_drop_bar:.4f} bar, "
        f"BPR = {'yes' if calculations.bpr_required else 'no'}"
    )

    # 3c. Design space grid search — enumerate feasible (τ, d, Q) candidates
    logger.info("Step 3c: Design space grid search")
    from flora_translate.engine.design_space import (
        DesignSpaceSearch,
        candidates_to_dicts,
        get_council_starting_point,
        feasible_candidates_as_council_seeds,
    )
    design_candidates = [] if scientific else DesignSpaceSearch().run(
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        calculations=calculations,
        inventory=inventory,
        reaction_class=chemistry_plan.reaction_class if chemistry_plan else "default",
    )
    logger.info(f"  Design space: {len(design_candidates)} candidates, "
                f"{sum(1 for c in design_candidates if c.feasible)} feasible")

    # Use top design space candidate as council starting point (replaces LLM guess)
    _top_candidate = get_council_starting_point(design_candidates)
    if _top_candidate:
        logger.info(f"  Top candidate: τ={_top_candidate.tau_min}min, "
                    f"Q={_top_candidate.Q_mL_min}mL/min, d={_top_candidate.d_mm}mm, "
                    f"L={_top_candidate.L_m}m, score={_top_candidate.score:.3f}")

    # 4. Generate flow proposal
    logger.info("Step 4: Generating flow proposal via LLM")
    system_prompt, user_prompt = TranslationPromptBuilder().build(
        batch_record, analogies, chemistry_plan=chemistry_plan,
        calculations=calculations, inventory=inventory,
        intake_package=intake,
    )
    if scientific:
        user_prompt += "\n\nSCIENTIFIC POLICY (overrides generic time heuristics):\n" + json.dumps(chemistry_plan.scientific_context)
        user_prompt += "\nPreserve each stage's reagent addition order. Propose feed concentrations and stoichiometry from the source; do not claim conversion or intensification. Hardware-bound joint candidates are calculated and reviewed next."
    proposal = TranslationLLM().generate(system_prompt, user_prompt)
    scientific_snapshot("translation", {"system_prompt": system_prompt, "user_prompt": user_prompt, "proposal": proposal.model_dump()})
    logger.info(f"  Proposal: {proposal.residence_time_min}min, {proposal.reactor_type}, "
                f"{len(proposal.streams)} streams")

    # Feed the council a pressure-feasible candidate. Otherwise every council
    # member spends its effort scoring an unavailable or sub-floor BPR value,
    # and the skeptic can disqualify the entire matrix before the deterministic
    # inventory pass gets a chance to select the real cartridge.
    if not scientific and any(_stream_is_gas(stream) for stream in proposal.streams or []):
        pressure_settings = available_pressure_settings(inventory)
        eligible_pressure = [value for value in pressure_settings if value >= 3.0]
        if eligible_pressure and float(proposal.BPR_bar or 0.0) < 3.0:
            previous_bpr = float(proposal.BPR_bar or 0.0)
            proposal.BPR_bar = min(eligible_pressure)
            proposal.BPR_basis = "gauge"
            proposal.pressure_absolute_bar = round(proposal.BPR_bar + 1.01325, 6)
            logger.info(
                "  Pre-council inventory pressure reconciliation: %.3g -> %.3g bar gauge",
                previous_bpr,
                proposal.BPR_bar,
            )

    # The grid winner is a council seed, not an authority allowed to overwrite
    # a chemistry-supported proposal. Its soft score deliberately rewards
    # productivity and can therefore select the shortest feasible screen even
    # when the calculator and translation independently support a longer
    # residence time. Preserve the proposal here; downstream deterministic
    # realization will bind it to discrete inventory and close V/Q/tau.
    if _top_candidate:
        logger.info(
            "  Design-space winner retained as council seed; proposal geometry "
            "preserved for deliberation"
        )

    # 5. ENGINE deliberation council — Layer 3
    logger.info("Step 5: Multi-agent deliberation council (ENGINE)")
    pre_council_proposal = proposal.model_dump()  # snapshot before council modifies it
    from copy import deepcopy
    from dataclasses import asdict

    pre_council_calculations = deepcopy(asdict(calculations))
    # Pre-package Design Space feasible candidates as council seeds. If the
    # Council Designer's LLM-guided sampling finds 0 feasible points, the
    # council falls back to these — preventing silent council-skip when the
    # sampling envelope happens to miss the design-space sweet spot.
    _ds_seeds = feasible_candidates_as_council_seeds(
        design_candidates,
        BPR_bar=proposal.BPR_bar or 0.0,
        tubing_material=proposal.tubing_material or "FEP",
        concentration_M=proposal.concentration_M or 0.1,
        temperature_C=proposal.temperature_C or 25.0,
        batch_time_min=(batch_record.reaction_time_h or 0.0) * 60.0 if batch_record else None,
        n_max=6,
    )
    design_candidate, calculations = CouncilV4().run(
        proposal, batch_record, analogies, inventory,
        chemistry_plan=chemistry_plan, calculations=calculations,
        objectives=objectives,
        design_space_seed_candidates=_ds_seeds,
        intake_package=intake,
        candidate_budget=runtime.candidate_budget,
        benchmark_recorder=runtime.benchmark_recorder,
        benchmark_strict_scoring=runtime.benchmark_strict_scoring,
        benchmark_scoring_batch_size=runtime.benchmark_scoring_batch_size,
        benchmark_strong_revision_mode=runtime.benchmark_strong_revision_mode,
        benchmark_branching_revision_mode=(
            runtime.benchmark_branching_revision_mode
        ),
        benchmark_max_descendants_per_candidate=(
            runtime.benchmark_max_descendants_per_candidate
        ),
        benchmark_max_total_revised_candidates=(
            runtime.benchmark_max_total_revised_candidates
        ),
        execution_config=runtime.council_execution,
        council_backflow_review=runtime.council_backflow_review,
        council_physics_profile=runtime.council_physics_profile,
        hard_constraints=merged_hard_constraints(runtime.hard_constraints, intake.operating_limits if intake else None),
    )

    # 6. Format output
    if scientific and design_candidate.chemistry_plan is not None:
        chemistry_plan = design_candidate.chemistry_plan
    logger.info("Step 6: Formatting output")
    result = OutputFormatter().format(design_candidate, analogies)
    result["engineering_history"] = {
        "schema_version": "flowpilot_engineering_history_v1",
        "before_council": {
            "source": "calculator and translation proposal before CouncilV4.run",
            "proposal": deepcopy(pre_council_proposal),
            "calculations": pre_council_calculations,
        },
        "after_council": {
            "source": "CouncilV4 selection before deterministic inventory realization",
            "proposal": deepcopy(result.get("proposal") or {}),
            "calculations": deepcopy(asdict(calculations)),
        },
    }
    result["chemistry_plan"] = chemistry_plan.model_dump(exclude_none=True)
    result["chemistry_reconciliation"] = chemistry_reconciliation
    result["batch_record"] = batch_record.model_dump(exclude_none=True)
    result["inventory_snapshot"] = inventory.model_dump(exclude_none=True)
    if intake:
        result["intake_package"] = intake.model_dump()
        result["intake_context"] = intake_block

    # Attach 9-step design calculations for Streamlit rendering
    from dataclasses import asdict
    result["design_calculations"] = asdict(calculations)
    # Scientific candidates are already jointly realized before council review.
    # Legacy lumped recalculations cannot rewrite that frozen selection.
    inventory_reviewed = bool(design_candidate.proposal.inventory_constraints.get("council_reviewed_parameters"))
    if not scientific and not inventory_reviewed:
        _reconcile_final_bpr(result, design_candidate, inventory)
        _sync_final_stream_flowrates(result, design_candidate)

    # If the user included measured flow experiments in the prompt, apply the
    # deterministic closed-loop calibration before building the final topology.
    # This prevents unsupported first-pass intensification from overriding
    # observed conversion/yield data.
    try:
        if intake:
            raw_input_text = "\n\n".join(
                part
                for part in [
                    intake.raw_protocol,
                    historical_text_from_package(intake),
                ]
                if part
            )
        else:
            raw_input_text = batch_input if isinstance(batch_input, str) else json.dumps(batch_input, default=str)
        from flora_translate.experiment_loop import (
            extract_experiments_from_text,
            refine_from_experimental_campaign,
        )
        experiments = extract_experiments_from_text(raw_input_text)
        if len(experiments) >= 2 and not scientific:
            logger.info(
                "Step 6b: Applying evidence-calibrated closed-loop refinement from %d experiments",
                len(experiments),
            )
            closed_loop = refine_from_experimental_campaign(
                result,
                experiments,
                target_yield_pct=75.0,
                target_conversion_pct=75.0,
                target_selectivity_pct=85.0,
            )
            result = closed_loop.refined_result
            design_candidate.proposal = FlowProposal(**result["proposal"])
            _sync_final_stream_flowrates(result, design_candidate)
    except Exception as exc:
        logger.warning("Evidence-calibrated closed-loop refinement skipped: %s", exc)

    # Atomically enforce inventory, gas basis, and geometry after all model and
    # campaign refinements.
    final_validation = None
    try:
        if scientific:
            inventory_proposal = design_candidate.proposal
            selected = next(c for c in inventory_proposal.scientific_design["candidates"]
                            if c["candidate_id"] == inventory_proposal.scientific_design["selected_candidate_id"])
            final_validation = deepcopy(selected["validation"])
        elif inventory_reviewed:
            inventory_proposal = design_candidate.proposal
            final_validation = deepcopy(inventory_proposal.inventory_constraints["council_reviewed_validation"])
        else:
            inventory_proposal, calculations, final_validation = finalize_design(
                design_candidate.proposal,
                batch_record=batch_record,
                chemistry_plan=chemistry_plan,
                analogies=analogies,
                inventory=inventory,
            )
        design_candidate.proposal = inventory_proposal
        result["proposal"] = design_candidate.proposal.model_dump()
        result["design_calculations"] = asdict(calculations)
        result["inventory_enforcement"] = final_validation.get(
            "inventory_enforcement", {}
        )
        result["final_validation"] = final_validation
        result["explanation"] = (
            (result.get("explanation") or "").rstrip()
            + "\n\nFinal engineering validation: "
            + final_validation["status"]
            + "."
        ).strip()
        logger.info(
            "Step 6c: Final engineering validation %s",
            final_validation["status"],
        )
        if not scientific and not inventory_reviewed:
            _reconcile_final_bpr(result, design_candidate, inventory)
            _sync_final_stream_flowrates(result, design_candidate)
    except Exception as exc:
        logger.warning("Final engineering validation skipped: %s", exc)

    # A global single-reactor council candidate is not a valid representation
    # of a multistage process. Resolve each stage to exact inventory IDs and
    # recompute stagewise residence times before topology generation.
    multistage_inventory_report = {"applied": False}
    try:
        reconciled_proposal, multistage_inventory_report = (
            reconcile_multistage_inventory(
                design_candidate.proposal,
                chemistry_plan,
                inventory,
                operating_limits=(intake.operating_limits if intake else None),
            )
        )
        if multistage_inventory_report.get("applied"):
            design_candidate.proposal = reconciled_proposal
            result["proposal"] = reconciled_proposal.model_dump()
            result["multistage_inventory_plan"] = multistage_inventory_report
            result.setdefault("design_calculations", {}).update(
                {
                    "calculation_mode": "stagewise_inventory_closed",
                    "reactor_volume_mL": reconciled_proposal.reactor_volume_mL,
                    "residence_time_min": reconciled_proposal.residence_time_min,
                    "residence_time_inlet_min": (
                        reconciled_proposal.residence_time_inlet_min
                    ),
                    "residence_time_in_channel_min": (
                        reconciled_proposal.residence_time_in_channel_min
                    ),
                    "stage_calculations": multistage_inventory_report.get(
                        "stage_parameters", []
                    ),
                }
            )
            if final_validation is not None:
                checks = final_validation.setdefault("checks", {})
                stage_checks = multistage_inventory_report.get("checks", {})
                stage_complete = multistage_inventory_report.get("status") == "complete"
                checks["reactor_inventory_match"] = bool(
                    stage_checks.get("all_stage_reactors_resolved")
                )
                checks["geometry_closure"] = bool(
                    stage_checks.get("stage_geometry_closed")
                )
                checks["calculation_matches_serialized_design"] = stage_complete
                checks["multistage_stage_inventory_complete"] = stage_complete
                unresolved = [name for name, passed in checks.items() if not passed]
                final_validation["unresolved_reasons"] = unresolved
                final_validation["status"] = (
                    "ready" if not unresolved else "screen_required"
                )
                final_validation["calculation_mode"] = "stagewise_inventory_closed"
                final_validation["multistage_inventory_plan"] = (
                    multistage_inventory_report
                )
                result["final_validation"] = final_validation
    except Exception as exc:
        logger.warning("Multistage inventory reconciliation failed: %s", exc)

    # Final authoritative realization. Everything above this point may propose
    # or refine values; this pass jointly binds feeds, pumps/MFCs, pressure,
    # reactors, and stage timing. No model is allowed to mutate run parameters
    # after this point.
    try:
        realized_proposal, realization_report, final_validation = (
            realize_executable_design(
                design_candidate.proposal,
                batch_record=batch_record,
                chemistry_plan=chemistry_plan,
                inventory=inventory,
                hard_constraints=merged_hard_constraints(
                    runtime.hard_constraints,
                    intake.operating_limits if intake else None,
                ),
                operating_limits=(intake.operating_limits if intake else None),
            )
        )
        design_candidate.proposal = realized_proposal
        result["proposal"] = realized_proposal.model_dump()
        result["design_realization"] = realization_report
        result["final_validation"] = final_validation
        if realization_report.get("multistage", {}).get("applied"):
            result["multistage_inventory_plan"] = realization_report["multistage"]

        # Recompute engineering annotations once from the realized candidate,
        # then overwrite its canonical V/Q/tau fields from the authoritative
        # proposal so diagnostic calculations cannot become a second design.
        from flora_translate.design_calculator import DesignCalculator

        realized_calculations = DesignCalculator().run(
            batch_record,
            chemistry_plan=chemistry_plan,
            proposal=realized_proposal,
            inventory=inventory,
            analogies=analogies,
            target_flow_rate_mL_min=realized_proposal.flow_rate_mL_min,
            target_tubing_ID_mm=realized_proposal.tubing_ID_mm,
            target_residence_time_min=realized_proposal.residence_time_min,
        )
        calculation_payload = asdict(realized_calculations)
        calculation_payload.update(
            {
                "flow_rate_mL_min": realized_proposal.flow_rate_mL_min,
                "liquid_flow_rate_mL_min": realized_proposal.flow_rate_mL_min,
                "reactor_volume_mL": realized_proposal.reactor_volume_mL,
                "tubing_ID_mm": realized_proposal.tubing_ID_mm,
                "residence_time_min": realized_proposal.residence_time_min,
                "residence_time_inlet_min": realized_proposal.residence_time_inlet_min,
                "residence_time_in_channel_min": realized_proposal.residence_time_in_channel_min,
                "bpr_pressure_bar": realized_proposal.BPR_bar,
                "stage_calculations": list(realized_proposal.stage_parameters or []),
                "calculation_mode": "deterministic_post_council_realization",
            }
        )
        calculation_payload.update(realized_proposal.multiphase_metrics or {})
        result["design_calculations"] = calculation_payload
        from flora_translate.final_engineering import calculate_final_stages

        result["final_stage_engineering"] = calculate_final_stages(
            realized_proposal, batch_record, chemistry_plan, inventory, analogies,
        )
        result["design_calculations"]["annotation_scope"] = (
            "lumped diagnostic; final engineering is in final_stage_engineering"
        )
        if scientific:
            from flora_translate.engine.council_v4.scientific import signature
            evidence = realized_proposal.scientific_design
            unchanged = signature(realized_proposal) == evidence["selected_signature"]
            final_validation["checks"]["scientific_selected_design_preserved"] = unchanged
            result["scientific_assessment"] = evidence
            result["scientific_assessment"]["selected_design_preserved"] = unchanged
            if not unchanged:
                final_validation["status"] = "blocked"
                final_validation.setdefault("unresolved_reasons", []).append("scientific_selected_design_preserved")
        logger.info(
            "Step 6d: Deterministic design realization %s",
            realization_report["status"],
        )
    except Exception as exc:
        logger.exception("Deterministic design realization failed: %s", exc)
        if final_validation is None:
            final_validation = {
                "schema_version": "flowpilot_final_validation_v2.0",
                "status": "blocked",
                "checks": {"design_realization_complete": False},
                "unresolved_reasons": ["design_realization_complete"],
            }
        result["final_validation"] = final_validation

    # Attach design space grid search results
    result["design_space"] = candidates_to_dicts(design_candidates)

    _apply_final_design_guards(result, design_candidate)

    # Build the abstract process requirements once, then compile every
    # executable unit operation against physical inventory before disposition.
    compiled_topology = None
    try:
        abstract_topology = _build_translate_topology(
            design_candidate.proposal, chemistry_plan, batch_record, inventory
        )
        result["process_requirements_topology"] = abstract_topology.model_dump()
        topology_matches = _topology_matches_serialized_proposal(
            abstract_topology, design_candidate.proposal
        )
        compiled_topology, allocation_report = compile_inventory_topology(
            abstract_topology,
            proposal=design_candidate.proposal,
            inventory=inventory,
        )
        compiled_topology = normalize_topology_semantics(compiled_topology)
        result["inventory_allocation"] = allocation_report
        result["instrument_manifest"] = allocation_report.get(
            "instrument_manifest", []
        )
        if _sync_stage_hardware_from_compiled_topology(
            design_candidate.proposal, compiled_topology
        ):
            synced_stages = list(design_candidate.proposal.stage_parameters or [])
            result["proposal"] = design_candidate.proposal.model_dump()
            if result.get("multistage_inventory_plan"):
                result["multistage_inventory_plan"]["stage_parameters"] = synced_stages
            result.setdefault("design_calculations", {})[
                "stage_calculations"
            ] = synced_stages
        if final_validation is None:
            final_validation = {
                "schema_version": "flowpilot_final_validation_v1.0",
                "status": "screen_required",
                "checks": {},
                "unresolved_reasons": [],
            }
        checks = final_validation.setdefault("checks", {})
        checks["topology_matches_serialized_design"] = topology_matches
        checks["inventory_topology_assignment_complete"] = bool(
            allocation_report.get("checks", {}).get(
                "all_required_operations_assigned", False
            )
        )
        unresolved = [name for name, passed in checks.items() if not passed]
        final_validation["unresolved_reasons"] = unresolved
        final_validation["status"] = "ready" if not unresolved else "screen_required"
        result["final_validation"] = final_validation
    except Exception as exc:
        logger.warning("Inventory topology compilation failed: %s", exc)
        result["inventory_allocation"] = {
            "schema_version": "flowpilot_inventory_allocation_v1.0",
            "status": "failed",
            "strict_assignment": bool(getattr(inventory, "strict_assignment", False)),
            "checks": {"all_required_operations_assigned": False},
            "assignments": [],
            "instrument_manifest": [],
            "unresolved_requirements": [
                {
                    "requirement_id": "INV-COMPILER",
                    "operation_id": "process_topology",
                    "category": "compiler",
                    "reason": str(exc),
                    "criteria": {},
                }
            ],
        }
        if final_validation is None:
            final_validation = {"checks": {}, "unresolved_reasons": []}
        final_validation.setdefault("checks", {})[
            "inventory_topology_assignment_complete"
        ] = False

    # One deterministic top-level decision is authoritative after every model,
    # council, campaign, inventory, and engineering revision. Hard conflicts
    # cannot be downgraded to a generic experimental screen.
    disposition = apply_design_disposition_gate(
        result,
        proposal=design_candidate.proposal,
        final_validation=final_validation,
        inventory=inventory,
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        objective=objectives,
        hard_constraints=merged_hard_constraints(
            runtime.hard_constraints,
            intake.operating_limits if intake else None,
        ),
        council_safety_report=design_candidate.safety_report,
    )
    result["explanation"] = (
        (result.get("explanation") or "").rstrip()
        + "\n\nFinal design disposition: "
        + disposition.recommended_disposition
        + ". "
        + disposition.rationale
    ).strip()
    logger.info(
        "Step 6d: Final design disposition %s (%d hard failures)",
        disposition.recommended_disposition,
        len(disposition.hard_failures),
    )

    # Store analogies for the revision agent (confidence + context)
    result["_analogies"] = analogies

    # Attach deliberation log for Streamlit rendering
    if design_candidate.deliberation_log:
        result["deliberation_log"] = design_candidate.deliberation_log.model_dump()

    # Pre-council snapshot for before/after comparison in UI
    result["pre_council_proposal"] = pre_council_proposal

    # 7. Build chemistry-aware topology + generate diagram. A blocked
    # candidate is diagnostic output, not an executable process, so it must
    # never receive a polished flowsheet or overwrite validated parameters.
    if disposition.recommended_disposition == "BLOCK":
        logger.warning(
            "Step 7: Executable diagram suppressed; rendering diagnostic topology"
        )
        _store_blocked_topology(result, disposition)
        diagnostic_topology = compiled_topology
        if diagnostic_topology is not None and diagnostic_topology.unit_operations:
            result["diagnostic_topology"] = diagnostic_topology.model_dump()
            try:
                from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder

                diagnostic_artifacts = render_topology_artifacts(
                    diagnostic_topology,
                    title=(
                        "DIAGNOSTIC - NOT EXECUTABLE - "
                        + str(
                            getattr(chemistry_plan, "reaction_name", "")
                            or batch_record.reaction_description
                            or "Process requirements"
                        )[:72]
                    ),
                    builder=FlowsheetBuilder(),
                )
                result["diagnostic_svg_path"] = diagnostic_artifacts["svg_path"]
                result["diagnostic_png_path"] = diagnostic_artifacts["png_path"]
                result["diagnostic_diagram_artifacts"] = {
                    key: value
                    for key, value in diagnostic_artifacts.items()
                    if key != "manifest"
                }
                result["diagnostic_diagram_render_manifest"] = (
                    diagnostic_artifacts["manifest"]
                )
                logger.info(
                    "  Diagnostic diagram saved: %s",
                    diagnostic_artifacts["svg_path"],
                )
            except Exception as exc:
                logger.warning("  Diagnostic diagram generation failed: %s", exc)
                result["diagnostic_svg_path"] = ""
                result["diagnostic_png_path"] = ""
                result["diagnostic_diagram_artifacts"] = {
                    "render_status": "failed",
                    "warnings": [str(exc)],
                }
    else:
        logger.info("Step 7: Generating process flow diagram")
        try:
            from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder

            if compiled_topology is None:
                raise RuntimeError("Inventory-compiled topology is unavailable")
            topology = compiled_topology
            topology_matches = _topology_matches_serialized_proposal(
                topology, design_candidate.proposal
            )
            if topology_matches:
                # The inventory-compiled graph is the authoritative process
                # artifact. Store it before rendering so a Graphviz/PNG failure
                # cannot erase an otherwise valid executable design.
                _store_process_topology(result, topology)
                artifacts = render_topology_artifacts(
                    topology,
                    title=(
                        getattr(chemistry_plan, "reaction_name", "")
                        or batch_record.reaction_description
                    ),
                    builder=FlowsheetBuilder(),
                )
                result["svg_path"] = artifacts["svg_path"]
                result["png_path"] = artifacts["png_path"]
                result["diagram_artifacts"] = {
                    key: value for key, value in artifacts.items() if key != "manifest"
                }
                result["diagram_render_manifest"] = artifacts["manifest"]
                logger.info("  Diagram saved: %s", artifacts["svg_path"])
            else:
                raise RuntimeError(
                    "Compiled topology volume does not match the validated proposal"
                )
        except Exception as e:
            logger.warning("  Diagram generation failed: %s", e)
            result["svg_path"] = ""
            result["png_path"] = ""
            result["diagram_artifacts"] = {
                "render_status": "failed",
                "warnings": [str(e)],
            }

    # Every consumer, including the GUI and autosave output, reads this frozen
    # post-validation contract. Intermediate council/calculator values remain
    # available for audit but can no longer be presented as run instructions.
    result["pipeline_runtime"] = runtime.provenance()
    if scientific and result.get("scientific_assessment"):
        from flora_translate.engine.council_v4.scientific import signature, pressure_headroom
        evidence = result["scientific_assessment"]
        unchanged = signature(FlowProposal.model_validate(result["proposal"])) == evidence["selected_signature"]
        evidence["selected_design_preserved"] = unchanged
        result["final_validation"]["checks"]["scientific_selected_design_preserved"] = unchanged
        engineering = result.get("final_stage_engineering", {})
        pressure_ok = bool(engineering.get("complete")) and pressure_headroom(
            FlowProposal.model_validate(result["proposal"]), engineering, inventory)["passed"]
        result["final_validation"]["checks"]["scientific_final_pressure_headroom"] = pressure_ok
        if not pressure_ok:
            result["final_validation"]["status"] = "blocked"
            result["final_validation"].setdefault("unresolved_reasons", []).append("scientific_final_pressure_headroom")
        if not unchanged:
            result["final_validation"]["status"] = "blocked"
            result["final_validation"].setdefault("unresolved_reasons", []).append("scientific_selected_design_preserved")
    if scientific and runtime.council_backflow_review and result.get("process_topology"):
        from flora_translate.engine.council_v4.backflow import assess_topology
        result["backflow_assessment"] = assess_topology(result["process_topology"],
            FlowProposal.model_validate(result["proposal"]), inventory)
        flow_review = result.get("scientific_assessment", {}).get("backflow_review", {})
        result["backflow_assessment"]["council_decision"] = deepcopy(flow_review.get("decision"))
        result["backflow_assessment"]["revision_status"] = flow_review.get("application_status")
    result["final_design"] = build_final_design_contract(result)
    if result.get("backflow_assessment"):
        result["final_design"]["flow_operability"] = deepcopy(result["backflow_assessment"])
    if result.get("scientific_assessment", {}).get("physics_review"):
        physics_review = result["scientific_assessment"]["physics_review"]
        result["final_design"].setdefault("flow_operability", {})["physics_screen"] = {
            "schema_version": physics_review["schema_version"],
            "profile_provenance": physics_review["profile"]["provenance"],
            "laboratory_execution_status": "review_required", "backflow_probability": None,
            "design_modification_applied": False,
            "selected_candidate_results": next((deepcopy(r) for r in physics_review["results"]
                if r["candidate_id"] == result["scientific_assessment"]["selected_candidate_id"]), None),
            "reviewer_assessments": deepcopy(physics_review.get("reviewer_assessments", {})),
            "limitations": physics_review["limitations"]}
        from flora_translate.engine.council_v4.physics_tools import selected_physics_findings
        operability = result["final_design"]["flow_operability"]
        findings = selected_physics_findings(operability["physics_screen"]["selected_candidate_results"])
        operability.setdefault("findings", []).extend(findings)
        operability["laboratory_execution_status"] = "review_required"
        if result.get("backflow_assessment") is not None:
            result["backflow_assessment"]["findings"].extend(deepcopy(findings))
    publish_final_design_artifacts(result, result["final_design"])
    if (
        result["final_design"]["status"] != "executable"
        and str(result.get("recommended_disposition") or "").upper() != "BLOCK"
    ):
        contract_issues = result["final_design"]["consistency"]["issues"]
        result["recommended_disposition"] = "BLOCK"
        result["reported_disposition"] = "BLOCK"
        result["disposition_rationale"] = (
            "The post-validation final-design contract did not close; do not execute."
        )
        disposition_payload = dict(result.get("design_disposition") or {})
        disposition_payload.update(
            {
                "recommended_disposition": "BLOCK",
                "rationale": result["disposition_rationale"],
                "hard_failures": [
                    {
                        "finding_id": issue.get("code", "FINAL-CHECK"),
                        "message": issue.get("message", "Final consistency check failed."),
                    }
                    for issue in contract_issues
                ],
            }
        )
        result["design_disposition"] = disposition_payload
        if (result.get("process_topology") or {}).get("unit_operations"):
            result["diagnostic_topology"] = result["process_topology"]
            result["diagnostic_svg_path"] = result.get("svg_path", "")
            result["diagnostic_png_path"] = result.get("png_path", "")
            result["diagnostic_diagram_render_manifest"] = result.get(
                "diagram_render_manifest", {}
            )
        result["svg_path"] = ""
        result["png_path"] = ""
        result["process_topology"] = {
            "topology_id": "blocked",
            "generation_status": "blocked",
            "unit_operations": [],
            "streams": [],
            "pid_description": "",
        }
    logger.info(
        "Step 8: Final design contract %s",
        result["final_design"]["status"],
    )
    logger.info(f"Done — Confidence: {result['confidence']}")
    scientific_snapshot("result", result)
    return result


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------

def index_records(records_dir: str = str(RECORDS_DIR)) -> int:
    """Index extraction results into ChromaDB."""
    logger.info(f"Indexing records from {records_dir}")
    store = VectorStore()
    count = store.index_folder(records_dir)
    logger.info(f"Indexed {count} records ({store.pairs_count} with translation pairs)")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n"
              "  python -m flora_translate.main translate '<batch_protocol>'\n"
              "  python -m flora_translate.main index [records_dir]\n")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index":
        records_dir = sys.argv[2] if len(sys.argv) > 2 else str(RECORDS_DIR)
        index_records(records_dir)
    elif cmd == "translate":
        if len(sys.argv) < 3:
            print("Error: provide batch protocol text or JSON file path")
            sys.exit(1)
        batch_input = sys.argv[2]
        if Path(batch_input).exists():
            batch_input = Path(batch_input).read_text()
        result = translate(batch_input)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
