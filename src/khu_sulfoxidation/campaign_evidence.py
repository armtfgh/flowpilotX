"""Compact, complete offline evidence for the prospective homogeneous campaign.

Column-described arrays reduce repeated prompt text without selecting favorable
rows. Every canonical row remains present once; source-row pointers resolve via
the hash registry. The complete lexical/formula archive remains in history.py.
No provider client, retrieval store, model execution or flow prediction is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .evidence import FACTOR_FIELDS, build_evidence_packet
from .history import CONDITION_FIELDS, OUTCOME_FIELDS, ingest_history


CONDITION_COLUMNS = [*CONDITION_FIELDS, "solvent_ratio_basis"]
REPORTED_OUTCOME_COLUMNS = list(OUTCOME_FIELDS)
CALCULATED_OUTCOME_COLUMNS = ["apparent_conversion_pct", "two_product_selectivity_pct", "component_sum_pct"]
ROW_COLUMNS = ["source_entry", "version_code", "conditions", "reported_outcomes",
               "calculated_outcomes", "analytical_flag_codes", "source_rows"]
SLIDE_VALUE_COLUMNS = ["solvent", "photocatalyst", "catalyst_loading_mol_pct", "light_source", "oxidant",
                       "concentration_mol_L", "batch_duration_h", "component_1_pct", "component_2_pct",
                       "component_3_pct", "reported_balance_pct"]
SOURCE_ALIASES = {
    ("homogeneous", "xlsx"): "H_X", ("homogeneous", "csv"): "H_C",
    ("heterogeneous", "xlsx"): "T_X", ("heterogeneous", "csv"): "T_C",
}
VERSION_CODES = {"B": "common_csv_xlsx", "X": "xlsx_only", "C": "csv_only"}
CATEGORICAL_CONDITIONS = ("solvent_1", "solvent_2", "photocatalyst", "oxidant_reported", "light_source_reported")


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def common_study_brief() -> dict:
    """Shared scientific brief without optimization rows or batch outcome values.

    This is an information boundary, not a confirmed apparatus or release record.
    The source of the shared recipe is disclosed without its measured outcomes.
    """
    return {
        "schema_version": "khu-common-study-brief-v1",
        "study_id": "khu_sulfoxidation_v1",
        "scope": "First homogeneous acridine flow screen; heterogeneous work is a later phase",
        "shared_recipe": {
            "substrate": "thioanisole", "catalyst": "acridine", "solvent": "EtOH",
            "substrate_concentration_M": 0.25, "catalyst_loading_mol_percent": 3.0,
            "batch_temperature_C": 24.0, "batch_duration_h": 10.0, "batch_oxidant": "air",
            "status": "shared proposed starting composition; flow settings and actual gas service need confirmation",
        },
        "recipe_selection_disclosure": (
            "The shared recipe was chosen from a selected batch anchor. Both arms know this recipe; "
            "its measured outcomes, optimization table, rankings and alternative-condition results "
            "are withheld from the history-omitted arm. This is a comparison with versus without "
            "batch optimization history, not with versus without any batch knowledge."
        ),
        "objective": {"primary": "high measured sulfoxide yield subject to an agreed sulfone limit",
                      "sulfone_limit_pct": None,
                      "separate_outcomes": ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct",
                                            "other_products_pct", "unknown_components_pct"]},
        "proposed_run_budget": {"core_time_coordinates": 3, "core_oxygen_delivery_levels": 2,
                                "unique_core_conditions": 6, "independent_repeats": 1,
                                "core_runs": 7, "optional_history_omitted_nominations": [1, 2],
                                "maximum_planned_runs": 9,
                                "additional_blanks_and_calibration_runs": "separate explicit budget if required"},
        "analytical_proposal": {"method": "crude-mixture 1H NMR", "internal_standard": "1,3-benzodioxole",
                                "status": "proposed reuse; calibration, uncertainty, sampling and quench/handling must be confirmed"},
        "design_boundaries": [
            "Batch duration is not flow residence time or a fitted flow kinetic parameter.",
            "Use the same independently defined physical candidate pool and apparatus constraints for both arms.",
            "Separate supplied O2 equivalents, reference gas volume and calculated time indices from measurements.",
            "No monotonic yield/time rule, mandatory time reduction, response clipping or mass-closure normalization.",
            "Keep one or two nominations per arm under matched evaluation opportunities; allow convergence.",
            "Freeze both ordered nominations and their complete visible inputs before flow outcomes.",
        ],
        "flow_outcomes": None, "experimental_validation": False,
        "laboratory_release": "pending confirmed settings and laboratory review",
    }


def _registry(history: dict, system: str) -> dict:
    registry = {}
    for record in history["systems"][system]["records"]:
        for source in record["source_representations"]:
            alias = SOURCE_ALIASES[(system, source["format"])]
            descriptor = {"source_id": source["source_id"], "sha256": source["source_sha256"],
                          "path": source["source_path"], "format": source["format"], "sheet": source["sheet"]}
            if alias in registry and registry[alias] != descriptor:
                raise ValueError(f"Multiple unreviewed sources for {alias}")
            registry[alias] = descriptor
    ppt = next(source for source in history["sources"] if source["path"].endswith(".pptx"))
    registry["PPT"] = {key: ppt[key] for key in ("source_id", "sha256", "path")}
    return registry


def _row(record: dict, system: str, codebooks: dict, flag_codes: list[str]) -> list:
    conditions, outcomes = record["reported_actual_conditions"], record["outcomes"]
    source_rows = {s["format"]: s["source_row"] for s in record["source_representations"]}
    encoded_conditions = [codebooks[column].index(conditions[column])
                          if column in codebooks and conditions.get(column) is not None
                          else conditions.get(column) for column in CONDITION_COLUMNS]
    version = next(code for code, label in VERSION_CODES.items() if label == record["version_membership"])
    return [int(record["source_entry"]), version, encoded_conditions,
            [outcomes.get(column) for column in REPORTED_OUTCOME_COLUMNS],
            [outcomes["calculated"].get(column) for column in CALCULATED_OUTCOME_COLUMNS],
            [flag_codes.index(flag) for flag in outcomes["flags"]],
            [source_rows.get("xlsx"), source_rows.get("csv")]]


def _slide_payload(history: dict, system: str) -> dict:
    slides = history["slide_evidence"]
    rows = [r for r in slides["records"] if r["catalyst_system"] == system]
    links = [link for link in slides["candidate_links"]
             if link["raw_record_id"].startswith(f"khu:{system}:raw-entry:")]
    return {
        "scope": "Separate image-table evidence; never added to raw-observation counts",
        "source_alias": "PPT", "transcription_method": slides["transcription_method"],
        "row_columns": ["evidence_id", "slide", "table_entry", "reported_values", "footnotes"],
        "reported_value_columns": list(SLIDE_VALUE_COLUMNS),
        "rows": [[r["evidence_id"], r["slide"], r["table_entry"],
                  [r["raw_image_table_transcription"].get(column) for column in SLIDE_VALUE_COLUMNS],
                  deepcopy(r["footnotes"])] for r in rows],
        "global_slide_statements": deepcopy(slides["global_slide_statements"]),
        "candidate_raw_links_and_disagreements": deepcopy(links),
        "render_sources": deepcopy(slides["render_sources"]),
        "interpretation": (
            "PPT entry numbers are not raw record IDs. Candidate links are not confirmed experiment identities. "
            "A dash remains unquantified, not zero. The slide scheme/method mapping across all raw rows is provisional. "
            "Wavelengths printed on slides are not assigned to raw rows that only report a light color."
        ),
    }


def _factor_payload(packet: dict) -> dict:
    entry_by_id = {r["record_id"]: r["source_entry"] for r in packet["all_records"]}
    factors = {}
    for factor, groups in packet["recorded_condition_factor_comparisons"].items():
        factors[factor] = {
            "varied_recorded_fields": list(FACTOR_FIELDS[factor]),
            "comparison_group_count": len(groups),
            "record_count": len({rid for group in groups for rid in group["record_ids"]}),
            "groups": [[group["group_id"], [int(entry_by_id[rid]) for rid in group["record_ids"]]] for group in groups],
        }
    return {
        "entry_reference_scope": "Entries resolve only within this payload's catalyst system to the full canonical row IDs above",
        "fixed_conditions_basis": "All other recorded fields match the first member's fields; the full conditions for every member are retained in raw_records",
        "reported_levels_basis": "All observed levels are the distinct decoded values of the family's varied fields among its member entries; no level or member is omitted",
        "group_columns": ["group_id", "source_entries"],
        "interpretation": "Descriptive comparisons only; unrecorded conditions, repeat independence and chronology remain unknown. No causal attribution, rankings or flow kinetics.",
        "families": factors,
        "batch_time_groups_with_observed_sulfoxide_decreases": [
            {"group_id": group["group_id"], "source_entry_pairs": [
                [int(entry_by_id[pair["earlier_record_id"]]), int(entry_by_id[pair["later_record_id"]])]
                for pair in group["observed_decreases_at_longer_time"]]}
            for group in packet["time_groups"] if group["observed_decreases_at_longer_time"]],
    }


def _system_payload(history: dict, system: str) -> tuple[dict, dict]:
    packet = build_evidence_packet(history, system=system)
    records = packet["all_records"]
    codebooks = {field: sorted({r["reported_actual_conditions"][field] for r in records
                              if r["reported_actual_conditions"].get(field) is not None})
                 for field in CATEGORICAL_CONDITIONS}
    flag_codes = sorted({flag for record in records for flag in record["outcomes"]["flags"]})
    rows = [_row(record, system, codebooks, flag_codes) for record in records]
    canonical_ids = [record["record_id"] for record in history["systems"][system]["records"]]
    citation_format = f"khu:{system}:raw-entry:{{entry:04d}}"
    represented_ids = [citation_format.format(entry=row[0]) for row in rows]
    if represented_ids != canonical_ids or len(represented_ids) != len(set(represented_ids)):
        raise ValueError("Canonical evidence coverage must be complete, ordered and one row per entry")
    disputes = {}
    for record in records:
        if record["disputed_source_fields"]:
            disputes[record["source_entry"]] = {
                "interpretation_eligible": False,
                "reason": "Conflicting export values require source adjudication; retained but excluded from interpreted comparisons",
                "values": {field: [[SOURCE_ALIASES[(system, source["format"])], source["values"].get(field)]
                                   for source in record["source_representations"]]
                           for field in record["disputed_source_fields"]},
            }
    proof = {
        "catalyst_system": system, "canonical_record_count": len(canonical_ids),
        "payload_record_count": len(rows), "source_representation_count": packet["coverage_ledger"]["source_representation_count"],
        "all_canonical_records_present_once": True, "canonical_record_ids_sha256": _digest(canonical_ids),
        "payload_record_ids_sha256": _digest(represented_ids),
        "interpretation_eligible_count": packet["coverage_ledger"]["interpretation_eligible_count"],
        "held_for_source_resolution_count": packet["coverage_ledger"]["held_for_source_resolution_count"],
        "records_in_any_factor_comparison": packet["coverage_ledger"]["record_count_in_any_factor_comparison"],
    }
    payload = {
        "schema_version": "khu-campaign-system-evidence-v1", "catalyst_system": system,
        "summary_source_id": f"khu:{system}:campaign-evidence:v1",
        "experiment_mode": "batch", "upstream_evidence_schema": packet["schema_version"],
        "source_registry": _registry(history, system),
        "raw_records": {
            "row_columns": list(ROW_COLUMNS), "condition_columns": list(CONDITION_COLUMNS),
            "reported_outcome_columns": list(REPORTED_OUTCOME_COLUMNS),
            "calculated_outcome_columns": list(CALCULATED_OUTCOME_COLUMNS),
            "version_codebook": dict(VERSION_CODES),
            "categorical_condition_codebooks": codebooks, "analytical_flag_codebook": flag_codes,
            "codebook_definition": "A non-null integer in a categorical condition column or flag array indexes its zero-based codebook. All other condition numbers are uncoded. Null remains unknown.",
            "record_citation_id_format": citation_format,
            "source_row_alias_order": [SOURCE_ALIASES[(system, extension)] for extension in ("xlsx", "csv")],
            "source_pointer_definition": "Look up alias in source_registry. XLSX pointer is sheet!row; CSV pointer is parsed source-row number. Source record ID is source_id#sheet!row:N for XLSX or source_id#row:N for CSV.",
            "rows": rows,
            "interpretation_eligibility_default": True, "source_disputes_by_entry": disputes,
            "conditions_note": "Values interpreted from selected workbook caches; raw light labels retained. Wavelength per raw row and solvent-ratio basis remain unconfirmed. No missing field is changed to zero.",
            "outcome_definitions": {
                "apparent_conversion_pct": "100 minus residual sulfide; calculated, not an independent measurement",
                "two_product_selectivity_pct": "100*sulfoxide/(sulfoxide+sulfone); excludes other/unknown components and is not primary yield",
                "component_sum_pct": "Sum of all five reported components when all are present; not independent gravimetric recovery",
            },
            "identity_and_analysis_scope": {
                "candidate_substrate": "thioanisole", "row_identity_confirmed": False,
                "candidate_method": "crude-mixture 1H NMR", "candidate_internal_standard": "1,3-benzodioxole",
                "status": "Inferred from slide scheme/method; scope across all raw rows remains unconfirmed",
            },
        },
        "export_reconciliation": deepcopy(history["systems"][system]["reconciliation"]),
        "slide_table_evidence": _slide_payload(history, system),
        "recorded_factor_comparisons": _factor_payload(packet),
        "coverage": {**proof,
                     "membership_basis": "Every factor-group membership is recorded in recorded_factor_comparisons.families; a row absent from those lists remains fully visible in raw_records",
                     "entries_without_factor_comparison": [int(r["source_entry"]) for r in packet["coverage_ledger"]["records"] if not r["factor_comparison_memberships"]],
                     "entries_with_unreported_factor_values": {r["source_entry"]: r["factors_without_reported_value"] for r in packet["coverage_ledger"]["records"] if r["factors_without_reported_value"]}},
        "analytical_flag_counts": deepcopy(packet["review_flags"]),
        "unresolved_issues": deepcopy(history["issues"]),
        "interpretation_limits": deepcopy(packet["interpretation_limits"]),
        "not_used_for": ["fitted flow kinetics", "predicted flow yields", "automatic batch-to-flow time scaling"],
    }
    return payload, proof


def _visible_ids(payload: dict) -> list[str]:
    raw = payload["raw_records"]
    return sorted({payload["summary_source_id"],
        *(raw["record_citation_id_format"].format(entry=row[0]) for row in raw["rows"]),
        *(row[0] for row in payload["slide_table_evidence"]["rows"]),
        *(group[0] for family in payload["recorded_factor_comparisons"]["families"].values() for group in family["groups"]),
        *(source["source_id"] for source in payload["source_registry"].values())})


def build_campaign_evidence(root: str | Path) -> dict:
    """Return informed homogeneous data and a separate later-phase archive.

    Only informed_history_payload belongs in a homogeneous history-informed arm.
    Do not send this outer object to either arm: it contains both study phases.
    The history-omitted arm may receive common_brief, never either evidence key.
    """
    history = ingest_history(Path(root))
    informed, homogeneous_proof = _system_payload(history, "homogeneous")
    later, heterogeneous_proof = _system_payload(history, "heterogeneous")
    serialized = _canonical(informed)
    return {
        "schema_version": "khu-campaign-evidence-v1", "common_brief": common_study_brief(),
        "informed_history_payload": informed,
        "later_phase_evidence": {"status": "heterogeneous_later_phase_not_for_homogeneous_arms", "payload": later},
        "visible_citation_ids": {"history_informed": _visible_ids(informed), "later_phase": _visible_ids(later)},
        "coverage_proof": {"homogeneous": homogeneous_proof, "heterogeneous": heterogeneous_proof,
                           "slide_table_rows": len(history["slide_evidence"]["records"]),
                           "slide_table_rows_homogeneous": len(informed["slide_table_evidence"]["rows"]),
                           "slide_table_rows_heterogeneous": len(later["slide_table_evidence"]["rows"])},
        "informed_payload_size": {"compact_json_characters": len(serialized),
                                  "compact_json_utf8_bytes": len(serialized.encode("utf-8")),
                                  "tokens_estimated_at_four_characters_per_token": (len(serialized) + 3) // 4,
                                  "token_count_status": "rough size heuristic; exact count depends on the chosen model tokenizer"},
        "source_archive_note": {"canonical_history_content_sha256": _digest(history),
                                "history_schema_version": history["schema_version"],
                                "regeneration_api": "history.ingest_history(root)",
                                "scope": "Full raw lexical values, cached formulas, source alternatives and hashes remain in the canonical history archive; this compact payload does not replace it"},
    }
