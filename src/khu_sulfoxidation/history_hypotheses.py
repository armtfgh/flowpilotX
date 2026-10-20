"""Post-campaign, exhaustive descriptive review of frozen homogeneous evidence.

This module does not select conditions, predict yields or call a model. It makes
the distinction between evidence provided, explicit archived citations and a
new systematic review inspectable. Original nominations are historical records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

from .provenance import canonical_json, digest, file_hash
from .history import _load_private_context


COMPONENTS = ("residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct",
              "other_products_pct", "unknown_components_pct")
ROLES = ("Upstream", "Chemistry", "KineticsSelectivity", "FluidicsPhotoreactor",
         "Safety", "Skeptic", "Chief")
FACTOR_NOTES = {
    "solvent": (
        "Solvent composition may change the balance between incomplete and further oxidation; assess all components.",
        "Solvent columns are varied jointly. Solvent-ratio basis, dissolved oxygen, photon absorption and flow compatibility remain unknown."),
    "concentration": (
        "Concentration may change incomplete oxidation and sulfone formation; the selected feed is a provisional recipe, not an established optimum.",
        "Recorded concentration contrasts do not measure optical attenuation, oxygen uptake or flow mass transfer."),
    "catalyst_loading": (
        "Catalyst loading may alter residual sulfide and further oxidation; lower loading cannot be assumed to preserve the endpoint.",
        "Photocatalyst dose, delivered photons, solubility and transfer differ between batch and flow; percentages alone establish no rate law."),
    "light": (
        "Reported light colors give different component endpoints; retain spectral compatibility as an explicit design question.",
        "Raw records report color and nominal power, not calibrated wavelength, photon flux or equal absorbed dose. Slide wavelengths are separate evidence."),
    "oxidant": (
        "Changing air to oxygen may improve substrate disappearance without improving desired-product yield; test supply without assuming monotonic benefit.",
        "Batch air/oxygen labels are not molar flow supply or uptake. Gas delivery also changes two-phase hydrodynamics."),
    "temperature": (
        "Temperature may change the balance of residual sulfide, sulfoxide and sulfone; catalogue temperature limits are not a chemical optimum.",
        "Recorded temperatures do not establish illuminated liquid temperature, uncertainty, heat transfer or a transferable kinetic law."),
    "batch_duration": (
        "Incomplete oxidation and further sulfoxide oxidation are competing explanations to test; longer duration need not improve sulfoxide yield.",
        "These are batch endpoints, with unknown chronology and independent-repeat relationships. Batch hours cannot set flow residence times."),
}
ILLUSTRATIVE_ENTRIES = {
    "solvent": [15, 22, 23, 36, 72, 102],
    "concentration": [5, 33, 16], "catalyst_loading": [5, 114, 115, 116],
    "light": [5, 112, 113], "oxidant": [22, 35, 28, 40],
    "temperature": [40, 4], "batch_duration": [110, 99, 32, 49, 28, 33, 35, 59, 105, 111],
}


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _decode_records(evidence):
    raw = evidence["raw_records"]
    records = []
    for values in raw["rows"]:
        if len(values) != len(raw["row_columns"]):
            raise ValueError("Frozen row does not match its declared schema")
        encoded = dict(zip(raw["row_columns"], values))
        entry = encoded["source_entry"]
        conditions = dict(zip(raw["condition_columns"], encoded["conditions"]))
        for field, codebook in raw["categorical_condition_codebooks"].items():
            code = conditions[field]
            conditions[field] = None if code is None else codebook[code]
        dispute = deepcopy(raw["source_disputes_by_entry"].get(str(entry)))
        sources = []
        for alias, row in zip(raw["source_row_alias_order"], encoded["source_rows"]):
            if row is None:
                continue
            descriptor = evidence["source_registry"][alias]
            pointer = f"{descriptor['sheet']}!row:{row}" if descriptor.get("sheet") else f"row:{row}"
            sources.append({"source_alias": alias, "source_row": row,
                            "source_record_id": f"{descriptor['source_id']}#{pointer}"})
        records.append({
            "record_id": raw["record_citation_id_format"].format(entry=entry),
            "source_entry": entry,
            "version_membership": raw["version_codebook"][encoded["version_code"]],
            "reported_actual_conditions": conditions,
            "reported_outcomes": dict(zip(raw["reported_outcome_columns"], encoded["reported_outcomes"])),
            "calculated_outcomes": dict(zip(raw["calculated_outcome_columns"], encoded["calculated_outcomes"])),
            "analytical_flags": [raw["analytical_flag_codebook"][c] for c in encoded["analytical_flag_codes"]],
            "source_rows": sources, "source_dispute": dispute,
            "interpretation_eligible": dispute is None and raw["interpretation_eligibility_default"],
        })
    ids = [r["record_id"] for r in records]
    proof = evidence["coverage"]
    if (len(ids) != len(set(ids)) or len(ids) != proof["canonical_record_count"]
            or digest(ids) != proof["canonical_record_ids_sha256"]):
        raise ValueError("Frozen canonical record coverage is incomplete or duplicated")
    return records


def _outcomes(record):
    return {field: record["reported_outcomes"].get(field) for field in COMPONENTS}


def _contrast(first, second, fields):
    a = {f: first["reported_actual_conditions"][f] for f in fields}
    b = {f: second["reported_actual_conditions"][f] for f in fields}
    if a == b:
        return None
    numeric = len(fields) == 1 and _number(a[fields[0]]) and _number(b[fields[0]])
    reverse = a[fields[0]] > b[fields[0]] if numeric else canonical_json(a) > canonical_json(b)
    if reverse:
        first, second, a, b = second, first, b, a
    deltas, directions = {}, {}
    for name in COMPONENTS:
        x, y = first["reported_outcomes"].get(name), second["reported_outcomes"].get(name)
        delta = y - x if _number(x) and _number(y) else None
        deltas[name] = delta
        directions[name] = "unknown" if delta is None else "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
    return {
        "record_ids": [first["record_id"], second["record_id"]],
        "factor_levels_in_order": [a, b],
        "ordering": "lower_to_higher_recorded_value" if numeric else "lexical_only_no_physical_order",
        "reported_component_delta_second_minus_first_pct_points": deltas,
        "directions": directions,
        "less_sulfoxide_with_more_sulfone": directions["sulfoxide_pct"] == "decrease" and directions["sulfone_pct"] == "increase",
        "more_sulfoxide_with_more_sulfone": directions["sulfoxide_pct"] == "increase" and directions["sulfone_pct"] == "increase",
        "interpretation": "Descriptive endpoint difference, not a causal effect, independent replicate or significance test",
    }


def _factor_review(name, family, by_entry):
    fields = family["varied_recorded_fields"]
    groups, all_contrasts, member_ids = [], [], set()
    for group_id, entries in family["groups"]:
        members = [by_entry[e] for e in entries]
        fixed = {k: v for k, v in members[0]["reported_actual_conditions"].items() if k not in fields}
        if any({k: v for k, v in row["reported_actual_conditions"].items() if k not in fields} != fixed for row in members):
            raise ValueError(f"Mismatched recorded conditions in {group_id}")
        eligible = [r for r in members if r["interpretation_eligible"]]
        contrasts = [contrast for a, b in combinations(eligible, 2)
                     if (contrast := _contrast(a, b, fields)) is not None]
        groups.append({
            "group_id": group_id, "fixed_recorded_conditions": fixed,
            "record_ids": [r["record_id"] for r in members],
            "observations": [{"record_id": r["record_id"],
                              "factor_values": {f: r["reported_actual_conditions"][f] for f in fields},
                              "reported_components": _outcomes(r),
                              "reported_component_sum_pct": r["reported_outcomes"].get("reported_component_sum_pct"),
                              "analytical_flags": r["analytical_flags"],
                              "interpretation_eligible": r["interpretation_eligible"]} for r in members],
            "excluded_from_interpreted_contrasts": [{"record_id": r["record_id"], "reason": "Source dispute or explicit ineligibility; all values remain visible"}
                                                    for r in members if not r["interpretation_eligible"]],
            "contrasts": contrasts,
        })
        all_contrasts.extend(contrasts)
        member_ids.update(r["record_id"] for r in members)
    return {
        "hypothesis_id": f"khu:history-hypothesis:{name}:v1",
        "hypothesis": FACTOR_NOTES[name][0], "applicability_limit": FACTOR_NOTES[name][1],
        "varied_recorded_fields": list(fields), "group_count": len(groups),
        "record_count": len(member_ids), "record_ids": sorted(member_ids),
        "groups": groups,
        "descriptive_pattern_summary": {
            "comparison_pair_count": len(all_contrasts),
            "component_direction_counts": {f: dict(Counter(c["directions"][f] for c in all_contrasts)) for f in COMPONENTS},
            "less_sulfoxide_with_more_sulfone_pair_count": sum(c["less_sulfoxide_with_more_sulfone"] for c in all_contrasts),
            "more_sulfoxide_with_more_sulfone_pair_count": sum(c["more_sulfoxide_with_more_sulfone"] for c in all_contrasts),
            "counting_limit": "Pairs overlap and are not independent experiments. Categorical ordering is lexical, not a dose response. No uncertainty model or significance threshold is available.",
        },
    }


def _archived_use(directory, records, group_members):
    known_ids = {r["record_id"] for r in records}
    roles, explicit, grouped = {}, set(), set()
    for role in ROLES:
        path = directory / "history_informed" / f"{role}.json"
        archived = json.loads(path.read_text(encoding="utf-8"))
        response = archived["response"]
        sites = []
        for field in ("findings", "candidate_reviews"):
            for index, item in enumerate(response[field]):
                sites.append({"json_pointer": f"/response/{field}/{index}", "archived_claim": deepcopy(item)})
        citations = {c for site in sites for c in site["archived_claim"].get("citations", [])}
        direct = citations & known_ids
        groups = citations & group_members.keys()
        reached = {rid for gid in groups for rid in group_members[gid]}
        explicit.update(direct)
        grouped.update(reached)
        roles[role] = {
            "source_path": str(path), "sha256": file_hash(path),
            "model": archived.get("api_metadata", {}).get("model"),
            "explicit_raw_record_citations": sorted(direct), "cited_factor_group_ids": sorted(groups),
            "records_in_cited_factor_groups": sorted(reached),
            "other_citation_ids": sorted(citations - direct - groups), "citation_sites": sites,
            "archived_conclusion": response["conclusion"],
            "archived_core_candidate_ids": response["recommended_core_candidate_ids"],
            "archived_comparison_nomination_ids": response["comparison_nomination_ids"],
            "archived_repeat_candidate_id": response["repeat_candidate_id"],
        }
    return {
        "status": "Archived claims retained verbatim for audit; not independently validated explanations or current recommendations",
        "separate_prerecommendation_hypothesis_stage": False,
        "original_sequence": "Upstream produced findings and selections together; four specialists reviewed that proposal, then Skeptic, then Chief.",
        "all_116_in_prompt_does_not_prove_all_116_influenced_selection": True,
        "explicit_raw_record_citation_ids": sorted(explicit),
        "records_in_cited_factor_groups": sorted(grouped),
        "raw_or_group_citation_reachable_record_ids": sorted(explicit | grouped),
        "records_without_raw_or_factor_group_citation": sorted(known_ids - explicit - grouped),
        "citation_limit": "A group citation makes its members traceable, not proof the model considered each member; a summary citation is not counted as explicit use of every record.",
        "roles": roles,
    }


def _report_summary(result):
    """Small display view; examples supplement, never replace, complete groups."""
    private_display = _load_private_context()["hypothesis_display"]
    records = result["records"]
    by_entry = {r["source_entry"]: r for r in records}
    archived, coverage = result["archived_model_use"], result["coverage"]
    factors, references = [], []
    for factor, review in result["factor_hypotheses"].items():
        counts = review["descriptive_pattern_summary"]["component_direction_counts"]["sulfoxide_pct"]
        fields = review["varied_recorded_fields"]
        numeric = len(fields) == 1 and factor in {"concentration", "catalyst_loading", "temperature", "batch_duration"}
        count_text = (f"As the recorded factor increases, sulfoxide rises in {counts.get('increase', 0)}, "
                      f"falls in {counts.get('decrease', 0)} and is unchanged in {counts.get('unchanged', 0)} pairs. "
                      "Pairs overlap; these are not independent statistics. ") if numeric else "Categorical contrasts have no dose ordering. "
        examples, evidence_ids = [], [review["hypothesis_id"]]
        for entry in ILLUSTRATIVE_ENTRIES[factor]:
            if entry not in by_entry:
                continue
            record = by_entry[entry]
            levels = ", ".join(f"{f}={record['reported_actual_conditions'][f]}" for f in fields
                               if record["reported_actual_conditions"][f] is not None)
            values = "/".join("unknown" if record["reported_outcomes"].get(f) is None
                              else f"{record['reported_outcomes'][f]:g}" for f in COMPONENTS[:3])
            hold = " [source dispute; not interpreted]" if not record["interpretation_eligible"] else ""
            examples.append(f"entry {entry} ({levels}): {values}{hold}")
            evidence_ids.append(record["record_id"])
        factors.append({
            "factor": factor,
            "evidence_summary": f"{review['group_count']} matched-recorded-field groups / {review['record_count']} records. "
                                + count_text + "Illustrative endpoints, residual sulfide/sulfoxide/sulfone (%): " + "; ".join(examples) + ". All remaining groups and endpoints are preserved in the full ledger.",
            "flow_implication": review["hypothesis"], "limitations": review["applicability_limit"],
            "evidence_ids": evidence_ids,
        })
        references.append({"evidence_id": review["hypothesis_id"], "label": f"Complete descriptive review: {factor}",
                           "source_location": f"history_hypotheses.json#/factor_hypotheses/{factor}"})
        for group in review["groups"]:
            references.append({"evidence_id": group["group_id"], "label": f"Matched recorded-field {factor} family, entries " + ", ".join(str(int(r.rsplit(':', 1)[1])) for r in group["record_ids"]),
                               "source_location": f"{result['provenance']['input_freeze_path']}#/snapshot/inputs/history_informed/batch_optimization_evidence/recorded_factor_comparisons/families/{factor}"})
    for record in records:
        locations = []
        for source in record["source_rows"]:
            descriptor = result["source_registry"][source["source_alias"]]
            sheet = f"{descriptor['sheet']}!" if descriptor.get("sheet") else ""
            locations.append(f"{descriptor['path']}#{sheet}row:{source['source_row']}")
        references.append({"evidence_id": record["record_id"], "label": f"Homogeneous raw entry {record['source_entry']}",
                           "source_location": "; ".join(locations)})
    uncertainty = result["anomalies_and_repeated_settings"]
    factors.append({
        "factor": "analytical_and_repeat_uncertainty",
        "evidence_summary": f"Flag counts: {canonical_json(uncertainty['flag_counts'])}. "
                            f"{len(uncertainty['identical_recorded_setting_groups'])} groups share every recorded condition. "
                            + private_display["analytical_uncertainty"],
        "flow_implication": uncertainty["hypothesis"],
        "limitations": "Analytical uncertainty and independent-repeat relationships are unknown; no normalization, averaging or significance claim.",
        "evidence_ids": [uncertainty["hypothesis_id"]],
    })
    references.append({"evidence_id": uncertainty["hypothesis_id"], "label": "Analytical anomalies and identical recorded settings",
                       "source_location": "history_hypotheses.json#/anomalies_and_repeated_settings"})
    return {
        "coverage_rows": [
            {"item": "Homogeneous prompt coverage", "value": coverage["canonical_record_count"],
             "interpretation": "Every canonical record was provided once; provision is not proof of model attention or use."},
            {"item": "Linked export row representations", "value": coverage["source_representation_count"],
             "interpretation": "116 Excel plus 111 CSV rows; overlapping exports are not independent observations."},
            {"item": "New descriptive factor review", "value": coverage["records_in_factor_comparisons"],
             "interpretation": "Records in matched-field groups; every group/pair is retained, including mixed outcomes."},
            {"item": "Records without a factor comparison", "value": len(coverage["records_without_factor_comparison"]),
             "interpretation": "Entries 21/38/68/73 share settings but lack a matched varied-factor partner; entry 108 is unmatched. All remain visible."},
            {"item": "Explicit raw-record citations across archived roles", "value": len(archived["explicit_raw_record_citation_ids"]),
             "interpretation": "Distinct record IDs explicitly cited; prose or summary references are not silently counted as all-record use."},
            {"item": "Raw or cited-factor reachable rows", "value": len(archived["raw_or_group_citation_reachable_record_ids"]),
             "interpretation": "Includes members of cited groups; not proof each member influenced selection."},
            {"item": "Separate original hypothesis stage", "value": "absent",
             "interpretation": "Upstream combined findings and selection; council reviewed the proposal. This ledger was created afterward."},
            {"item": "Heterogeneous rows in this ledger", "value": 0,
             "interpretation": "42 records remain separate later-phase evidence; not pooled or sent to either homogeneous arm."},
        ], "factor_rows": factors, "reference_rows": references,
    }


def build_history_hypothesis_ledger(campaign_directory: str | Path) -> dict:
    """Review every frozen homogeneous entry without mutating campaign artifacts.

    ``campaign_directory`` contains input_freeze.json and history_informed role
    archives. The returned JSON-compatible object is a new, deterministic review;
    use provenance.write_json with a new filename to preserve it if desired.
    """
    directory = Path(campaign_directory)
    freeze_path = directory / "input_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    snapshot = frozen["snapshot"]
    if digest(snapshot) != frozen["snapshot_sha256"]:
        raise ValueError("Frozen input content hash mismatch")
    evidence = snapshot["inputs"]["history_informed"]["batch_optimization_evidence"]
    if evidence["catalyst_system"] != "homogeneous" or evidence["experiment_mode"] != "batch":
        raise ValueError("This ledger is exclusively for homogeneous batch evidence")
    records = _decode_records(evidence)
    by_entry = {r["source_entry"]: r for r in records}
    families = evidence["recorded_factor_comparisons"]["families"]
    if set(families) != set(FACTOR_NOTES):
        raise ValueError("Every declared factor family requires an explicit review")
    hypotheses = {name: _factor_review(name, family, by_entry) for name, family in families.items()}
    groups = {g["group_id"]: g["record_ids"] for h in hypotheses.values() for g in h["groups"]}
    membership = defaultdict(list)
    for gid, ids in groups.items():
        for rid in ids:
            membership[rid].append(gid)
    recorded_settings = defaultdict(list)
    for row in records:
        recorded_settings[canonical_json(row["reported_actual_conditions"])].append(row)
    repeated = []
    for setting, rows in recorded_settings.items():
        if len(rows) < 2:
            continue
        repeated.append({"record_ids": [r["record_id"] for r in rows], "reported_actual_conditions": json.loads(setting),
                         "reported_outcomes_by_record": {r["record_id"]: r["reported_outcomes"] for r in rows},
                         "interpretation": "Identical recorded settings do not establish independent repeats or shared experimental identity; do not average or deduplicate."})
    archived = _archived_use(directory, records, groups)
    direct, grouped = set(archived["explicit_raw_record_citation_ids"]), set(archived["records_in_cited_factor_groups"])
    anomalies = [r for r in records if r["analytical_flags"]]
    source_count = sum(len(r["source_rows"]) for r in records)
    if source_count != evidence["coverage"]["source_representation_count"]:
        raise ValueError("Source representation coverage mismatch")
    result = {
        "schema_version": "khu-post-campaign-history-hypotheses-v1",
        "status": "post_campaign_offline_evidence_review_not_new_nominations",
        "provenance": {"input_freeze_path": str(freeze_path), "input_freeze_sha256": file_hash(freeze_path),
                       "snapshot_sha256": frozen["snapshot_sha256"], "original_frozen_at": frozen["frozen_at"]},
        "scientific_scope": {
            "system": "homogeneous", "experiment_mode": "batch", "flow_predictions": False,
            "new_model_calls": 0, "new_history_omitted_arm": False, "new_condition_selection": False,
            "original_numerical_screen_status": "withdrawn historical proposals; this evidence review does not repair or validate their numerical time grid",
            "method": "Exhaustive decoding and matched-recorded-field endpoint contrasts; no causal fit, ranking, clipping, normalization or outcome-based pruning",
            "limits": deepcopy(evidence["interpretation_limits"]),
            "identity_and_analysis_scope": deepcopy(evidence["raw_records"]["identity_and_analysis_scope"]),
        },
        "source_registry": deepcopy(evidence["source_registry"]),
        "records": records, "factor_hypotheses": hypotheses,
        "anomalies_and_repeated_settings": {
            "hypothesis_id": "khu:history-hypothesis:analytical-and-repeat-uncertainty:v1",
            "hypothesis": "Endpoint uncertainty and unknown repeat relationships can mimic differences; independent operation and calibrated separate-component analysis are needed.",
            "flag_counts": dict(Counter(flag for r in records for flag in r["analytical_flags"])),
            "flagged_records": [{"record_id": r["record_id"], "reported_outcomes": r["reported_outcomes"],
                                 "calculated_outcomes": r["calculated_outcomes"], "analytical_flags": r["analytical_flags"]} for r in anomalies],
            "identical_recorded_setting_groups": repeated,
            "curation_policy": "Every row remains. Neither poor performance nor >100% values cause removal; source disputes are visible and excluded from interpreted contrasts independently of outcome.",
        },
        "separate_slide_evidence": deepcopy(evidence["slide_table_evidence"]),
        "archived_model_use": archived,
        "coverage": {
            "canonical_record_count": len(records), "source_representation_count": source_count,
            "every_record_decoded_once": True, "record_ids_sha256": digest([r["record_id"] for r in records]),
            "xlsx_only_entries": [r["source_entry"] for r in records if r["version_membership"] == "xlsx_only"],
            "records_in_factor_comparisons": len(membership),
            "records_without_factor_comparison": [r["record_id"] for r in records if r["record_id"] not in membership],
            "heterogeneous_records_in_this_ledger": 0,
            "heterogeneous_later_phase_count_in_frozen_coverage": snapshot["evidence_coverage"]["heterogeneous"]["canonical_record_count"],
            "record_ledger": [{"record_id": r["record_id"], "factor_group_ids": membership[r["record_id"]],
                               "directly_cited_in_archived_model_response": r["record_id"] in direct,
                               "member_of_archived_cited_factor_group": r["record_id"] in grouped,
                               "interpretation_eligible": r["interpretation_eligible"]} for r in records],
        },
    }
    result["report_summary"] = _report_summary(result)
    return result


def attach_design_rationales(design: dict, ledger: dict) -> dict:
    """Attach transparent offline hypotheses; never change or rank setpoints.

    Requires a six-condition 3x2 design and the comprehensive homogeneous ledger.
    Numerical coordinates remain the caller's engineering choices. This is not
    a new model recommendation or an independent history-omitted comparison.
    """
    result = deepcopy(design)
    rows = result["conditions"]
    times = sorted({r["target"]["nominal_mixture_transit_min"] for r in rows})
    levels = sorted({r["target"]["inlet_oxygen_equivalents"] for r in rows})
    pairs = {(r["target"]["nominal_mixture_transit_min"], r["target"]["inlet_oxygen_equivalents"]) for r in rows}
    if len(rows) != 6 or len(times) != 3 or len(levels) != 2 or len(pairs) != 6:
        raise ValueError("Rationales require a complete three-time by two-oxygen design")
    if levels != [0.5, 1.0]:
        raise ValueError("This rationale version applies to nominal 0.5 and 1.0 inlet oxygen equivalents")
    private_display = _load_private_context()["hypothesis_display"]
    expected_recipe = private_display["expected_recipe"]
    if any(result["scenario"]["chemistry"].get(k) != v for k, v in expected_recipe.items()):
        raise ValueError("Held-fixed recipe rationale does not match the proposed composition")
    ids = [r["condition_id"] for r in rows]
    if len(set(ids)) != 6 or result["independent_repeat"]["repeat_of"] not in ids:
        raise ValueError("Unique condition IDs and a valid independent-repeat reference required")
    by_entry = {r["source_entry"]: r for r in ledger["records"]}
    hypothesis = lambda factor: ledger["factor_hypotheses"][factor]["hypothesis_id"]
    raw = lambda entry: by_entry[entry]["record_id"]
    def endpoints(entry):
        return "/".join("unknown" if by_entry[entry]["reported_outcomes"].get(f) is None
                        else f"{by_entry[entry]['reported_outcomes'][f]:g}" for f in COMPONENTS[:3])
    required = {5, 21, 22, 28, 32, 33, 35, 38, 40, 59, 68, 73, 105, 110, 111, 112, 113, 114, 115, 116}
    if not required <= by_entry.keys() or ledger["scientific_scope"]["system"] != "homogeneous":
        raise ValueError("The complete homogeneous KHU evidence ledger is required")
    if any(not by_entry[e]["interpretation_eligible"] for e in required):
        raise ValueError("Resolve cited source disputes before attaching interpreted design rationales")
    limits = ("Numerical time/O2 targets are explicit engineering-planning choices, not derived from batch kinetics or predicted yields. "
              "Nominal gas-liquid transit is an ideal-gas/no-slip estimate, not measured liquid residence. "
              "At matched nominal transit, oxygen changes phase fraction, mass transfer and light exposure; supplied oxygen is not uptake.")
    readout = ("Measure residual sulfide, sulfoxide, sulfone, other products and unknown components separately; preserve original percentages and analytical recovery. "
               "Apply a pre-agreed sulfone limit and analytical uncertainty; record actual flows, pressure, temperature, collection and operational observations.")
    time_summary = (private_display["time_prefix"]
                    + ", ".join(f"{e}: {endpoints(e)}" for e in [110, 32, 28, 33, 35])
                    + ". These endpoints are nonmonotonic and at a different concentration from the flow recipe.")
    oxygen_summary = private_display["oxygen_summary"].format(
        oxygen12=endpoints(22), air12=endpoints(35), oxygen8=endpoints(40), air8=endpoints(28))
    further_summary = private_display["further_summary"].format(
        first=endpoints(59), second=endpoints(105), third=endpoints(111))
    descriptions = {
        (0, 0): ("A short estimated exposure at lower oxygen supply may leave residual sulfide; establish the incomplete-oxidation end of the screen.",
                 [hypothesis("batch_duration"), raw(110), raw(32)], time_summary,
                 "Shortest/lower-supply corner provides a baseline for the same-time higher-supply pair and longer lower-supply conditions."),
        (0, 1): ("Additional supplied oxygen at the shortest nominal transit may change substrate disappearance and sulfoxide/sulfone balance; benefit is uncertain.",
                 [hypothesis("oxidant"), raw(22), raw(35), raw(28), raw(40)], oxygen_summary,
                 "Pair with the shortest lower-supply condition while solving both liquid and gas flows to maintain the same nominal transit target."),
        (1, 0): ("Intermediate estimated exposure at lower supply may reduce residual sulfide while limiting further oxidation; both possibilities require measurement.",
                 [hypothesis("batch_duration"), raw(32), raw(28), raw(33), raw(35)], time_summary,
                 "Middle/lower-supply condition connects the short and long lower-supply corners without assuming monotonic improvement."),
        (1, 1): ("Intermediate estimated exposure with higher oxygen supply may change the desired-product balance relative to its lower-supply pair.",
                 [hypothesis("batch_duration"), hypothesis("oxidant"), raw(22), raw(35), raw(28), raw(40)], time_summary + " " + oxygen_summary,
                 "Middle/higher-supply condition provides a paired oxygen comparison and a central coordinate for an independent local repeat if selected; it is not ranked as an optimum."),
        (2, 0): ("Longer estimated exposure at lower supply tests whether remaining sulfide can be consumed without excessive further oxidation.",
                 [hypothesis("batch_duration"), raw(28), raw(33), raw(35), raw(59), raw(105), raw(111)], time_summary + " " + further_summary,
                 "Longest/lower-supply corner contrasts greater estimated exposure with limited inlet supply; lower supply does not guarantee absence of sulfone."),
        (2, 1): ("The longest estimated exposure with higher inlet supply tests sensitivity to further oxidation, without predicting sulfone formation.",
                 [hypothesis("batch_duration"), hypothesis("oxidant"), raw(59), raw(105), raw(111)], further_summary + " " + oxygen_summary,
                 "Longest/higher-supply corner tests the combined exposure/supply extreme within the explicit engineering window; substrate disappearance alone cannot define success."),
    }
    for row in rows:
        target = row["target"]
        index = (times.index(target["nominal_mixture_transit_min"]), levels.index(target["inlet_oxygen_equivalents"]))
        idea, citations, evidence, why = descriptions[index]
        row["rationale"] = {"hypothesis": idea, "evidence_ids": citations, "evidence_summary": evidence,
                            "why_this_condition": f"Target {target['nominal_mixture_transit_min']:g} min nominal mixture transit and {target['inlet_oxygen_equivalents']:g} mol inlet O2/mol sulfide. " + why,
                            "readout": readout, "limitation": limits}
    uncertainty = ledger["anomalies_and_repeated_settings"]
    result["independent_repeat"]["rationale"] = {
        "hypothesis": "Independent preparation and operation may expose local variability hidden by one analytical result.",
        "evidence_ids": [uncertainty["hypothesis_id"], raw(21), raw(38), raw(68), raw(73)],
        "evidence_summary": private_display["repeat_summary"],
        "why_this_condition": f"Independently re-establish {result['independent_repeat']['repeat_of']}; one repeat checks local reproducibility without establishing variability across the entire screen.",
        "readout": readout + " Prepare and operate independently; repeated NMR integration of one aliquot does not count.",
        "limitation": "This repeat is an offline planning choice; it does not validate the condition or create an independent history-omitted arm.",
    }
    result["held_fixed_recipe_rationale"] = {
        "hypothesis": "Hold the shared starting composition and blue-light compatibility fixed while investigating coupled exposure and oxygen supply.",
        "evidence_ids": [hypothesis(f) for f in FACTOR_NOTES] + [raw(e) for e in [5, 112, 113, 114, 115, 116]],
        "evidence_summary": private_display["recipe_summary_prefix"].format(endpoints=endpoints(5))
                            + "Lower loading entries114–116: " + ", ".join(endpoints(e) for e in [114, 115, 116])
                            + "%; green112: " + endpoints(112) + "%; purple113: " + endpoints(113)
                            + "%. All seven factor families, including mixed concentration, solvent and temperature contrasts, remain in the ledger; this is not a single-anchor evidence review.",
        "why_this_condition": "Composition is a disclosed provisional starting recipe. The proposed module's temperature and spectral setting need their own engineering and laboratory basis.",
        "readout": "Confirm homogeneous feed solubility and actual composition, light setting and temperature; retain separate-component analysis.",
        "limitation": "The anchor was shared by both original arms. Its preservation is not proof of optimality, a matched-photon transfer or a new history-omitted comparison. Raw row identity and analytical-method scope remain provisional.",
    }
    result["history_rationale_provenance"] = {
        "status": "post_campaign_offline_interpretation_not_new_model_output", "history_ledger_sha256": digest(ledger),
        "input_freeze_sha256": ledger["provenance"]["input_freeze_sha256"],
        "all_canonical_records_reviewed_descriptively": len(ledger["records"]),
        "numeric_targets_derived_from_batch": False, "model_calls": 0, "new_history_omitted_arm": False,
        "condition_ranking": None,
    }
    return result
