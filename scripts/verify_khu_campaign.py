#!/usr/bin/env python3
"""Read-only final KHU campaign audit; no API client imports or network calls.

Usage:
  python scripts/verify_khu_campaign.py CAMPAIGN [--report-dir REPORT]
      [--output NEW_AUDIT.json]

Exit 0: checks passed; 1: failed; 2: not run (campaign unfinished).
An optional audit is created exclusively, never overwritten. This verifies
archival/structural/calculation integrity, not chemical truth, laboratory release,
arbitrary-text information leakage, provider billing, or rendered PDF layout.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys

ARMS = ("history_omitted", "history_informed")
STAGES = ("Upstream", "Chemistry", "KineticsSelectivity", "FluidicsPhotoreactor", "Safety", "Skeptic", "Chief")
COMMON_IDS = {"study:brief", "apparatus:conditional", "calculation:pool"}
PRICES = {"gpt-6-astra": {"input": 10, "cached_input": 1, "cache_write": 12.5, "output": 50},
          "gpt-6.1-sol": {"input": 2, "cached_input": .1, "cache_write": 2.5, "output": 10}}
R = 0.0831446261815324  # mL bar / mmol K; independent ideal-gas recomputation.
STATUS = "conditional_review_not_released_for_experiment"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(type(actual) in (float, int) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"{label}: {actual!r} != independently calculated {expected!r}")


def inside(root, name):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"Unsafe archive path: {name}")
    return path


def artifacts(folder, mapping, extra=()):
    require({p.name for p in folder.iterdir()} == set(mapping) | set(extra), f"Artifact inventory mismatch: {folder}")
    for name, expected in mapping.items():
        path = inside(folder, name)
        require(path.is_file() and file_hash(path) == expected, f"Artifact hash mismatch: {path}")


def history_ids(history):
    raw = history["raw_records"]
    return {history["summary_source_id"],
            *(raw["record_citation_id_format"].format(entry=r[0]) for r in raw["rows"]),
            *(r[0] for r in history["slide_table_evidence"]["rows"]),
            *(g[0] for family in history["recorded_factor_comparisons"]["families"].values() for g in family["groups"]),
            *(s["source_id"] for s in history["source_registry"].values())}


def check_proposal(response, role, pool, citations, upstream):
    require(response["role"] == role, f"Response role mismatch: {role}")
    by_id = {c["candidate_id"]: c for c in pool["candidates"]}
    core = response["recommended_core_candidate_ids"]
    if role in ("Upstream", "Chief"):
        require(len(core) == len(set(core)) == 6, f"{role}: six distinct core IDs required")
        selected = [by_id[c] for c in core]
        require(all(c["status"] == "eligible_under_declared_assumptions" for c in selected), f"{role}: rejected selection")
        times = {c["time_coordinate"] for c in selected}
        oxygen = {c["oxygen_coordinate"] for c in selected}
        require(len(times) == 3 and len(oxygen) == 2 and
                {(c["time_coordinate"], c["oxygen_coordinate"]) for c in selected} == set(product(times, oxygen)), f"{role}: incomplete 3x2 grid")
        require(any(set(g["candidate_ids"]) == set(core) for g in pool["grid_groups"]), f"{role}: absent frozen grid")
        ranked = response["comparison_nomination_ids"]
        require(len(ranked) == len(set(ranked)) == 2 and set(ranked) <= set(core), f"{role}: invalid ranked nominees")
        require(response["repeat_candidate_id"] in core, f"{role}: repeat outside core")
        review_ids = set(core)
    else:
        require(core == [] and response["comparison_nomination_ids"] == [] and response["repeat_candidate_id"] is None, f"{role}: unauthorized selection")
        review_ids = set(upstream["recommended_core_candidate_ids"])
    reviews = response["candidate_reviews"]
    require(len(reviews) == 6 and {r["candidate_id"] for r in reviews} == review_ids, f"{role}: reviews must cover six relevant IDs")
    for row in reviews:
        require(row["verdict"] in {"retain", "reserve", "reject"} and row["rationale"].strip(), f"{role}: invalid review")
        require(role != "Chief" or row["verdict"] == "retain", "Chief selected a rejected/reserved core")
    for row in reviews + response["findings"]:
        require(bool(row["citations"]) and set(row["citations"]) <= citations, f"{role}: missing/inaccessible citation")
    require(response["conclusion"].strip() and response["limitations"], f"{role}: missing conclusion/limitations")


def check_pool(pool):
    scenario, policy = pool["scenario"], pool["generation_policy"]
    require(pool["configuration_sha256"] == digest({k: pool[k] for k in ("sources", "scenario", "generation_policy")}), "Pool configuration hash mismatch")
    require(policy["independent_of_optimization_history"] is True and policy["optimization_history_files_read"] == [] and policy["batch_time_used"] is False, "Pool origin is not independent of optimization history")
    require(pool["experimental_validation"] is False, "Pool falsely claims experimental validation")
    require(all(h["actual_assignment"] is None for h in scenario["hardware_choices"].values()), "Assumed apparatus became actual")
    require(scenario["bpr_assignment"] is None and scenario["reactor_pressure_bar_abs"] is None, "Unconfirmed pressure became a setting")
    assumptions = {a["id"]: a for a in scenario["assumptions"]}
    gas, air = assumptions["A_GAS_REFERENCE"], assumptions["A_AIR_COMPOSITION"]
    require(gas["status"] == air["status"] == "assumed_unconfirmed", "Gas assumption status changed")
    temp, pressure, fraction = gas["temperature_K"], gas["pressure_bar_abs"], air["oxygen_mole_fraction"]
    volume = scenario["geometry"]["volume_mL"]
    recipe = scenario["shared_recipe"]
    concentration, loading = recipe["substrate_concentration_M"], recipe["catalyst_loading_mol_percent"]
    close(scenario["geometry"]["calculated_total_tubing_length_m"], 4 * volume / math.pi / scenario["geometry"]["ID_mm"] ** 2, "Tubing geometry")
    for key, expected in [("substrate_mmol_per_mL_feed", concentration), ("catalyst_mmol_per_mL_feed", concentration * loading / 100), ("catalyst_concentration_mM", concentration * loading * 10)]:
        close(scenario["feed_calculations"][key], expected, key)
    pump = scenario["hardware_choices"]["pump"]["source_reported_inventory_record"]
    mfc = scenario["hardware_choices"]["mfc"]["source_reported_inventory_record"]
    candidates = pool["candidates"]
    require(len({c["candidate_id"] for c in candidates}) == len(candidates), "Duplicate pool candidate ID")
    eligible = []
    for candidate in candidates:
        cid, settings, calc = candidate["candidate_id"], candidate["proposed_setpoints"], candidate["calculated"]
        ql, qg = settings["liquid_flow_mL_min"], settings["gas_reference_flow_mL_min"]
        target = candidate["nominal_oxygen_equivalents"]
        substrate = concentration * ql
        total_gas = pressure * qg / (R * temp)
        oxygen = total_gas * fraction
        equivalents = oxygen / substrate
        error = abs(equivalents - target) / target
        ideal = target * substrate * R * temp / pressure / fraction
        increment = Decimal(str(mfc["flow_rate_increment_sccm"]))
        quantized = float((Decimal(str(ideal)) / increment).to_integral_value(rounding=ROUND_HALF_UP) * increment)
        close(candidate["ideal_unquantized_gas_reference_flow_mL_min"], ideal, f"{cid} ideal gas flow")
        close(qg, quantized, f"{cid} rounded gas flow")
        for key, value in {"liquid_only_space_time_min": volume / ql, "inlet_reference_volume_index_min": volume / (ql + qg),
                           "substrate_mmol_min": substrate, "catalyst_mmol_min": substrate * loading / 100,
                           "total_gas_mmol_min": total_gas, "oxygen_mmol_min_supplied": oxygen,
                           "oxygen_equivalents_supplied": equivalents, "oxygen_relative_quantization_error": error,
                           "feed_mL_per_elapsed_hour_at_continuous_setting": ql * 60,
                           "substrate_mmol_per_elapsed_hour_at_continuous_setting": substrate * 60,
                           "catalyst_mmol_per_elapsed_hour_at_continuous_setting": substrate * loading / 100 * 60}.items():
            close(calc[key], value, f"{cid} {key}")
        require(calc["actual_liquid_contact_time_min"] is None and settings["bpr_setting"] is None and settings["light_power_setting"] is None, f"{cid}: unknown changed into actual setting")
        coefficient = qg * pressure * (settings["temperature_C"] + 273.15) / temp
        close(calc["mixture_transit"]["gas_volume_pressure_coefficient_mL_bar_min"], coefficient, f"{cid} gas pressure coefficient")
        for illustration in calc["mixture_transit"]["illustrations"]:
            reactor_gas = coefficient / illustration["pressure_bar_abs"]
            close(illustration["gas_reactor_flow_mL_min_estimate"], reactor_gas, f"{cid} reactor gas")
            close(illustration["no_slip_mixture_transit_min_estimate"], volume / (ql + reactor_gas), f"{cid} mixture transit")
        close(ql / pump["flow_rate_increment_mL_min"], round(ql / pump["flow_rate_increment_mL_min"]), f"{cid} pump increment")
        feasible = (pump["min_flow_rate_mL_min"] <= ql <= pump["max_flow_rate_mL_min"] and
                    mfc["min_flow_sccm"] <= qg <= mfc["max_flow_sccm"] and error <= policy["oxygen_relative_tolerance_after_quantization"])
        require((candidate["status"] == "eligible_under_declared_assumptions") == feasible, f"{cid}: eligibility inconsistent with arithmetic")
        require(bool(candidate["rejection_reasons"]) != feasible, f"{cid}: rejection reason missing or inappropriate")
        if feasible:
            eligible.append(cid)
    require(pool["feasible_candidates"] == eligible, "Feasible candidate list mismatch")
    lookup = {(c["time_coordinate"], c["oxygen_coordinate"]): c["candidate_id"] for c in candidates if c["candidate_id"] in eligible}
    expected_grids = {frozenset(lookup[t, o] for t, o in product(times, oxygens))
                      for times in combinations(sorted({t for t, _ in lookup}), 3)
                      for oxygens in combinations(sorted({o for _, o in lookup}), 2)
                      if all((t, o) in lookup for t, o in product(times, oxygens))}
    require({frozenset(g["candidate_ids"]) for g in pool["grid_groups"]} == expected_grids, "Grid enumeration incomplete/invalid")
    return {"candidate_count": len(candidates), "eligible_count": len(eligible), "complete_grids": len(expected_grids)}


def check_campaign(directory, checks):
    original = read(directory / "input_freeze.json")
    nominations = read(directory / "nominations_freeze.json")
    snapshot = original["snapshot"]
    require(original["snapshot_sha256"] == nominations["input_freeze_sha256"] == digest(snapshot), "Input freeze hash mismatch")
    require(original["flow_outcomes_seen"] is nominations["flow_outcomes_seen"] is False and nominations["laboratory_released"] is False, "Campaign must remain pre-outcome and unreleased")
    require(datetime.fromisoformat(original["frozen_at"]) < datetime.fromisoformat(nominations["frozen_at"]), "Nomination/input freeze chronology invalid")
    config = snapshot["configuration"]
    require(read(directory / "api/configuration.json") == config, "API configuration differs from freeze")
    require(snapshot["source_hashes"] == config["source_hashes"] == read(directory / "source_snapshot/manifest.json")["source_hashes"], "Source registry mismatch")
    for name, expected in snapshot["source_hashes"].items():
        require(file_hash(inside(directory / "source_snapshot", name)) == expected, f"Frozen source hash mismatch: {name}")
    checks["source_snapshot"] = {"status": "passed", "file_count": len(snapshot["source_hashes"])}
    require(config["authorized"] is True and config["endpoint"] == "https://api.openai.com/v1/responses", "No matching explicit authorization/endpoint")
    require(config["max_attempts"] == 14 and config["max_attempts_per_arm"] == 7 and config["maximum_cost_USD"] == 25 and
            config["max_input_tokens"] == 64000 and config["max_output_tokens"] == 16000, "Approved budget/limits changed")
    require(config["automatic_retries"] == 0 and config["fallback_models"] == [] and config["tools_enabled"] is False and config["automatic_reembedding"] is False, "Unauthorized retry/fallback/tool/reembedding")
    require(config["reasoning_effort"] == "high" and config["service_tier"] == "default" and config["pricing_USD_per_million_tokens"] == PRICES, "Settings/pricing differ from approved basis")
    expected_route = {role: "gpt-6-astra" if role == "Upstream" else "gpt-6.1-sol" for role in STAGES}
    require(config["routing"] == expected_route, "Routing differs from approved Astra/Sol roles")
    checks["pool_and_independent_arithmetic"] = {"status": "passed", **check_pool(snapshot["pool"])}
    inputs = snapshot["inputs"]
    require(set(inputs) == set(ARMS) and inputs[ARMS[0]]["common"] == inputs[ARMS[1]]["common"], "Common inputs differ between arms")
    require(inputs["history_omitted"]["batch_optimization_evidence"] is None, "History present in omitted input")
    visible_pool = inputs[ARMS[0]]["common"]["conditional_pool"]
    pool = snapshot["pool"]
    require({k: v for k, v in visible_pool.items() if k != "candidates"} ==
            {k: v for k, v in pool.items() if k not in {"candidates", "grid_groups", "feasible_candidates"}}, "Visible pool metadata differs from full frozen pool")
    require(len(visible_pool["candidates"]) == len(pool["candidates"]), "Visible candidate pool incomplete")
    for visible, full in zip(visible_pool["candidates"], pool["candidates"]):
        require(all(visible[k] == full[k] for k in visible if k != "calculated"), "Visible candidate differs from full pool")
        for key, value in visible["calculated"].items():
            expected = full["calculated"]["mixture_transit"]["gas_volume_pressure_coefficient_mL_bar_min"] if key == "gas_volume_pressure_coefficient_mL_bar_min" else full["calculated"]["mixture_transit"]["illustrations"] if key == "no_slip_transit_illustrations" else full["calculated"][key]
            require(value == expected, "Visible candidate calculation differs from full pool")
    for arm in ARMS:
        require(inputs[arm]["comparison_arm"] == arm and inputs[arm]["other_arm_outputs"] is None and inputs[arm]["flow_outcomes"] is None and
                inputs[arm]["access"] == {"retrieval": False, "previous_response_id": None, "shared_conversation": False}, f"{arm}: information boundary mismatch")
    history = inputs["history_informed"]["batch_optimization_evidence"]
    records = history["raw_records"]["rows"]
    entries = [r[0] for r in records]
    require(history["catalyst_system"] == "homogeneous" and history["experiment_mode"] == "batch" and len(entries) == len(set(entries)) == 116, "Homogeneous history missing/duplicated/misclassified")
    require(history["export_reconciliation"]["csv_count"] == 111 and history["export_reconciliation"]["xlsx_count"] == 116, "111/116 discrepancy not retained")
    record_ids = [history["raw_records"]["record_citation_id_format"].format(entry=e) for e in entries]
    require(digest(record_ids) == history["coverage"]["payload_record_ids_sha256"] == history["coverage"]["canonical_record_ids_sha256"], "History identity coverage mismatch")
    for source in history["source_registry"].values():
        require(snapshot["source_hashes"][source["path"]] == source["sha256"], "History registry hash mismatch")
    checks["matched_information_boundary"] = {"status": "passed", "homogeneous_records": 116,
        "limitation": "Structured equality and visible citation checks do not prove arbitrary prose is free of information leakage; scientific review remains necessary."}
    # The system prompt is read as an AST literal from frozen source, never executed.
    tree = ast.parse((directory / "source_snapshot/src/khu_sulfoxidation/campaign.py").read_text())
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in n.targets))
    for node in tree.body:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "SYSTEM":
            require(isinstance(node.op, ast.Add), "Unsupported frozen system-prompt expression")
            system += ast.literal_eval(node.value)
    previous = None
    ledger_paths = sorted((directory / "api/ledger").glob("*.json"))
    for sequence, path in enumerate(ledger_paths):
        ledger = read(path)
        require(path.name == f"{sequence:06d}.json" and ledger["sequence"] == sequence and ledger["previous_sha256"] == (None if previous is None else digest(previous)), "Ledger hash chain broken")
        require(ledger["configuration_sha256"] == digest(config), "Ledger configuration mismatch")
        require(len(ledger["entries"]) <= 14 and sum(Decimal(str(e["accounted_cost_USD"])) for e in ledger["entries"]) <= Decimal(25), "Historical ledger exceeded attempt/cost cap")
        previous = ledger
    ledger = previous
    require(ledger is not None and len(ledger["entries"]) == 14, "Final ledger does not contain exactly 14 attempts")
    attempts = ledger["entries"]
    require({(e["arm"], e["role"]) for e in attempts} == set(product(ARMS, STAGES)), "Repeated/missing role or arm")
    require({p.name for p in (directory / "api/attempts").iterdir()} == {e["attempt_id"] for e in attempts}, "Unledgered attempt archive")
    count_folders = list((directory / "api/token_counts").iterdir())
    for folder in count_folders:
        manifest = read(folder / "manifest.json")
        artifacts(folder, manifest["files_sha256"], ("manifest.json",))
        response = read(folder / "response.raw")
        require(manifest["endpoint"] == "https://api.openai.com/v1/responses/input_tokens" and manifest["generation"] is False and manifest["error"] is None and
                manifest["credential_redacted"] is False and manifest["received_sha256"] == file_hash(folder / "response.raw") and
                response["object"] == "response.input_tokens" and response["input_tokens"] == manifest["input_tokens"], "Invalid token-count archive")
    known = Decimal(0)
    accounted = Decimal(0)
    full_reserves = Decimal(0)
    for i, entry in enumerate(attempts, 1):
        arm, role = entry["arm"], entry["role"]
        require(entry["attempt_id"] == f"attempt-{i:04d}" and entry["status"] == "completed" and entry["halt_campaign"] is False, "Attempt incomplete/halted/out of sequence")
        folder = directory / "api/attempts" / entry["attempt_id"]
        artifacts(folder, entry["files_sha256"])
        request, payload, result, raw = [read(folder / name) for name in ("input.json", "request.json", "result.json", "response.raw")]
        stored = read(directory / arm / f"{role}.json")
        require(stored["request_sha256"] == digest(request) and request["source_hashes"] == snapshot["source_hashes"], f"{arm}/{role}: request provenance mismatch")
        user = json.loads(request["user_prompt"])
        own = nominations["arms"][arm]
        available = {} if role == "Upstream" else {"Upstream": own["Upstream"]} if role in STAGES[1:5] else {r: own[r] for r in STAGES[:STAGES.index(role)]}
        citations = COMMON_IDS | {f"review:{r}" for r in available} | (history_ids(history) if arm == "history_informed" else set())
        require(user == {"role": role, "visible_input": inputs[arm], "prior_within_arm_reviews": available, "visible_citation_ids": sorted(citations)}, f"{arm}/{role}: exact visible input/prior boundary mismatch")
        expected_payload = {"model": expected_route[role], "instructions": system,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": request["user_prompt"]}]}],
            "reasoning": {"effort": "high"}, "max_output_tokens": 16000, "service_tier": "default", "store": False,
            "stream": False, "truncation": "disabled", "text": {"format": {"type": "json_schema", "name": "khu_study_response", "strict": True, "schema": request["response_schema"]}}}
        require(request["system_prompt"] == system and payload == expected_payload and entry["model"] == expected_route[role], f"{arm}/{role}: API payload/settings mismatch")
        dispatch = read(folder / "dispatch.json")
        require(dispatch["endpoint"] == config["endpoint"] and dispatch["authorization_header_archived"] is False, "Dispatch endpoint/credential archive invalid")
        require(datetime.fromisoformat(original["frozen_at"]) <= datetime.fromisoformat(dispatch["at"]) <= datetime.fromisoformat(nominations["frozen_at"]), "Dispatch outside freeze chronology")
        require(raw["model"] == expected_route[role] and raw["service_tier"] == "default" and raw["status"] == "completed", "Provider route/status mismatch")
        text = "".join(p["text"] for out in raw["output"] if out["type"] == "message" for p in out["content"] if p["type"] == "output_text")
        require(result["ok"] is True and not result["errors"] and result["credential_redacted"] is False and result["response_sha256_received"] == file_hash(folder / "response.raw"), "Failed/redacted or mismatched provider response")
        require(result["text"] == text and json.loads(text) == result["parsed_json"] == stored["response"] == own[role], f"{arm}/{role}: frozen response differs from exact provider output")
        require(stored["api_metadata"] == {k: v for k, v in result.items() if k not in {"text", "parsed_json"}}, "Stage metadata mismatch")
        check_proposal(own[role], role, snapshot["pool"], citations, own["Upstream"])
        require(stored["validation"]["structural_validation"] == "passed" and stored["validation"]["laboratory_release"] is False, "Stage validation changed")
        guard = entry["input_guard"]
        if guard["basis"] == "utf8_bytes_of_entire_payload_plus_1024_headroom":
            bound = len(canonical(payload).encode()) + 1024
            require(guard["upper_bound"] == bound, "Byte-bound input guard mismatch")
        else:
            require(guard["basis"] == "official_count_including_text_schema_plus_1024_headroom", "Unknown input guard")
            count_dir = directory / "api/token_counts" / Path(guard["archive_path"]).name
            count_manifest = read(count_dir / "manifest.json")
            require(read(count_dir / "request.json") == {k: payload[k] for k in ("model", "instructions", "input", "text")}, "Token count did not include exact request/schema")
            require(guard["input_tokens"] == count_manifest["input_tokens"], "Token count mismatch")
            bound = count_manifest["input_tokens"] + count_manifest["framing_headroom_tokens"]
        require(bound == entry["input_token_upper_bound"] and bound <= 64000, "Input cap/preflight invalid")
        usage = raw["usage"]
        require(result["usage"] == usage and 0 <= usage["input_tokens"] <= 64000 and 0 <= usage["output_tokens"] <= 16000, "Usage cap exceeded")
        price = {k: Decimal(str(v)) for k, v in PRICES[entry["model"]].items()}
        reserve = (64000 * price["cache_write"] + 16000 * price["output"]) / 1000000
        require(Decimal(str(entry["reserved_cost_USD"])) == reserve == Decimal(str(result["reserved_cost_USD"])), "Reservation arithmetic mismatch")
        details = usage.get("input_tokens_details", {})
        if "cached_tokens" in details and "cache_write_tokens" in details:
            cached, written = details["cached_tokens"], details["cache_write_tokens"]
            require(cached >= 0 and written >= 0 and cached + written <= usage["input_tokens"], "Invalid cache usage")
            cost = ((usage["input_tokens"] - cached - written) * price["input"] + cached * price["cached_input"] + written * price["cache_write"] + usage["output_tokens"] * price["output"]) / 1000000
            require(Decimal(str(entry["cost_USD"])) == Decimal(str(result["cost_USD"])) == cost, "Actual cost arithmetic mismatch")
            known += cost
        else:
            require(entry["cost_USD"] is None and result["cost_USD"] is None, "Unknown cache cost silently inferred")
            cost = reserve
        require(Decimal(str(entry["accounted_cost_USD"])) == Decimal(str(result["accounted_cost_USD"])) == cost, "Accounted cost mismatch")
        accounted += cost
        full_reserves += reserve
    audit = nominations["api_audit"]
    require(audit["entries"] == attempts and audit["ledger_sha256"] == digest(ledger) and audit["configuration_sha256"] == digest(config), "Final API audit detached from ledger")
    require(audit["attempt_count"] == 14 and audit["per_arm_attempts"] == {a: 7 for a in ARMS} and audit["input_count_operations"] == len(count_folders) and audit["unresolved_inflight"] == [], "Final API audit counts invalid")
    close(audit["known_token_cost_USD"], float(known), "Total known token cost")
    close(audit["accounted_cost_USD"], float(accounted), "Total accounted cost")
    close(audit["remaining_budget_USD"], float(Decimal(25) - accounted), "Remaining budget")
    require(accounted <= 25 and full_reserves <= 25, "Campaign exceeded hard cap")
    checks["requests_responses_ledger_routing_budget"] = {"status": "passed", "attempts": 14, "input_count_operations": len(count_folders),
        "known_token_cost_USD": float(known), "accounted_cost_USD": float(accounted), "maximum_full_reservations_USD": float(full_reserves), "budget_USD": 25}
    checks["frozen_choices"] = {a: {k: nominations["arms"][a]["Chief"][k] for k in ("recommended_core_candidate_ids", "comparison_nomination_ids", "repeat_candidate_id")} for a in ARMS}
    return snapshot, nominations


def flat_condition(row):
    """Independent mapping of authoritative fields to every conditions.csv cell."""
    calc, setting, gas = row["calculated"], row["proposed_setpoints"], row["gas_reference_assumptions"]
    result = {k: row[k] for k in ("condition_id", "candidate_id", "comparison_arm", "table_order_only", "proposed_execution_order", "repeat_of_condition_id", "preparation_type", "status", "scenario_id", "time_coordinate", "oxygen_coordinate", "nominal_oxygen_equivalents", "source_configuration_sha256")}
    for key in ("liquid_only_space_time_min", "oxygen_relative_quantization_error", "inlet_reference_volume_index_min", "actual_liquid_contact_time_min", "substrate_mmol_min", "catalyst_mmol_min", "oxygen_mmol_min_supplied", "feed_mL_per_elapsed_hour_at_continuous_setting", "substrate_mmol_per_elapsed_hour_at_continuous_setting", "catalyst_mmol_per_elapsed_hour_at_continuous_setting"):
        result[key] = calc[key]
    result.update(proposed_liquid_flow_mL_min=setting["liquid_flow_mL_min"], proposed_gas_reference_flow_mL_min=setting["gas_reference_flow_mL_min"],
        calculated_oxygen_equivalents_supplied=calc["oxygen_equivalents_supplied"], assumed_gas_reference_temperature_K=gas["temperature_K"],
        assumed_gas_reference_pressure_bar_abs=gas["pressure_bar_abs"], assumed_air_oxygen_mole_fraction=gas["oxygen_mole_fraction"],
        proposed_temperature_C=setting["temperature_C"], selected_BPR=None, reactor_pressure_bar_abs=None, actual_light_power_setting=None,
        review_verdict=row["review"]["verdict"], rationale=row["review"]["rationale"], citations=" | ".join(row["review"]["citations"]))
    for item in calc["mixture_transit"]["illustrations"]:
        pressure = item["pressure_bar_abs"]
        result[f"mixture_transit_min_at_unconfirmed_{pressure:g}bar_abs_illustration"] = item["no_slip_mixture_transit_min_estimate"]
    return {k: "" if v is None else str(v) for k, v in result.items()}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_report(report, campaign, snapshot, nominations, checks):
    manifest = read(report / "manifest.json")
    artifacts(report, {k: v["sha256"] for k, v in manifest["files"].items()}, ("manifest.json",))
    for name, info in manifest["files"].items():
        require((report / name).stat().st_size == info["size_bytes"], f"Report size mismatch: {name}")
    for name in ("input_freeze.json", "nominations_freeze.json"):
        require(manifest["original_freeze_sha256"][name] == file_hash(report / name) == file_hash(campaign / name), "Report freeze provenance mismatch")
    require(manifest["input_snapshot_sha256"] == digest(snapshot) and manifest["source_hashes"] == snapshot["source_hashes"], "Report source provenance mismatch")
    require(read(report / "preserved_pool.json") == snapshot["pool"] and read(report / "api_audit.json") == nominations["api_audit"], "Report pool/audit mismatch")
    authoritative = read(report / "authoritative_conditions.json")
    rows, comparison = authoritative["conditions"], authoritative["comparison"]
    require(authoritative["status"] == STATUS and authoritative["input_snapshot_sha256"] == digest(snapshot) and
            authoritative["nomination_freeze_file_sha256"] == file_hash(campaign / "nominations_freeze.json") and authoritative["scenario"] == snapshot["pool"]["scenario"], "Authoritative provenance/status mismatch")
    chiefs = {arm: nominations["arms"][arm]["Chief"] for arm in ARMS}
    core = chiefs["history_informed"]["recommended_core_candidate_ids"]
    additional = [c for c in chiefs["history_omitted"]["comparison_nomination_ids"] if c not in core]
    expected_ids = core + [chiefs["history_informed"]["repeat_candidate_id"]] + additional
    require([r["candidate_id"] for r in rows] == expected_ids and len(rows) == 7 + len(additional), "Report does not implement core + independent repeat + deduplicated omitted nominations")
    require(len({r["condition_id"] for r in rows}) == len(rows), "Duplicate report run ID")
    primary = {r["candidate_id"]: r["condition_id"] for r in rows if r["repeat_of_condition_id"] is None}
    candidates = {c["candidate_id"]: c for c in snapshot["pool"]["candidates"]}
    for index, row in enumerate(rows, 1):
        candidate = candidates[row["candidate_id"]]
        for key in ("candidate_id", "scenario_id", "time_coordinate", "oxygen_coordinate", "nominal_oxygen_equivalents", "proposed_setpoints", "calculated"):
            require(row[key] == candidate[key], f"{row['condition_id']}: authoritative candidate value changed: {key}")
        require(row["source_configuration_sha256"] == snapshot["pool"]["configuration_sha256"] and row["table_order_only"] == index and row["proposed_execution_order"] is None, "Configuration/table order mismatch")
        require(row["status"] == STATUS and row["actual_conditions"] is None and row["actual_outcomes"] is None, "Proposed result changed into actual")
        reviews = {r["candidate_id"]: r for r in chiefs[row["comparison_arm"]]["candidate_reviews"]}
        require(row["review"] == reviews[row["candidate_id"]], "Report rationale/citations differ from frozen Chief")
    repeats = [r for r in rows if r["repeat_of_condition_id"] is not None]
    require(len(repeats) == 1 and repeats[0]["repeat_of_condition_id"] == primary[chiefs["history_informed"]["repeat_candidate_id"]] and repeats[0]["preparation_type"] == "independent_preparation_and_operation", "Independent repeat missing/misassigned")
    require(comparison == manifest["comparison"] and comparison["additional_omitted_runs"] == len(additional), "Manifest comparison mismatch")
    for arm in ARMS:
        items = comparison["arms"][arm]
        require([r["candidate_id"] for r in items] == chiefs[arm]["comparison_nomination_ids"] and [r["rank"] for r in items] == [1, 2], "Report altered frozen nomination ranks")
        require(all(r["condition_id"] == primary[r["candidate_id"]] for r in items), "Overlapping nominations do not share their primary measurement")
    require(manifest["condition_count"] == len(rows) and manifest["unique_condition_count"] == len(set(expected_ids)), "Manifest run counts ignore repeat or overlap")
    require(read_csv(report / "conditions.csv") == [flat_condition(r) for r in rows], "conditions.csv differs from authoritative JSON")
    results = read(report / "results_template.json")["records"]
    require(len(results) == len(rows), "Result template run count mismatch")
    for row, template in zip(rows, results):
        require(template["proposed_conditions"] == row and all(template[k] == row[k] for k in ("condition_id", "candidate_id", "repeat_of_condition_id")), "Results proposal/identity mismatch")
        require(all(v is None for k, v in template.items() if k not in {"proposed_conditions", "condition_id", "candidate_id", "repeat_of_condition_id"}), "Unknown actual/result field is not blank")
    expected_results_csv = [{k: "" if v is None else str(v) for k, v in r.items() if k != "proposed_conditions"} for r in results]
    require(read_csv(report / "results_template.csv") == expected_results_csv, "Results CSV/JSON mismatch")
    prep = read(report / "feed_preparation.json")
    require(prep == authoritative["feed_preparation"], "Feed preparation JSON mismatch")
    recipe = snapshot["pool"]["scenario"]["shared_recipe"]
    require(prep["substrate"] == recipe["substrate"] and prep["catalyst"] == recipe["catalyst"], "Feed identity mismatch")
    amount = recipe["substrate_concentration_M"] * prep["final_volume_example_mL"]
    catalyst = amount * recipe["catalyst_loading_mol_percent"] / 100
    close(prep["substrate_mmol_example"], amount, "Feed substrate amount")
    close(prep["catalyst_mmol_example"], catalyst, "Feed catalyst amount")
    close(prep["catalyst_concentration_mM"], recipe["substrate_concentration_M"] * recipe["catalyst_loading_mol_percent"] * 10, "Feed catalyst concentration")
    for compound, mmol in ((prep["substrate"], amount), (prep["catalyst"], catalyst)):
        close(prep["pure_equivalent_mass_mg"][compound], mmol * prep["molecular_weight_g_mol"][compound], f"{compound} pure-equivalent mass")
        require(prep["actual_bottle_purity_fraction"][compound] is None, "Unknown bottle purity assigned")
    require(all(prep[k] is None for k in ("actual_weighed_masses_mg", "actual_bottle_identity_confirmed", "solubility_confirmed", "total_campaign_feed_volume_mL", "total_substrate_mmol", "total_catalyst_mmol")), "Missing feed data treated as known")
    dispositions = read(report / "pool_selection_dispositions.json")
    require(len(dispositions) == len(candidates) and {d["candidate_id"] for d in dispositions} == set(candidates), "Report omitted candidate dispositions")
    for item in dispositions:
        candidate = candidates[item["candidate_id"]]
        expected = "selected_for_proposed_measurement" if item["candidate_id"] in set(expected_ids) else "rejected_by_pool_constraints" if candidate["status"] == "rejected" else "eligible_not_selected"
        require(item["disposition"] == expected and item["pool_rejection_reasons"] == candidate["rejection_reasons"], "Pool disposition/rejection changed")
    checks["report_manifest_authoritative_csv_results_and_feed"] = {"status": "passed", "planned_runs": len(rows), "unique_conditions": len(set(expected_ids)), "additional_omitted_runs": len(additional),
        "pdf_layout": "Not checked by this script; visual inspection is separate.", "report_manifest_sha256": file_hash(report / "manifest.json")}


def verify(campaign, report=None):
    result = {"schema_version": "khu_read_only_campaign_verification_v1", "verified_at": datetime.now(timezone.utc).isoformat(),
              "campaign_directory": str(campaign.resolve()), "report_directory": str(report.resolve()) if report else None,
              "verifier_sha256": file_hash(Path(__file__)), "checks": {},
              "scope_limitations": ["No laboratory release or experimental validation is asserted.",
                  "No network or model calls; no API credential is read; no gateway lock is acquired.",
                  "Structural boundaries and citations cannot prove arbitrary-text leakage absent.",
                  "Token prices exclude taxes and are not an invoice reconciliation; PDF visual QA is separate."]}
    if not (campaign / "nominations_freeze.json").is_file():
        result.update(status="not_run", reason="Final nominations_freeze.json is absent; campaign is unfinished. Run again only after both arms freeze.")
        return result
    try:
        # Snapshot the final bytes/ledger inventory to detect concurrent mutation.
        start = {name: file_hash(campaign / name) for name in ("input_freeze.json", "nominations_freeze.json")}
        ledger_before = sorted(p.name for p in (campaign / "api/ledger").glob("*.json"))
        snapshot, nominations = check_campaign(campaign, result["checks"])
        if report:
            check_report(report, campaign, snapshot, nominations, result["checks"])
        require(start == {name: file_hash(campaign / name) for name in start} and ledger_before == sorted(p.name for p in (campaign / "api/ledger").glob("*.json")), "Campaign changed during read-only verification")
        result.update(status="passed", final_freeze_sha256=start)
    except (ValueError, TypeError, KeyError, IndexError, StopIteration, OSError) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Create a new JSON verification report exclusively; never overwrite")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new path")
    result = verify(args.campaign, args.report_dir)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return {"passed": 0, "failed": 1, "not_run": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
