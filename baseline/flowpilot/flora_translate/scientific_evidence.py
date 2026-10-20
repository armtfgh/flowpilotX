"""Source-linked evidence for the opt-in scientific policy; no invented kinetics."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy

from flora_translate.component_identity import component_name, component_key, SOLVENT_IDENTITY_ALIASES


MODE = "scientific_v2"
UPSTREAM_POLICY = """
SCIENTIFIC PREVIEW POLICY: Preserve explicit stage-specific batch times,
temperatures, charges and later additions, with original compound names and
abbreviations. Do not infer a rate-limiting step from equal holding times.
Do not predict conversion, assign a kinetic constant, or demand a time-reduction
factor without measured evidence. Label mechanistic and rate-limiting proposals
as hypotheses. Set no mandatory intensification target. Source facts, chemist
hypotheses and model inferences must remain distinguishable.
When a gas-feed ratio is delegated to the agent, provide a positive numeric
molar_equiv as a proposed screening setting and justify it, even if actual gas
consumption is unknown. Do not put null or a range in molar_equiv. Distinguish the
chosen delivery ratio from measured stoichiometric demand. Keep the same ratio
in global stream_logic and its stage feed; it must not default silently to 1.
"""


def explicit_gas_ratios(data):
    """Require a deliberate gas-feed setting before defaults can erase nulls."""
    import math
    normalized = deepcopy(data)
    gas_feeds = {s.get("stream_label"): s for s in normalized.get("stream_logic", []) if s.get("phase") == "gas"}

    def validate(feed):
        value = feed.get("molar_equiv")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Gas stream {feed.get('stream_label')}: provide a positive numeric proposed molar_equiv with justification; unknown consumption must not default to 1.")

    for feed in gas_feeds.values():
        validate(feed)
    for stage in normalized.get("stages", []):
        for feed in stage.get("feed_streams", []):
            if feed.get("phase") != "gas" or feed.get("delivery_mode") == "carried_from_previous":
                continue
            global_feed = gas_feeds.get(feed.get("stream_label"))
            if feed.get("molar_equiv") is None and global_feed:
                feed["molar_equiv"] = global_feed["molar_equiv"]
                feed["molar_equiv_basis"] = "explicit global stream setting"
            validate(feed)
            if global_feed and feed["molar_equiv"] != global_feed["molar_equiv"]:
                raise ValueError(f"Conflicting stage and global gas ratios for stream {feed.get('stream_label')}")
    return normalized


def enabled(plan):
    return (getattr(plan, "scientific_context", None) or {}).get("mode") == MODE


def aliases(name):
    base = component_name(name).rstrip(". ")
    names = [base]
    aqueous_buffer = re.fullmatch(r"pH\s+(\d+(?:\.\d+)?)\s+aqueous\s+buffer", base, re.I)
    if aqueous_buffer:
        names.append(f"pH {aqueous_buffer[1]} buffer")
    physical_state = re.search(r"\s+\((?:aqueous|aq\.?|liquid|solid|gas)\)$", base, re.I)
    if physical_state:
        names.append(base[:physical_state.start()].strip())
    abbreviation = re.search(r"\s+\(([A-Za-z0-9][A-Za-z0-9-]{1,20})\)$", base)
    if abbreviation and sum(c.isupper() for c in abbreviation[1]) < 2:
        abbreviation = None
    if not abbreviation:
        abbreviation = re.search(r"\(([A-Z][A-Z0-9-]{1,12})\)$", base)
    if abbreviation:
        names.extend([abbreviation[1], base[:abbreviation.start()].strip()])
    leading = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9-]{1,20})\s+\((.+)\)", base)
    if not leading:
        leading = re.fullmatch(r"([A-Z][A-Z0-9-]{1,12})\s*\((.+)\)", base)
    if leading and sum(c.isupper() for c in leading[1]) >= 2:
        names.extend([leading[1], leading[2]])
    keys = {component_key(n) for n in names}
    for group in SOLVENT_IDENTITY_ALIASES:
        if keys.intersection(component_key(n) for n in group):
            names.extend(group)
    return list(dict.fromkeys(n for n in names if n))


def identify(name, components):
    keys = {component_key(a) for a in aliases(name)}
    return next((c for c in components if keys.intersection(component_key(a) for a in c["aliases"])), None)


def source_context(batch, plan):
    """Locate explicit timed reaction holds and reagent mentions in original text.

    This deliberately supports explicit sequential prose, not arbitrary chemistry
    NER. Unresolved hold counts/identities require confirmation, not guessed facts.
    """
    text = batch.raw_text or batch.reaction_description
    holds = []
    pattern = r"\bfor\s+(?:another\s+|an?\s+additional\s+)?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b"
    for m in re.finditer(pattern, text, re.I):
        prefix = text[max(0, m.start() - 240):m.start()]
        verbs = list(re.finditer(r"cool\w*|stirr\w*|heat\w*|irradiat\w*|react\w*|reflux\w*|incubat\w*|maintain\w*", prefix, re.I))
        if not verbs or verbs[-1][0].lower().startswith("cool"):
            continue
        minutes = float(m[1]) * (60 if m[2].lower().startswith("h") else 1)
        if minutes > 0:
            segment_start = holds[-1]["source_end"] if holds else 0
            temperatures = list(re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:\u00b0|deg(?:rees?)?\.?\s*|o\s*)?C\b", text[segment_start:m.start()]))
            temperature = temperatures[-1] if temperatures else None
            holds.append({"stage_number": len(holds) + 1, "batch_time_min": minutes,
                          "source_start": m.start(), "source_end": m.end(), "quote": m[0],
                          "batch_temperature_C": float(temperature[1]) if temperature else None,
                          "temperature_quote": temperature[0] if temperature else None,
                          "temperature_source_start": segment_start + temperature.start() if temperature else None,
                          "temperature_basis": "nearest explicit temperature preceding this hold" if temperature else "not stage-resolved"})
    count = len(plan.stages) or 1
    issues = []
    if len(holds) != count:
        issues.append(f"Protocol has {len(holds)} verified timed holds, but chemistry plan has {count} stages; confirm stage definitions.")
    components = []
    for reagent in plan.reagents:
        names = aliases(reagent.name)
        mentions = []
        for name in names:
            for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text, re.I):
                number = 1 + sum(h["source_end"] < match.start() for h in holds)
                if number <= count:
                    mentions.append({"stage_number": number, "source_start": match.start(),
                                     "source_end": match.end(), "quote": match[0]})
        components.append({"name": reagent.name, "aliases": names, "role": reagent.role,
                           "addition_stages": sorted({m["stage_number"] for m in mentions}),
                           "source_mentions": mentions})
    from flora_translate.chemistry_contract import _protocol_reagent_gases
    for species in _protocol_reagent_gases(batch):
        if identify(species, components):
            continue
        names = {"air": ["air"], "O2": ["O2", "O\u2082", "oxygen"], "H2": ["H2", "H\u2082", "hydrogen"]}.get(species, [species])
        mentions = []
        for name in names:
            for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text, re.I):
                number = 1 + sum(h["source_end"] < match.start() for h in holds)
                if number <= count:
                    mentions.append({"stage_number": number, "source_start": match.start(), "source_end": match.end(), "quote": match[0]})
        components.append({"name": species, "aliases": names, "role": "physical reagent gas",
                           "addition_stages": sorted({m["stage_number"] for m in mentions}), "source_mentions": mentions})
    return {"mode": MODE, "schema_version": "flowpilot_scientific_evidence_v1",
            "protocol_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "timed_holds": holds, "components": components, "issues": issues,
            "kinetics_status": "uncharacterized", "predicted_yield_pct": None,
            "predicted_conversion_pct": None,
            "policy": "Batch times anchor an exploratory screen, not measured flow kinetics. No automatic intensification factor or inferred conversion.",
            "objective_policy": "chemist objective first; no global shortest-time or 30-percent conversion floor"}


def preserve_stage_additions(plan, stages, context, decisions):
    """Reject global-list leakage and deduplicate explicit aliases before freezing."""
    components = context["components"]
    gas_stage = context.get("gas_stage_deviation", {}).get("requested_stage")
    gas_components = {c["name"] for s in stages for f in s.feed_streams if f.phase == "gas"
                      for name in f.reagents for c in components if identify(name, [c])}
    for stage in stages:
        held = next((h for h in context["timed_holds"] if h["stage_number"] == stage.stage_number), None)
        if held and len(context["timed_holds"]) == len(stages):
            stage.batch_time_h = held["batch_time_min"] / 60
        for feed in stage.feed_streams:
            if feed.delivery_mode == "carried_from_previous":
                continue
            kept, seen = [], set()
            for name in feed.reagents:
                c = identify(name, components)
                allowed_stages = [gas_stage] if gas_stage and feed.phase == "gas" else (c["addition_stages"] if c else [])
                if c and allowed_stages and stage.stage_number not in allowed_stages:
                    decisions.append({"decision": "remove_wrong_stage_component", "stage": stage.stage_number,
                                      "component": name, "allowed_stages": c["addition_stages"], "source_evidence": c["source_mentions"]})
                    continue
                key = c["name"].casefold() if c else component_key(name)
                if key in seen:
                    continue
                seen.add(key)
                kept.append(name)
                if not c or not c["source_mentions"]:
                    context["issues"].append(f"Stage {stage.stage_number}: no exact protocol identity evidence for {component_name(name)}.")
            feed.reagents = kept
            feed.introduction_stage = stage.stage_number
            feed.source_evidence = [m["quote"] for c in components for m in c["source_mentions"]
                                    if m["stage_number"] == stage.stage_number and c["name"].casefold() in seen]
            feed.requirement_authority = "protocol_fact" if feed.source_evidence else "model_inference"
        stage.feed_streams = [f for f in stage.feed_streams if f.reagents]
    # Every non-solvent protocol component must be assigned to its stated feed stage.
    for c in components:
        if c["role"].lower() in {"solvent", "product", "byproduct", "intermediate"}:
            continue
        required_stages = [gas_stage] if gas_stage and c["name"] in gas_components else c["addition_stages"]
        for number in required_stages:
            feeds = [f for s in stages if s.stage_number == number for f in s.feed_streams]
            if not any(identify(n, [c]) for f in feeds for n in f.reagents):
                context["issues"].append(f"Protocol component {c['name']} has no feed in stage {number}.")
    context["issues"] = list(dict.fromkeys(context["issues"]))
    plan.scientific_context = context


def assess_analogies(analogies, plan):
    """Retrieval similarity alone is not authority to transfer a rate constant."""
    rows = []
    for a in analogies:
        full = a.get("full_record") or {}
        cls = str((a.get("metadata") or {}).get("chemistry_class") or "")
        source_phase = (a.get("metadata") or {}).get("phase_regime")
        same_class = bool(cls) and cls.casefold() == plan.reaction_class.casefold()
        same_light = ("photo" in cls.casefold()) == any(s.requires_light for s in plan.stages)
        rows.append({"record_id": a.get("record_id"), "retrieval_score": a.get("final_score", a.get("score")),
                     "source_class": cls, "source_phase": source_phase,
                     "same_class": same_class, "same_light_regime": same_light,
                     "usable_as_kinetic_evidence": False,
                     "reason": "Class or light regime differs." if not same_class or not same_light else
                     "Related precedent only: no independently validated stage-resolved kinetic transfer model.",
                     "batch_baseline": deepcopy(full.get("batch_baseline")),
                     "flow_optimized": deepcopy(full.get("flow_optimized"))})
    return rows


def scientific_class(batch, plan):
    """Use structured reaction identity; element abbreviations require boundaries."""
    identity = " ".join([plan.reaction_class, plan.mechanism_type, batch.reaction_description]).casefold()
    if re.search(r"\bamide\b|aminolysis|acyl substitution", identity):
        return "amide_formation"
    if re.search(r"\b(?:suzuki|heck|buchwald|sonogashira)\b|cross[- ]coupling", identity):
        return "cross-coupling"
    if re.search(r"\bphotoredox\b|photocatal", identity):
        return "photoredox"
    if any(s.requires_light for s in plan.stages):
        return "photochem"
    if re.search(r"\bhydrogenation\b", identity):
        return "hydrogenation"
    return "thermal" if re.search(r"thermal|heat|stirr|reflux", identity) else "unknown"
