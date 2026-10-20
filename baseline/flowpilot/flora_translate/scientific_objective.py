"""Versioned screening intent, separate from kinetics and hard constraints."""
from __future__ import annotations

import hashlib
import json
import re

PRIORITIES = {"auto", "balanced", "yield_priority", "throughput_priority"}
DESCRIPTIONS = {
    "balanced": "Establish an interpretable baseline and compare shorter and longer stage holds.",
    "yield_priority": "Investigate conversion adequacy in every stage, including longer holds; weigh degradation and stability risks. Do not assume longer means higher yield.",
    "throughput_priority": "Investigate shorter holds and higher feed throughput while retaining baseline controls; do not assume product throughput without measured yield.",
}


def resolve_objective(objective, intake=None):
    data = intake.model_dump() if hasattr(intake, "model_dump") else (intake or {})
    requested = data.get("screening_priority", "auto")
    if requested not in PRIORITIES:
        raise ValueError("Unknown screening_priority")
    text = " ".join(str(objective or "").lower().split())
    matches = []
    # Auto recognizes only explicit optimization language, not incidental 'high yield'.
    pattern = r"\b(?:maximi[sz]e|prioriti[sz]e|optimi[sz]e)\b[^.;\n]{0,55}?\b(yield|conversion|throughput|productivity)\b"
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start() - 30):match.start()]
        clause = re.split(r"[.;!?]", prefix)[-1]
        if re.search(r"\b(?:not|never|avoid|without|don.t)\b", clause):
            continue
        mode = "yield_priority" if match[1] in {"yield", "conversion"} else "throughput_priority"
        matches.append({"priority": mode, "quote": match[0]})
        tail = re.split(r"[.;!?]", text[match.end():])[0]
        joint = re.match(r"\s+(?:and|or)\s+(yield|conversion|throughput|productivity)\b", tail)
        if joint:
            joint_mode = "yield_priority" if joint[1] in {"yield", "conversion"} else "throughput_priority"
            matches.append({"priority": joint_mode, "quote": joint[0]})
    time_pattern = r"\b(?:reduce[sd]?|shorten|minimi[sz]e)\s+(?:(?:the|overall|total|reaction|processing|residence)\s+)*time\b"
    for match in re.finditer(time_pattern, text):
        clause = re.split(r"[.;!?]", text[max(0, match.start() - 30):match.start()])[-1]
        if not re.search(r"\b(?:not|never|avoid|without|don.t)\b", clause):
            matches.append({"priority": "throughput_priority", "quote": match[0]})
    detected = {m["priority"] for m in matches}
    priority = requested if requested != "auto" else next(iter(detected)) if len(detected) == 1 else "balanced"
    basis = "chemist_selected" if requested != "auto" else "explicit_objective_phrase" if len(detected) == 1 else "balanced_default"
    return {"version": "flowpilot_screening_intent_v1.1", "priority": priority,
            "requested": requested, "basis": basis, "objective": objective,
            "matched_phrases": matches, "selection_instruction": DESCRIPTIONS[priority],
            "interpretation_note": "Ambiguous or unrecognized wording uses a balanced screen; select a priority explicitly to override." if basis == "balanced_default" else "Priority changes exploration and decision trade-offs, not chemical facts or predicted yield.",
            "hypotheses_are_not_constraints": True}


def screen_patterns(priority, stage_count):
    if stage_count == 1:
        values = {"balanced": (.25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8),
                  "yield_priority": (.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 6, 8),
                  "throughput_priority": (.125, .2, .25, .3, .4, .5, .6, .75, 1, 1.25, 1.5, 2)}
        return [(v,) for v in values[priority]]
    if stage_count != 2:
        raise ValueError("Objective screens support one or two stages")
    controls = [(.5, .5), (.5, 1), (1, .5), (1, 1)]
    extensions = {
        "balanced": [(1, 2), (2, 1), (2, 2), (.5, 2), (2, .5), (1, 4), (4, 1), (4, 4)],
        "yield_priority": [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)],
        "throughput_priority": [(.25, .5), (.5, .25), (.25, 1), (1, .25), (.75, .75), (.5, .75), (.75, .5), (.25, .25)],
    }
    return controls + extensions[priority]


def source_stage_temperature(plan, stage_number):
    hold = next((h for h in plan.scientific_context.get("timed_holds", []) if h["stage_number"] == stage_number), {})
    return hold.get("batch_temperature_C")


def objective_fit(proposal, plan, targets):
    stages = proposal.stage_parameters
    return {"stage_hold_ratios": [round(s["residence_time_min"] / (st.batch_time_h * 60), 5)
                                  for s, st in zip(stages, plan.stages)],
            "target_stage_times_min": targets,
            "total_time_min": sum(s["residence_time_min"] for s in stages),
            "stage_temperature_changes": [{"stage_number": s["stage_number"], "source_C": source_stage_temperature(plan, s["stage_number"]), "screen_C": s["temperature_C"]} for s in stages],
            "interpretation": "Hold ratios describe experimental exposure only; no kinetic or yield ranking."}


def answer_effects(intake, intent, selection):
    data = intake.model_dump() if hasattr(intake, "model_dump") else (intake or {})
    effects = []
    for a in data.get("answers", []):
        q = a["question_id"]
        effect = {
            "Q-OBJ-001": f"Resolved to {intent['priority']} ({intent['basis']}); controls the 12-point screen and council trade-offs.",
            "Q-HYP-001": "Supplied to every reviewer as hypotheses, not measured rates or mandatory equipment changes.",
            "Q-HIST-001": "Measured evidence is supplied to the council; this policy does not fit a kinetic model automatically.",
            "Q-INV-001": "Equipment selection and feasibility gates use the frozen inventory.",
            "Q-CONSTR-001": "Structured operating limits constrain candidate realization; free-text limits still require verified normalization.",
            "Q-CHEM-001": "Identity and order of addition inform the upstream chemistry plan and source reconciliation.",
        }.get(q, "Included in authority-labeled council context; no separate deterministic effect asserted.")
        if a["status"] == "unavailable":
            effect = "Explicitly unavailable; no measured value or equipment capability is inferred from this answer."
        effects.append({"question_id": q, "status": a["status"], "effect": effect,
                        "scope": "binding and review provenance, not a causal sensitivity experiment"})
    return {"priority": intent["priority"], "questions": effects,
            "selection": selection, "note": "Equivalent objectives may select the same design. A changed answer does not require an artificial topology change."}


def pool_fingerprint(pool):
    rows = [{"id": c["candidate_id"], "stages": c["objective_fit"],
             "reactors": [s["reactor_equipment_id"] for s in c["proposal"]["stage_parameters"]]} for c in pool]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
