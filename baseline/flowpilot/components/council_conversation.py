"""Council Conversation Generator.

Takes the raw ENGINE council messages and generates a human-readable
conversational discussion between the agents, using Claude.

The output reads like a real engineering review meeting — agents
challenge each other, explain reasoning, propose corrections, and
reach consensus through deliberation.
"""

from __future__ import annotations
import json
import logging

logger = logging.getLogger("flora.council_conversation")

_AGENT_PERSONA = {
    "DesignCalculator": (
        "Physics Engine",
        "You are the Physics Engine — a rigorous numerical analyst. "
        "You speak in hard numbers: equations, calculated values, error percentages. "
        "You are blunt when numbers don't match. You cite your equations explicitly."
    ),
    "KineticsAgent": (
        "Dr. Kinetics",
        "You are Dr. Kinetics — a reaction engineering specialist. "
        "You think in residence times, conversion, and intensification factors. "
        "You compare to literature, question overly optimistic assumptions, "
        "and insist on physically reasonable reaction timescales."
    ),
    "FluidicsAgent": (
        "Dr. Fluidics",
        "You are Dr. Fluidics — a fluid mechanics expert. "
        "You check pressure drops, Reynolds numbers, and pump constraints. "
        "You flag turbulent flow risks and hardware limitations immediately. "
        "You are pragmatic: if something won't fit in the lab, say so."
    ),
    "SafetyCriticAgent": (
        "Safety Officer",
        "You are the Safety Officer — your job is to stop disasters before they happen. "
        "You raise flags on incompatible materials, temperatures near boiling points, "
        "missing BPRs, and hazardous conditions. You are conservative by nature."
    ),
    "ChemistryValidator": (
        "Dr. Chemistry",
        "You are Dr. Chemistry — a synthetic chemist turned flow specialist. "
        "You verify that stream assignments match the mechanism, "
        "that sensitive reagents are protected, and that the chemistry "
        "makes sense before it goes near a pump."
    ),
    "ProcessArchitectAgent": (
        "Process Architect",
        "You are the Process Architect — you design the final unit operation sequence. "
        "You synthesize everyone's inputs into a coherent, buildable process. "
        "You close the meeting with the final validated design."
    ),
}


def _format_messages_for_prompt(council_messages: list[dict], proposal: dict, rounds: int) -> str:
    """Format council messages into a structured prompt for conversation generation."""
    lines = [
        f"This is a {rounds}-round engineering council review of a flow chemistry proposal.",
        "",
        "PROPOSED DESIGN:",
        f"  Residence time: {proposal.get('residence_time_min', '?')} min",
        f"  Flow rate: {proposal.get('flow_rate_mL_min', '?')} mL/min",
        f"  Reactor volume: {proposal.get('reactor_volume_mL', '?')} mL",
        f"  Tubing: {proposal.get('tubing_material', '?')} {proposal.get('tubing_ID_mm', '?')} mm ID",
        f"  Temperature: {proposal.get('temperature_C', '?')} °C",
        f"  BPR: {proposal.get('BPR_bar', 0)} bar",
        f"  Reactor type: {proposal.get('reactor_type', '?')}",
        "",
        "COUNCIL MESSAGES (raw):",
    ]
    for m in council_messages:
        agent = m.get("agent", "?")
        status = m.get("status", "?")
        field = m.get("field", "")
        concern = m.get("concern", "")
        value = m.get("value", "")
        suggestion = m.get("suggested_revision", "")
        line = f"  [{agent}] {status} on {field}: {concern or value}"
        if suggestion:
            line += f" → suggests: {suggestion}"
        lines.append(line)

    return "\n".join(lines)


def generate_council_conversation(
    council_messages: list[dict],
    proposal: dict,
    rounds: int,
) -> str:
    """Call Claude to generate a human conversational discussion from council messages.

    Returns the conversation as a markdown string.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        logger.warning(f"Cannot generate council conversation: {e}")
        return ""

    context = _format_messages_for_prompt(council_messages, proposal, rounds)

    # Build the list of agents that actually spoke
    agents_present = list(dict.fromkeys(
        m.get("agent", "") for m in council_messages if m.get("agent")
    ))
    persona_lines = []
    for ag in agents_present:
        name, _ = _AGENT_PERSONA.get(ag, (ag, ""))
        persona_lines.append(f"- {name} ({ag})")

    system = (
        "You are a technical writer generating a realistic engineering council "
        "meeting transcript for a flow chemistry process review. "
        "Write a natural, conversational discussion where the agents speak "
        "in first person, challenge each other, explain their reasoning, "
        "and reach consensus. Use the raw council messages as the factual basis — "
        "do not invent new facts or change numbers. "
        "Format as a dialogue with speaker names in bold, e.g.: "
        "**Dr. Kinetics:** I've looked at the residence time...\n"
        "Make it read like a real technical meeting. Include disagreements, "
        "corrections being accepted, agents asking each other questions. "
        "3-5 paragraphs of dialogue per round. Be specific with numbers."
    )

    user = (
        f"Generate a conversational council meeting discussion based on these facts:\n\n"
        f"{context}\n\n"
        f"Agents present: {', '.join(agents_present)}\n\n"
        f"Write the discussion as it would happen across {rounds} round(s). "
        f"Show how the design evolved — what was rejected, why, what was corrected. "
        f"End with the final agreed design parameters. "
        f"Keep it focused and technical but readable. Use markdown formatting."
    )

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    except Exception as e:
        logger.error(f"Council conversation generation failed: {e}")
        return ""
