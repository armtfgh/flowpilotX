"""
FLORA Translate — Conversational Agent.

Wraps the translate pipeline with multi-turn conversation logic.
Each call to .process() classifies the user intent and acts:

  TRANSLATE  — fresh batch protocol → run translate()
  REVISE     — modification request → re-run translate() with revision injected
  ANSWER     — question about current result → answer from context, no re-run
  ASK        — agent needs more info → return clarification questions
  NEW_QUERY  — user starts completely fresh

The FULL current design (streams, conditions, reasoning, chemistry plan) is
injected into the system prompt on every call — so the agent always has
complete situational awareness regardless of conversation length.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("flora.conversation")

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_messages

# ── Base system prompt (static part) ──────────────────────────────────────────

_BASE_SYSTEM = """\
You are FLORA, an expert flow chemistry AI copilot. You help researchers translate
batch chemistry protocols into continuous flow processes, answer detailed questions
about your designs, and revise them on request.

Your tone: precise, scientific, collaborative. Use correct chemistry terminology.

## Capabilities
1. **Translate** a batch protocol → complete flow process design
2. **Revise** an existing design on request (add unit ops, change conditions, etc.)
3. **Answer** specific questions about the current design and your reasoning
4. **Ask** for missing critical information before translating

## Classification rules
- User provides a batch protocol with no prior design → TRANSLATE
- Design exists + user requests any change/modification → REVISE
- Design exists + user asks a question about it → ANSWER
- Critical information is missing before translating → ASK
- User explicitly starts a completely different reaction → NEW_QUERY

## Output format — JSON only, no prose outside the JSON block
{
  "intent": "TRANSLATE | REVISE | ANSWER | ASK | NEW_QUERY",
  "message": "Your full scientific response. For ANSWER: be detailed and cite specific values from the design. For REVISE: explain what you will change and why.",
  "revision_instructions": "Precise technical description of changes needed (REVISE only, else null)",
  "questions": ["question 1", "question 2"],
  "needs_retranslate": true | false
}

Rules:
- "message" must always be present, non-empty, and scientifically precise
- For ANSWER: reference exact values from the Current Design (streams, conditions, reasoning)
- For REVISE: "revision_instructions" must be technically specific enough to rebuild the design
- "needs_retranslate": true for TRANSLATE / REVISE / NEW_QUERY; false for ANSWER / ASK
"""


# ── Full result context builder ────────────────────────────────────────────────

def _build_full_context(
    original_query: str | None,
    revisions: list[str],
    result: dict | None,
) -> str:
    """Build a comprehensive context block to inject into the system prompt.

    Includes: original protocol, all revisions, full proposal (streams,
    conditions, reasoning per field), chemistry plan (mechanism, stream logic,
    sensitivities), and unit operations sequence.
    """
    if not result and not original_query:
        return ""

    sections = []

    # ── Original query ────────────────────────────────────────────────────────
    if original_query:
        sections.append(f"## Original Batch Protocol\n{original_query}")

    # ── Revisions applied ──────────────────────────────────────────────────────
    if revisions:
        sections.append(
            "## Revisions Applied So Far\n" +
            "\n".join(f"  {i+1}. {r}" for i, r in enumerate(revisions))
        )

    if not result:
        return "\n\n".join(sections)

    proposal = result.get("proposal", {})
    plan     = result.get("chemistry_plan", {})
    topo     = result.get("process_topology", {})
    conf     = result.get("confidence", "?")

    # ── Core conditions ───────────────────────────────────────────────────────
    conditions = [
        f"  Reactor type       : {proposal.get('reactor_type', '?')}",
        f"  Tubing             : {proposal.get('tubing_material', '?')} {proposal.get('tubing_ID_mm', '?')} mm ID",
        f"  Residence time     : {proposal.get('residence_time_min', '?')} min",
        f"  Total flow rate    : {proposal.get('flow_rate_mL_min', '?')} mL/min",
        f"  Reactor volume     : {proposal.get('reactor_volume_mL', '?')} mL",
        f"  Temperature        : {proposal.get('temperature_C', '?')} °C",
        f"  Concentration      : {proposal.get('concentration_M', '?')} M",
        f"  Back pressure (BPR): {proposal.get('BPR_bar', '0')} bar",
        f"  Wavelength         : {proposal.get('wavelength_nm', 'N/A')} nm",
        f"  Light setup        : {proposal.get('light_setup', 'N/A')}",
        f"  Deoxygenation      : {proposal.get('deoxygenation_method', 'N/A')}",
        f"  Confidence         : {conf}",
    ]
    sections.append("## Current Flow Design — Conditions\n" + "\n".join(conditions))

    # ── Streams ────────────────────────────────────────────────────────────────
    streams = proposal.get("streams", [])
    if streams:
        stream_lines = []
        for s in streams:
            stream_lines.append(
                f"  Stream {s.get('stream_label', '?')} — {s.get('pump_role', '')}\n"
                f"    Contents    : {', '.join(s.get('contents', []))}\n"
                f"    Solvent     : {s.get('solvent', '?')}\n"
                f"    Conc.       : {s.get('concentration_M', '?')} M\n"
                f"    Flow rate   : {s.get('flow_rate_mL_min', '?')} mL/min\n"
                f"    Reasoning   : {s.get('reasoning', '')}"
            )
        sections.append("## Current Flow Design — Stream Assignments\n" + "\n".join(stream_lines))

    # ── Reasoning per field ────────────────────────────────────────────────────
    reasoning = proposal.get("reasoning_per_field", {})
    if reasoning:
        r_lines = [f"  {k}: {v}" for k, v in reasoning.items()]
        sections.append("## Current Flow Design — Why Each Parameter Was Chosen\n" + "\n".join(r_lines))

    # ── Pre/post reactor steps ────────────────────────────────────────────────
    pre  = proposal.get("pre_reactor_steps", [])
    post = proposal.get("post_reactor_steps", [])
    if pre or post:
        proc_lines = []
        if pre:
            proc_lines.append("  Pre-reactor : " + " | ".join(pre))
        if post:
            proc_lines.append("  Post-reactor: " + " | ".join(post))
        sections.append("## Current Flow Design — Process Steps\n" + "\n".join(proc_lines))

    # ── Chemistry notes ───────────────────────────────────────────────────────
    chem_notes = proposal.get("chemistry_notes", "")
    if chem_notes:
        sections.append(f"## Chemistry Design Notes\n{chem_notes}")

    # ── Chemistry plan ────────────────────────────────────────────────────────
    if plan:
        plan_lines = [
            f"  Reaction name  : {plan.get('reaction_name', '?')}",
            f"  Mechanism      : {plan.get('mechanism_type', '?')}",
            f"  Bond formed    : {plan.get('bond_formed', '?')}",
            f"  Key intermediate: {plan.get('key_intermediate', '?')}",
            f"  O2-sensitive   : {plan.get('oxygen_sensitive', '?')}",
            f"  Moisture-sens. : {plan.get('moisture_sensitive', '?')}",
            f"  Deoxygenation  : {plan.get('deoxygenation_required', '?')} — {plan.get('deoxygenation_reasoning', '')}",
            f"  Quench required: {plan.get('quench_required', '?')} — {plan.get('quench_reagent', '')}",
        ]
        # Stream logic from chemistry plan
        for sl in plan.get("stream_logic", []):
            plan_lines.append(
                f"  Stream {sl.get('stream_label', '?')} logic: {', '.join(sl.get('reagents', []))} — {sl.get('reasoning', '')}"
            )
        # Incompatible pairs
        for pair in plan.get("incompatible_pairs", []):
            if isinstance(pair, list) and len(pair) >= 2:
                plan_lines.append(f"  Incompatible: {pair[0]} + {pair[1]} must NOT be co-dissolved")
        sections.append("## Chemistry Plan (mechanism & stream logic)\n" + "\n".join(plan_lines))

    # ── Unit operations ───────────────────────────────────────────────────────
    ops = topo.get("unit_operations", [])
    if ops:
        op_labels = [op.get("label", op.get("op_type", "?")) for op in ops]
        sections.append("## Process Flowsheet\n  " + " → ".join(op_labels))

    # ── Literature used ───────────────────────────────────────────────────────
    lit = proposal.get("literature_analogies", [])
    if lit:
        sections.append("## Literature Analogies Used\n  " + ", ".join(lit))

    return "\n\n".join(sections)


class ConversationAgent:
    """Multi-turn conversational wrapper around the FLORA translate pipeline."""

    def __init__(self):
        self._history: list[dict] = []       # clean conversation turns {role, content}
        self.current_result: dict | None = None
        self.original_query: str | None = None
        self.revisions: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, user_input: str) -> ConversationResponse:
        """Process one user message. Returns a ConversationResponse."""

        # Build dynamic system prompt with full current design context
        context_block = _build_full_context(
            self.original_query, self.revisions, self.current_result
        )
        if context_block:
            dynamic_system = _BASE_SYSTEM + "\n\n---\n\n" + context_block
        else:
            dynamic_system = _BASE_SYSTEM

        # Add user turn to history BEFORE classifying
        self._history.append({"role": "user", "content": user_input})

        # Classify intent using full history + dynamic system
        classification = self._classify(dynamic_system, self._history)
        intent            = classification.get("intent", "TRANSLATE")
        msg               = classification.get("message", "")
        rev_inst          = classification.get("revision_instructions")
        questions         = classification.get("questions", [])
        needs_retranslate = classification.get("needs_retranslate", False)

        response = ConversationResponse(action=intent, message=msg, questions=questions)

        if intent in ("ASK", "ANSWER"):
            self._history.append({"role": "assistant", "content": msg})
            return response

        if needs_retranslate:
            result, error = self._run_translate(user_input, intent, rev_inst)
            if error:
                response.error   = error
                response.message = (
                    f"Something went wrong during translation: {error}\n\n"
                    "Please check your input and try again."
                )
            else:
                response.result      = result
                self.current_result  = result
                followup = self._check_missing_info(result)
                if followup:
                    response.message += (
                        "\n\n---\n\n**A few questions to help me refine the design:**\n" +
                        "\n".join(f"{i+1}. {q}" for i, q in enumerate(followup))
                    )
                    response.questions = followup

        self._history.append({"role": "assistant", "content": response.message})
        return response

    def reset(self):
        self._history.clear()
        self.current_result  = None
        self.original_query  = None
        self.revisions.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _classify(self, system: str, messages: list[dict]) -> dict:
        """Call the configured classification model with full context."""
        try:
            result = call_model_messages(
                model=cfg.MODEL_CONVERSATION_AGENT,
                api_name="conversation_agent",
                max_tokens=1024,          # enough for a detailed ANSWER response
                system=system,
                messages=messages[-20:],  # keep last 20 turns to stay within limits
            )
            raw = result.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            has_result = self.current_result is not None
            return {
                "intent":               "REVISE" if has_result else "TRANSLATE",
                "message":              "Let me work on that for you.",
                "revision_instructions": None,
                "questions":            [],
                "needs_retranslate":    True,
            }

    def _run_translate(
        self, user_input: str, intent: str, revision_instructions: str | None
    ) -> tuple[dict | None, str | None]:

        # ── REVISE: use the revision agent (fast, targeted) ───────────────
        if intent == "REVISE" and self.original_query and self.current_result:
            from flora_translate.revision_agent import RevisionAgent

            if revision_instructions:
                self.revisions.append(revision_instructions)

            instructions = revision_instructions or user_input
            try:
                result = RevisionAgent().revise(
                    current_result=self.current_result,
                    revision_instructions=instructions,
                    original_query=self.original_query,
                )
                return result, None
            except Exception as e:
                logger.error(f"RevisionAgent failed: {e}", exc_info=True)
                return None, str(e)

        # ── TRANSLATE / NEW_QUERY: full pipeline ──────────────────────────
        from flora_translate.main import translate

        if intent == "NEW_QUERY":
            self.original_query = user_input
            self.revisions.clear()
        else:
            # First translation
            self.original_query = user_input

        try:
            result = translate(user_input)
            return result, None
        except Exception as e:
            logger.error(f"translate() failed: {e}", exc_info=True)
            return None, str(e)

    def _check_missing_info(self, result: dict) -> list[str]:
        """Return up to 3 targeted clarifying questions after translation."""
        questions = []
        proposal  = result.get("proposal", {})
        plan      = result.get("chemistry_plan", {})
        conf      = result.get("confidence", "HIGH")

        if conf == "LOW":
            questions.append(
                "Confidence is LOW — can you add more detail about the reaction "
                "conditions (solvent, temperature, concentration, reaction time)?"
            )

        if not proposal.get("temperature_C") or proposal.get("temperature_C") == 25:
            if not any(k in (self.original_query or "").lower()
                       for k in ["°c", "room temp", "rt", "25", "temperature"]):
                questions.append(
                    "What temperature does this reaction run at in batch? "
                    "(Affects reactor material and BPR selection.)"
                )

        if plan.get("oxygen_sensitive") and not proposal.get("deoxygenation_method"):
            questions.append(
                "The reaction is O₂-sensitive — what deoxygenation method do you prefer? "
                "(inline sparging, freeze-pump-thaw, or Schlenk technique)"
            )

        if not any(k in (self.original_query or "").lower()
                   for k in ["mmol", "gram", "scale", "mg/h", "g/h", "ml/min"]):
            questions.append(
                "What throughput scale are you targeting? "
                "(mg/h for screening, g/h for optimization, kg/h for production)"
            )

        return questions[:3]


@dataclass
class ConversationResponse:
    action:    str
    message:   str
    result:    dict | None = None
    questions: list[str]   = field(default_factory=list)
    error:     str | None  = None
