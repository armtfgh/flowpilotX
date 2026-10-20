"""FLORA-Translate — Chemistry Reasoning Agent (Layer 1).

Runs BEFORE retrieval and hardware translation.
Analyzes the batch protocol purely from a chemistry perspective:
  - Identifies all species and their roles
  - Proposes a mechanism
  - Determines stream separation logic
  - Flags sensitivities (O2, moisture, light)
  - Generates retrieval hints for plan-aware search

Outputs a ChemistryPlan — no hardware decisions.
"""

import json
import logging
import re
import time

import flora_translate.config as cfg
from flora_translate.config import PROMPTS_DIR
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.intensification import ensure_intensification_mandate
from flora_translate.intake_agent import intake_context_block
from flora_translate.schemas import BatchRecord, ChemistryPlan

logger = logging.getLogger("flora.chemistry_agent")


def _parse_json_from_tagged(text: str) -> dict:
    """Extract JSON from between <JSON> ... </JSON> tags, or fall back to raw JSON."""
    if "<JSON>" in text:
        start = text.index("<JSON>") + len("<JSON>")
        end   = text.index("</JSON>") if "</JSON>" in text else len(text)
        text  = text[start:end].strip()
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def _strip_fundamentals_block(system: str) -> str:
    marker = "\n\n## FLOW CHEMISTRY HANDBOOK RULES\n"
    if marker not in system:
        return system
    return system.split(marker, 1)[0] + (
        "\n\n## FLOW CHEMISTRY HANDBOOK RULES\n"
        "Fundamentals rules were omitted in this fallback call because the "
        "larger prompt failed at the provider connection layer."
    )


CHEMISTRY_SYSTEM = """\
You are an expert organic chemist and flow chemistry engineer with deep expertise
across photocatalysis, thermal catalysis, electrochemistry, biocatalysis, and
organocatalysis. Your task: analyze a batch chemistry protocol and produce a
rigorous CHEMISTRY PLAN grounded in first principles.

## Output structure

First, write a BRIEF NOTES block (5-8 bullet points maximum, concise):
<NOTES>
• Chemistry type: [what type, what drives it]
• Rate-limiting step in batch: [what, why]
• Flow advantage: [specific quantitative reason, e.g. "100× better photon delivery in 1mm ID tube"]
• Key incompatible pairs: [species A + species B → why incompatible]
• Critical sensitivity: [O2/moisture/light/temperature + reason]
• Retrieval strategy: [2-3 most distinctive keywords for search]
</NOTES>

Then output the complete JSON (no markdown fences):
<JSON>
{ ... ChemistryPlan fields ... }
</JSON>

Rules:
- The NOTES must be bullet points only — no paragraphs, no lengthy explanations.
- The JSON must be complete, valid, and consistent with NOTES.
- Name every species explicitly. Never use "reagent 1" or "catalyst".
- For photochem: recommended_wavelength_nm must match photocatalyst λ_abs.
- For non-photochem: set recommended_wavelength_nm to null.
- confidence_notes: note anything you are uncertain about.
"""

CHEMISTRY_USER_TEMPLATE = """\
Analyze this batch protocol and produce a ChemistryPlan.

BATCH PROTOCOL:
{batch_json}

CRITICAL — MULTI-STEP REACTIONS:
If the protocol describes more than one synthetic step (even if it says
"one-pot" or "telescoped"), you MUST populate the "stages" array.
Each stage is a SEPARATE reactor zone in the flow process, with its own
feeds, temperature, reactor type, and inter-stage actions.

For EACH stage, specify:
- What feeds IN to this stage (new pump streams)
- What comes FROM the previous stage (the outlet stream)
- The batch reaction time for THIS stage as batch_time_h, if stated
- The reactor type for this stage (coil, packed_bed, chip, CSTR)
- Temperature, solvent, atmosphere for THIS stage
- What happens BETWEEN this stage and the next (quench, solvent switch,
  inline filter, heat exchange)

If it is a single-step reaction, set n_stages=1 and leave stages=[].

STRICT RULES FOR post_stage_action:
- Leave post_stage_action="" unless there is a SPECIFIC, CHEMISTRY-DRIVEN reason.
- "inline filter": ONLY if solid by-products, precipitates, or heterogeneous
  catalyst fines must be removed (e.g. Mg turnings, Pd/C, resin).
- "quench with X": ONLY if the reaction must be quenched inline (e.g. organolithium,
  strong acid/base work-up, reactive intermediate).
- "solvent switch to X": ONLY if the next stage requires a different solvent that
  is incompatible with the current one.
- "liquid-liquid extraction" / "separator": ONLY if TWO IMMISCIBLE LIQUID PHASES
  genuinely exist at the inter-stage point AND the batch protocol explicitly mentions
  an aqueous work-up, extraction, or phase separation between steps.
  DO NOT use this for: atmosphere changes, gas switching, deoxygenation, or just
  because the chemistry is sensitive. Gas-liquid phase changes use gas injection
  (pump_role="gas injection"), not a separator.
- post_stage_reasoning MUST be non-empty whenever post_stage_action is non-empty.
  Cite the specific chemical reason (species, phase, constraint).

Return JSON:
{{
  "reaction_name": "",
  "reaction_class": "",
  "mechanism_type": "",
  "bond_formed": "",
  "bond_broken": "",

  "n_stages": 1,
  "stages": [
    {{
      "stage_number": 1,
      "stage_name": "e.g. Grignard formation",
      "reaction_type": "e.g. organometallic",
      "reactor_type": "coil | packed_bed | chip | CSTR",
      "temperature_C": null,
      "requires_light": false,
      "wavelength_nm": null,
      "batch_time_h": null,
      "feed_streams": [
        {{"stream_label": "A", "reagents": ["ArBr (1.0 equiv)"], "reasoning": "substrate feed"}}
      ],
      "inlet_from_previous": "",
      "solvent": "",
      "atmosphere": "N2",
      "oxygen_sensitive": false,
      "moisture_sensitive": false,
      "deoxygenation_required": false,
      "post_stage_action": "e.g. inline filter to remove Mg fines",
      "post_stage_reasoning": "e.g. Mg particles must not enter next reactor"
    }}
  ],

  "reagents": [
    {{"name": "", "role": "", "equiv_or_loading": "", "smiles": null, "notes": ""}}
  ],
  "mechanism_steps": [
    {{"step_number": 1, "description": "", "species_involved": [], \
"is_photon_dependent": false, "is_rate_limiting": false}}
  ],
  "key_intermediate": "",
  "excited_state_type": "",
  "energy_transfer_or_redox": "",
  "oxygen_sensitive": false,
  "moisture_sensitive": false,
  "temperature_sensitive": false,
  "light_sensitive_reagents": [],
  "stream_logic": [
    {{"stream_label": "A", "reagents": [], "reasoning": "",
      "molar_equiv": 1.0, "phase": "liquid", "concentration_M": null}}
  ],
  "_stream_logic_instructions": "For EACH reagent entry, preserve its own stated equivalents or loading in reagents[].equiv_or_loading; never use one stream-level number as a substitute for multiple reactive components. For EACH stream entry, fill molar_equiv only when it accurately represents the stream's principal reactive component relative to the limiting reagent (substrate = 1.0). Include each reagent's quantity in the stream reagent text, for example 'TBHP (2.0 equiv)' or 'photocatalyst (1 mol%)'. If a reactive component is not quantified in the protocol, explicitly write '(quantity unresolved)' instead of inventing a number. Fill phase with one of: 'liquid', 'gas', 'solid'. For gas reagents (O2, H2, CO2, etc) ALWAYS extract molar_equiv from the protocol text. These fields are consumed by deterministic stoichiometry and release gates.",
  "mixing_order_reasoning": "",
  "incompatible_pairs": [],
  "deoxygenation_required": false,
  "deoxygenation_reasoning": "",
  "quench_required": false,
  "quench_reagent": "",
  "quench_reasoning": "",
  "retrieval_keywords": [],
  "similar_reaction_classes": [],
  "recommended_wavelength_nm": null,
  "wavelength_reasoning": "",
  "intensification_mandate": {{
    "tau_reduction_target": 2.0,
    "minimum_flow_advantage": "productivity",
    "required_mixing_regime": "laminar_acceptable",
    "flow_justification_basis": "Flow should provide a concrete productivity, safety, heat-transfer, selectivity, or hazardous-intermediate advantage over batch."
  }},
  "confidence_notes": ""
}}

Think carefully about the mechanism. Name every species explicitly and preserve
the protocol-stated or chemist-confirmed transformation identity. Model inference
is a hypothesis, not authority; state unresolved identity or quantities clearly.
For multi-step: the STAGES array is the most important part — get the
inter-stage connections right (what flows from where into what).
"""


def _normalize_plan_data(data: dict) -> dict:
    """Fix common LLM output deviations before Pydantic validation.

    The LLM sometimes returns fields in slightly different shapes
    than the schema expects. This normalizes them.
    """
    def unwrap(value, default=None):
        if not isinstance(value, dict):
            return value
        for key in (
            "value", "selected", "recommended", "estimate", "nominal",
            "result", "text", "name",
        ):
            if key in value and not isinstance(value[key], (dict, list)):
                return value[key]
        return default

    def coerce_bool(value) -> bool:
        value = unwrap(value, value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "required"}
        return bool(value)

    # incompatible_pairs: expected [[A, B], ...] but LLM may return
    # [{"species_1": A, "species_2": B, ...}, ...]
    pairs = data.get("incompatible_pairs", [])
    if isinstance(pairs, list):
        normalized = []
        for p in pairs:
            if isinstance(p, dict):
                # Extract the two species from whatever keys the LLM used
                vals = [v for k, v in p.items() if isinstance(v, str) and k != "reason" and k != "reasoning"]
                if len(vals) >= 2:
                    normalized.append(vals[:2])
                elif vals:
                    normalized.append(vals)
            elif isinstance(p, str) and p.strip():
                species = [
                    item.strip()
                    for item in re.split(r"\s*(?:\+|/|\band\b|\bwith\b)\s*", p)
                    if item.strip()
                ]
                normalized.append(species[:2] if species else [p.strip()])
            elif isinstance(p, list):
                normalized.append([str(item) for item in p[:2]])
            else:
                continue
        data["incompatible_pairs"] = normalized

    def normalize_stream(stream: dict) -> None:
        # reagents should be a list of strings
        reagents = stream.get("reagents", [])
        if reagents and isinstance(reagents, list) and isinstance(reagents[0], dict):
            stream["reagents"] = [
                item.get("name", str(item)) for item in reagents
            ]
        elif isinstance(reagents, str):
            stream["reagents"] = [reagents]

        for field, fallback in (
            ("molar_equiv", 1.0),
            ("concentration_M", None),
            ("gas_flow_sccm", None),
            ("gas_flow_actual_mL_min", None),
        ):
            value = unwrap(stream.get(field))
            if value is None:
                if fallback is not None:
                    stream[field] = fallback
                continue
            try:
                stream[field] = float(value)
            except (TypeError, ValueError):
                stream[field] = fallback

    # stream_logic: expected list of dicts with stream_label/reagents/reasoning
    # but LLM may nest differently or put phase labels in numeric fields.
    streams = data.get("stream_logic", [])
    if isinstance(streams, list):
        for s in streams:
            if isinstance(s, dict):
                normalize_stream(s)

    # mechanism_steps: species_involved should be list[str]
    steps = data.get("mechanism_steps", [])
    for step in steps:
        if isinstance(step, dict):
            sp = step.get("species_involved", [])
            if sp and isinstance(sp[0], dict):
                step["species_involved"] = [
                    item.get("name", str(item)) for item in sp
                ]

    # stages: feed_streams may come as dicts with wrong shape
    stages = data.get("stages", [])
    for stage in stages:
        if isinstance(stage, dict):
            for field in ("temperature_C", "wavelength_nm", "batch_time_h", "stage_number"):
                if field in stage:
                    stage[field] = unwrap(stage[field])
            for field in (
                "requires_light", "oxygen_sensitive", "moisture_sensitive",
                "deoxygenation_required",
            ):
                if field in stage:
                    stage[field] = coerce_bool(stage[field])
            feeds = stage.get("feed_streams", [])
            for f in feeds:
                if isinstance(f, dict):
                    normalize_stream(f)

    # ── Coerce None → "" for str fields with empty-string defaults ───────────
    # Pydantic v2 rejects None for plain `str` fields even with a "" default.
    _str_fields = [
        "excited_state_type", "energy_transfer_or_redox",
        "key_intermediate", "reaction_name", "reaction_class",
        "mechanism_type", "bond_formed", "phase_regime",
        "mixing_order_reasoning", "photocatalyst_loading", "scale_note",
    ]
    for field in _str_fields:
        if isinstance(data.get(field), dict):
            value = unwrap(data[field])
            data[field] = value if value is not None else json.dumps(
                data[field], sort_keys=True, ensure_ascii=True
            )
        if data.get(field) is None:
            data[field] = ""

    for field in (
        "oxygen_sensitive", "o2_is_reagent", "moisture_sensitive",
        "temperature_sensitive", "deoxygenation_required", "quench_required",
    ):
        if field in data:
            data[field] = coerce_bool(data[field])

    # ── Coerce "" / non-numeric strings → None for Optional[float/int] fields ─
    # The LLM sometimes returns "" or "N/A" for numeric fields that have no
    # value (e.g. recommended_wavelength_nm for a thermal reaction).
    _numeric_fields = [
        "recommended_wavelength_nm", "n_stages",
    ]
    for field in _numeric_fields:
        if field not in data:
            continue
        val = unwrap(data.get(field))
        data[field] = val
        if val is None:
            continue
        if isinstance(val, str):
            stripped = val.strip()
            try:
                data[field] = float(stripped) if stripped else None
            except ValueError:
                data[field] = None

    mandate = data.get("intensification_mandate")
    if isinstance(mandate, dict):
        if mandate.get("tau_reduction_target") is None:
            mandate.pop("tau_reduction_target", None)
        for field in (
            "minimum_flow_advantage",
            "required_mixing_regime",
            "flow_justification_basis",
        ):
            if mandate.get(field) is None:
                mandate.pop(field, None)

    return data


class ChemistryReasoningAgent:
    """Layer 1: Pure chemistry analysis before any hardware decisions.

    If FLORA-Fundamentals rules are available, they are injected into
    the system prompt so the LLM has access to handbook-level domain
    knowledge alongside its own training data.
    """

    def analyze(self, batch_record: BatchRecord, intake_package=None, scientific=False) -> ChemistryPlan:
        """Analyze a batch protocol and return a ChemistryPlan."""
        self._last_model_used = cfg.MODEL_CHEMISTRY_AGENT
        self._fallback_note = ""
        logger.info(f"  Chemistry Agent: Analyzing with {cfg.MODEL_CHEMISTRY_AGENT}")

        # Load fundamentals rules if available
        fundamentals_block = self._load_fundamentals(batch_record)
        system = CHEMISTRY_SYSTEM
        if fundamentals_block:
            system = system + "\n\n## FLOW CHEMISTRY HANDBOOK RULES\n" + fundamentals_block
            logger.info("    Injected fundamentals knowledge into prompt")
        if scientific:
            from flora_translate.scientific_evidence import UPSTREAM_POLICY
            system += "\n\n" + UPSTREAM_POLICY

        batch_json = json.dumps(
            batch_record.model_dump(exclude_none=True), indent=2
        )
        user_prompt = CHEMISTRY_USER_TEMPLATE.format(batch_json=batch_json)
        if intake_package is not None:
            user_prompt += (
                "\n\n"
                + intake_context_block(intake_package)
                + "\n\nUse measured evidence and hard constraints as context. "
                "Treat chemist hypotheses as hypotheses to test, not as facts."
            )

        raw_text = self._call_with_retry(system, user_prompt, scientific=scientific)

        # Extract reasoning block for logging
        reasoning = self._extract_reasoning(raw_text)
        if reasoning:
            logger.info(f"    Reasoning summary: {reasoning[:300]}...")

        data = _parse_json_from_tagged(raw_text)
        if scientific:
            from flora_translate.scientific_evidence import explicit_gas_ratios
            data = explicit_gas_ratios(data)
        data = _normalize_plan_data(data)
        plan = ChemistryPlan(**data)
        # IMPORTANT: attach _reasoning BEFORE the mandate is built, so
        # intensification.detect_batch_limitations can read the chemistry
        # agent's rate-limiting-step analysis. Reversing this order causes
        # the mandate to fall back to class defaults when the limitation
        # keywords live in the reasoning text rather than the JSON fields.
        plan._reasoning = reasoning
        setattr(plan, "_chemistry_model_used", getattr(self, "_last_model_used", cfg.MODEL_CHEMISTRY_AGENT))
        if getattr(self, "_fallback_note", ""):
            plan.confidence_notes = (
                (plan.confidence_notes or "").rstrip()
                + "\n"
                + self._fallback_note
            ).strip()
            setattr(plan, "_chemistry_fallback_note", self._fallback_note)
        plan = ensure_intensification_mandate(batch_record, plan)

        logger.info(f"    Reaction: {plan.reaction_name} ({plan.mechanism_type})")
        logger.info(f"    Key intermediate: {plan.key_intermediate}")
        logger.info(f"    O2-sensitive: {plan.oxygen_sensitive}")
        logger.info(f"    Streams: {len(plan.stream_logic)}")
        for sl in plan.stream_logic:
            logger.info(f"      Stream {sl.stream_label}: {sl.reagents}")
        logger.info(f"    Retrieval hints: {plan.retrieval_keywords[:5]}")

        return plan

    def _call_with_retry(self, system: str, user_prompt: str, *, scientific=False) -> str:
        """Call the model, retrying with compact context when output is incomplete."""
        candidates: list[tuple[str, str, str]] = [
            (cfg.MODEL_CHEMISTRY_AGENT, "chemistry_agent", system),
        ]
        fallback_model = getattr(cfg, "MODEL_TRANSLATION", None) or "claude-sonnet-4-6"
        if fallback_model != cfg.MODEL_CHEMISTRY_AGENT:
            candidates.append((fallback_model, "chemistry_agent_fallback", system))
        compact_system = _strip_fundamentals_block(system)
        if compact_system != system:
            candidates.append((fallback_model, "chemistry_agent_compact_fallback", compact_system))

        errors: list[str] = []
        for idx, (model, api_name, system_prompt) in enumerate(candidates):
            started = time.perf_counter()
            try:
                max_tokens = cfg.CHEMISTRY_MAX_TOKENS
                if api_name == "chemistry_agent_compact_fallback":
                    max_tokens = max(max_tokens * 2, 16384)
                result = call_model_text(
                    model=model,
                    api_name=api_name,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    user_content=user_prompt,
                )
                logger.debug(
                    "Chemistry agent LLM call completed in %.2f ms with %s",
                    (time.perf_counter() - started) * 1000,
                    model,
                )
                if result.stop_reason == "max_tokens" or result.finish_reason == "length":
                    message = "output hit token limit"
                    errors.append(f"{api_name}({model}): {message}")
                    logger.warning("    Chemistry Agent %s; retrying with fallback context", message)
                    continue
                try:
                    parsed = _parse_json_from_tagged(result.text)
                    if scientific:
                        from flora_translate.scientific_evidence import explicit_gas_ratios
                        explicit_gas_ratios(parsed)
                except Exception as exc:
                    errors.append(f"{api_name}({model}): invalid JSON: {exc}")
                    logger.warning(
                        "    Chemistry Agent returned invalid JSON via %s (%s); retrying: %s",
                        api_name,
                        model,
                        exc,
                    )
                    user_prompt += f"\nPrevious output failed validation: {exc}. Correct the structured output without inventing measured evidence."
                    continue
                if idx > 0:
                    self._fallback_note = (
                        f"Chemistry analysis used fallback model {model} after "
                        f"{cfg.MODEL_CHEMISTRY_AGENT} failed during GUI/API call."
                    )
                    logger.warning("    %s", self._fallback_note)
                self._last_model_used = model
                return result.text
            except Exception as exc:
                errors.append(f"{api_name}({model}): {exc}")
                logger.warning("    Chemistry Agent call failed via %s (%s): %s", api_name, model, exc)

        raise RuntimeError("Chemistry Agent failed after fallback attempts: " + " | ".join(errors))

    def _extract_reasoning(self, text: str) -> str:
        """Extract the NOTES section (between <NOTES> tags)."""
        if "<NOTES>" in text and "</NOTES>" in text:
            start = text.index("<NOTES>") + len("<NOTES>")
            end   = text.index("</NOTES>")
            return text[start:end].strip()
        if "<JSON>" in text:
            return text[:text.index("<JSON>")].strip()
        return ""

    def _load_fundamentals(self, batch_record: BatchRecord) -> str:
        """Load relevant fundamentals rules for injection into the prompt."""
        try:
            from flora_fundamentals.knowledge_store import KnowledgeStore

            store = KnowledgeStore()
            if store.n_rules == 0:
                return ""

            rules = store.query_for_reaction(
                photocatalyst=batch_record.photocatalyst or "",
                solvent=batch_record.solvent or "",
                temperature_C=batch_record.temperature_C,
                oxygen_sensitive=batch_record.atmosphere in ("N2", "Ar") if batch_record.atmosphere else False,
            )
            return store.format_for_prompt(rules)
        except Exception as e:
            logger.debug(f"    Fundamentals not available: {e}")
            return ""
