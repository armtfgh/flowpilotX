"""FLORA-Translate — Output formatter: structured JSON + human-readable."""

import json
import logging
import time

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.schemas import DesignCandidate

logger = logging.getLogger("flora.output")

EXPLANATION_SYSTEM = (
    "You are a flow chemistry expert. Convert this validated flow chemistry design "
    "into a clear, concise explanation for a synthetic chemist. Structure it as:\n"
    "(1) Proposed flow setup\n"
    "(2) Why these conditions were chosen (cite analogies by DOI or title)\n"
    "(3) Key differences from the batch protocol\n"
    "(4) Any warnings or flags from the engineering validation\n"
    "Use plain technical language. No marketing language."
)


class OutputFormatter:
    """Format DesignCandidate into structured JSON + human explanation."""

    def format(
        self,
        candidate: DesignCandidate,
        analogies: list[dict],
    ) -> dict:
        """Generate the final output dict with both structured and readable forms."""
        # Compute confidence from two independent signals:
        #   1. Analogy quality — max final_score from retrieval.
        #      Scores now use cosine similarity (corrected from L2 in retriever.py),
        #      so calibrated ranges are:
        #        > 0.75  → strong match (same class, similar conditions)
        #        > 0.50  → decent match (related class)
        #        ≤ 0.50  → weak/no match
        #   2. Council convergence — how many revision rounds were needed.
        #      MIN_COUNCIL_ROUNDS = 2, so 2 rounds = cleanest possible run.
        #        ≤ 2 rounds → no significant issues found
        #        3 rounds   → required corrections (acceptable)
        max_score = max(
            (a.get("final_score", 0) for a in analogies), default=0
        )
        rounds = candidate.council_rounds
        if max_score > 0.75 and rounds <= 2:
            confidence = "HIGH"
        elif max_score > 0.50 and rounds <= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        candidate.proposal.confidence = confidence

        # Generate human-readable explanation
        explanation = self._generate_explanation(candidate)
        candidate.human_explanation = explanation

        # Add confidence-appropriate note
        if confidence == "LOW":
            candidate.human_explanation += (
                "\n\n**Confidence: LOW** — The closest literature analogy has a "
                f"similarity score of {max_score:.2f} (threshold for MEDIUM: 0.50). "
                "No close precedent found in the corpus for this exact reaction class "
                "and conditions. Treat this proposal as a starting hypothesis "
                "requiring careful experimental validation before scale-up."
            )
        elif confidence == "MEDIUM":
            candidate.human_explanation += (
                f"\n\n**Confidence: MEDIUM** — Best analogy score: {max_score:.2f}. "
                "A related literature precedent was found but conditions differ. "
                "Validate key parameters (residence time, concentration) experimentally."
            )

        return {
            "proposal": candidate.proposal.model_dump(),
            "unit_operations": candidate.unit_operations,
            "pid_description": candidate.pid_description,
            "council_rounds": candidate.council_rounds,
            "safety_report": candidate.safety_report,
            "council_messages": [
                m.model_dump() for m in candidate.council_messages
            ],
            "confidence": confidence,
            "explanation": candidate.human_explanation,
        }

    def _generate_explanation(self, candidate: DesignCandidate) -> str:
        """Use Claude to generate a human-readable explanation."""
        p = candidate.proposal
        if p.scientific_design.get("mode") == "scientific_v2":
            # The chief already supplied a reviewed rationale. A second model
            # paraphrase must not turn an experimental screen into validation.
            a = p.scientific_design
            rows = next(c for c in a["candidates"] if c["candidate_id"] == a["selected_candidate_id"])
            return ("Experimental screening hypothesis; achieved yield and flow kinetics are unknown.\n\n"
                    + a["chief"]["justification"] + "\n\n"
                    + "\n".join(f"Stage {s['stage_number']}: {s['reactor_volume_mL']:g} mL, {s['temperature_C']:g} C, {s['residence_time_min']:g} min."
                                 for s in p.stage_parameters)
                    + "\n\n" + "\n".join(d["consequence"] for d in rows["temperature_deviations"])
                    + "\n\nRequired measurements: " + "; ".join(a["chief"].get("next_measurements", [])))
        # Build a concise "key numbers" block so the LLM cannot hallucinate
        # residence time, flow rate, or volume from inconsistent context.
        key_numbers = (
            f"\n\n## AUTHORITATIVE DESIGN NUMBERS — cite these exactly, do not approximate\n"
            f"- Residence time: {p.residence_time_min} min\n"
            f"- Flow rate: {p.flow_rate_mL_min} mL/min\n"
            f"- Reactor volume: {p.reactor_volume_mL} mL\n"
            f"- Temperature: {p.temperature_C} °C\n"
            f"- Concentration: {p.concentration_M} M\n"
            f"- Tubing: {p.tubing_material} {p.tubing_ID_mm} mm ID\n"
            + (f"- Wavelength: {p.wavelength_nm} nm\n" if p.wavelength_nm else "")
            + (f"- BPR: {p.BPR_bar} bar\n" if p.BPR_bar else "")
        )
        if p.multiphase_metrics:
            key_numbers += (
                "- Multiphase: "
                f"gas={p.multiphase_metrics.get('gas_species')}, "
                f"MFC={p.multiphase_metrics.get('gas_flow_sccm')} sccm, "
                f"gas holdup={p.multiphase_metrics.get('gas_holdup')}, "
                f"O2 transfer sufficiency={p.multiphase_metrics.get('o2_transfer_sufficiency')}x\n"
            )
        if p.heat_transfer_metrics:
            key_numbers += (
                "- Heat transfer: "
                f"UA={p.heat_transfer_metrics.get('UA_W_K')} W/K, "
                f"Da_th={p.heat_transfer_metrics.get('thermal_damkohler')}, "
                f"score={p.heat_transfer_metrics.get('heat_transfer_score')}\n"
            )
        system_with_numbers = EXPLANATION_SYSTEM + key_numbers
        try:
            started = time.perf_counter()
            result = call_model_text(
                model=cfg.MODEL_OUTPUT_FORMATTER,
                api_name="output_formatter",
                max_tokens=3000,
                system=system_with_numbers,
                user_content=json.dumps(
                    candidate.model_dump(), indent=2, default=str
                ),
            )
            logger.debug("Output formatter LLM call completed in %.2f ms", (time.perf_counter() - started) * 1000)
            return result.text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            # Fallback: basic text summary
            p = candidate.proposal
            return (
                f"Flow Proposal: {p.reactor_type} reactor, "
                f"{p.tubing_material} {p.tubing_ID_mm}mm ID, "
                f"{p.reactor_volume_mL}mL volume, "
                f"{p.residence_time_min} min residence time, "
                f"{p.flow_rate_mL_min} mL/min flow rate, "
                f"{p.temperature_C}°C, {p.concentration_M}M, "
                f"BPR {p.BPR_bar} bar. "
                f"Confidence: {p.confidence}."
            )
