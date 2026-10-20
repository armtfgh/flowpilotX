"""FLORA-Design — Chemistry Classifier: free text → ChemFeatures."""

import json
import logging
import re
from pathlib import Path

import anthropic

from flora_design.rules.photocatalyst_db import lookup_photocatalyst
from flora_design.rules.unit_op_rules import REACTION_KEYWORDS
from flora_translate.config import TRANSLATION_MODEL
from flora_translate.schemas import ChemFeatures

logger = logging.getLogger("flora.design.classifier")

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _get_client():
    return anthropic.Anthropic()


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


class ChemistryClassifier:
    """Convert free-text chemistry goal into structured ChemFeatures."""

    def classify(self, goal_text: str) -> ChemFeatures:
        logger.info("  Classifying chemistry goal")

        system = (PROMPTS_DIR / "classify_system.txt").read_text()
        user_template = (PROMPTS_DIR / "classify_user.txt").read_text()
        user = user_template.format(goal_text=goal_text)

        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        data = _parse_json(resp.content[0].text)
        features = ChemFeatures(**data)

        # Post-processing: infer reaction_class from keywords if unknown
        if features.reaction_class == "unknown" and features.photocatalyst:
            features.reaction_class = self._infer_class(goal_text)

        # Post-processing: fill wavelength from photocatalyst DB
        if features.wavelength_nm is None and features.photocatalyst:
            info = lookup_photocatalyst(features.photocatalyst)
            if info:
                features.wavelength_nm = info["wavelength_nm"]
                if not features.photocatalyst_class:
                    features.photocatalyst_class = info["class"]

        logger.info(f"    Class: {features.reaction_class}")
        logger.info(f"    Catalyst: {features.photocatalyst} ({features.wavelength_nm}nm)")
        logger.info(f"    Confidence: {features.classifier_confidence}")

        return features

    def _infer_class(self, text: str) -> str:
        """Keyword-based fallback for reaction classification."""
        text_lower = text.lower()
        for keyword, cls in REACTION_KEYWORDS.items():
            if keyword.lower() in text_lower:
                return cls
        return "unknown"
