"""FLORA-Fundamentals — Knowledge store: loads, queries, and injects rules.

The knowledge store is the bridge between extracted handbook rules and
the Chemistry Agent / ENGINE council. It provides:
  - Category-based lookup
  - Keyword search
  - Context injection (formats rules for LLM prompts)
"""

import json
import logging
from pathlib import Path

from flora_fundamentals.schemas import FlowRule, FundamentalsKnowledge

logger = logging.getLogger("flora.fundamentals.store")

DEFAULT_PATH = Path("flora_fundamentals/data/rules.json")


class KnowledgeStore:
    """Query and inject flow chemistry fundamentals rules."""

    def __init__(self, rules_path: str | Path = DEFAULT_PATH):
        self.rules_path = Path(rules_path)
        self._knowledge: FundamentalsKnowledge | None = None

    @property
    def rules(self) -> list[FlowRule]:
        self._ensure_loaded()
        return self._knowledge.rules

    @property
    def n_rules(self) -> int:
        self._ensure_loaded()
        return len(self._knowledge.rules)

    def _ensure_loaded(self):
        if self._knowledge is not None:
            return
        if self.rules_path.exists():
            data = json.loads(self.rules_path.read_text())
            self._knowledge = FundamentalsKnowledge(**data)
            logger.info(f"Loaded {len(self._knowledge.rules)} fundamentals rules")
        else:
            self._knowledge = FundamentalsKnowledge()
            logger.warning(f"No rules file at {self.rules_path}")

    def query_by_category(self, *categories: str) -> list[FlowRule]:
        """Get all rules matching one or more categories."""
        self._ensure_loaded()
        return [
            r for r in self._knowledge.rules
            if r.category in categories
        ]

    def query_by_keywords(self, keywords: list[str], max_results: int = 20) -> list[FlowRule]:
        """Search rules by keyword match in condition/recommendation/reasoning."""
        self._ensure_loaded()
        results = []
        for rule in self._knowledge.rules:
            text = f"{rule.condition} {rule.recommendation} {rule.reasoning} {rule.quantitative}".lower()
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > 0:
                results.append((score, rule))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:max_results]]

    def query_for_reaction(
        self,
        mechanism_type: str = "",
        photocatalyst: str = "",
        solvent: str = "",
        temperature_C: float | None = None,
        phase_regime: str = "",
        oxygen_sensitive: bool = False,
    ) -> list[FlowRule]:
        """Get rules relevant to a specific reaction context.

        This is the main entry point used by the Chemistry Agent
        and ENGINE council.
        """
        self._ensure_loaded()
        relevant = []

        keywords = []
        if mechanism_type:
            keywords.extend(mechanism_type.lower().split())
        if photocatalyst:
            keywords.append(photocatalyst.lower())
        if solvent:
            keywords.append(solvent.lower())
        if phase_regime:
            keywords.extend(phase_regime.lower().replace("_", " ").split())

        # Category-based selection
        categories = {"general", "reactor_design"}
        if oxygen_sensitive:
            categories.add("safety")
        if photocatalyst or "photo" in mechanism_type.lower():
            categories.add("photochemistry")
        if solvent:
            categories.add("solvent")
            categories.add("materials")
        if temperature_C and temperature_C > 50:
            categories.add("heat_transfer")
            categories.add("pressure")
        if "gas" in phase_regime.lower():
            categories.add("mass_transfer")
            categories.add("mixing")
        if "radical" in mechanism_type.lower() or "chain" in mechanism_type.lower():
            categories.add("residence_time")
            categories.add("mixing")

        # Get all matching rules
        cat_rules = self.query_by_category(*categories)
        kw_rules = self.query_by_keywords(keywords) if keywords else []

        # Deduplicate
        seen = set()
        for r in cat_rules + kw_rules:
            if r.rule_id not in seen:
                seen.add(r.rule_id)
                relevant.append(r)

        logger.info(f"  Fundamentals: {len(relevant)} rules relevant to this reaction")
        return relevant

    def format_for_prompt(self, rules: list[FlowRule], max_rules: int = 15) -> str:
        """Format rules as text for injection into an LLM prompt.

        Returns a concise text block that can be appended to the
        Chemistry Agent or Translation LLM system prompt.
        """
        if not rules:
            return ""

        # Prioritise hard_rules, then guidelines, then tips
        severity_order = {"hard_rule": 0, "guideline": 1, "tip": 2}
        sorted_rules = sorted(
            rules,
            key=lambda r: severity_order.get(r.severity, 2),
        )[:max_rules]

        lines = ["FLOW CHEMISTRY FUNDAMENTALS (from handbooks):"]
        for r in sorted_rules:
            severity_tag = "[MUST]" if r.severity == "hard_rule" else "[SHOULD]" if r.severity == "guideline" else "[TIP]"
            line = f"  {severity_tag} {r.recommendation}"
            if r.condition:
                line += f" — WHEN: {r.condition}"
            if r.quantitative:
                line += f" ({r.quantitative})"
            lines.append(line)

        return "\n".join(lines)
