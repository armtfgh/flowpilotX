"""FLORA-Fundamentals — Rule schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class FlowRule(BaseModel):
    """A single flow chemistry rule extracted from a handbook."""

    rule_id: str = ""
    category: str = ""
    # Categories: mixing, heat_transfer, photochemistry, pressure,
    # materials, scale_up, safety, residence_time, mass_transfer,
    # reactor_design, solvent, catalyst, general

    condition: str = ""          # WHEN does this rule apply?
    recommendation: str = ""     # WHAT should you do?
    reasoning: str = ""          # WHY? (the chemistry/physics behind it)
    quantitative: str = ""       # Any numbers: thresholds, ranges, formulas
    exceptions: str = ""         # When does this rule NOT apply?

    severity: str = "guideline"  # "hard_rule" | "guideline" | "tip"
    confidence: float = 1.0      # 0.0-1.0

    source_handbook: str = ""    # Which PDF this came from
    source_page: str = ""        # Page number or section
    source_context: str = ""     # Surrounding text for traceability


class HandbookIndex(BaseModel):
    """Metadata about an ingested handbook."""

    handbook_id: str = ""
    filename: str = ""
    title: str = ""
    authors: str = ""
    year: int = 0
    topics: list[str] = Field(default_factory=list)
    n_rules_extracted: int = 0
    n_pages: int = 0


class FundamentalsKnowledge(BaseModel):
    """The complete fundamentals knowledge base."""

    handbooks: list[HandbookIndex] = Field(default_factory=list)
    rules: list[FlowRule] = Field(default_factory=list)
    version: str = "1.0"
