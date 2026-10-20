"""Explicit, serializable council execution controls for architecture studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCORING_AGENTS = ("chemistry", "kinetics", "fluidics", "safety")


@dataclass(frozen=True)
class CouncilExecutionConfig:
    """Select council modules without changing the production defaults."""

    condition_id: str = "full"
    enabled_scoring_agents: tuple[str, ...] = SCORING_AGENTS
    enable_skeptic: bool = True
    enable_preselection_refinement: bool = True
    enable_winner_revision: bool = True
    enable_chief_llm: bool = True
    enable_dfmea: bool = True

    def __post_init__(self) -> None:
        agents = tuple(dict.fromkeys(str(item).strip().lower() for item in self.enabled_scoring_agents))
        unknown = sorted(set(agents) - set(SCORING_AGENTS))
        if unknown:
            raise ValueError(f"Unknown council scoring agents: {unknown}")
        if not agents:
            raise ValueError("At least one scoring agent must remain enabled")
        object.__setattr__(self, "enabled_scoring_agents", agents)
        if not str(self.condition_id).strip():
            raise ValueError("condition_id must not be empty")

    @classmethod
    def coerce(cls, value: "CouncilExecutionConfig | dict[str, Any] | None") -> "CouncilExecutionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        data = dict(value)
        data["enabled_scoring_agents"] = tuple(
            data.get("enabled_scoring_agents") or SCORING_AGENTS
        )
        return cls(**data)

    def provenance(self) -> dict[str, Any]:
        return asdict(self)

