"""Runtime controls for the shared production FlowPilot pipeline.

These controls change execution mechanics, not chemistry.  The GUI uses the
defaults; benchmark callers can freeze retrieval exclusions, council budget,
and recorder settings without maintaining a second implementation of the
pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import threading
from typing import Any, Iterable

from flora_translate.engine.council_v4.execution_config import CouncilExecutionConfig


@dataclass(frozen=True)
class PipelineRuntimeOptions:
    design_policy: str = "legacy"
    exclude_record_ids: frozenset[str] = field(default_factory=frozenset)
    retrieval_mode: str = "semantic"
    candidate_budget: int = 12
    council_backflow_review: bool = True
    council_physics_profile: dict[str, Any] | None = None
    objective_override: str | None = None
    hard_constraints: tuple[str, ...] = ()
    benchmark_recorder: Any = None
    benchmark_strict_scoring: bool = False
    benchmark_scoring_batch_size: int | None = None
    benchmark_strong_revision_mode: bool = False
    benchmark_branching_revision_mode: bool = False
    benchmark_max_descendants_per_candidate: int = 2
    benchmark_max_total_revised_candidates: int | None = None
    council_execution: CouncilExecutionConfig | dict[str, Any] | None = None
    upstream_model: str | None = None
    upstream_provider: str | None = None
    downstream_model: str | None = None
    downstream_provider: str | None = None
    model_endpoints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.council_backflow_review) is not bool:
            raise ValueError("council_backflow_review must be boolean")
        if self.design_policy not in {"legacy", "scientific_v2"}:
            raise ValueError("design_policy must be legacy or scientific_v2")
        if self.council_physics_profile is not None:
            if self.design_policy != "scientific_v2" or not self.council_backflow_review:
                raise ValueError("Physics tools require scientific_v2 and the topology backflow review")
            from flora_translate.engine.council_v4.transient_flow import TransientProfile
            object.__setattr__(self, "council_physics_profile",
                               TransientProfile.model_validate(self.council_physics_profile).model_dump())
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be at least 1")
        if self.design_policy == "scientific_v2" and self.candidate_budget != 12:
            raise ValueError("Scientific preview requires the full 12-candidate council")
        if self.retrieval_mode not in {"semantic", "lexical"}:
            raise ValueError("retrieval_mode must be 'semantic' or 'lexical'")
        if self.benchmark_max_descendants_per_candidate < 1:
            raise ValueError(
                "benchmark_max_descendants_per_candidate must be at least 1"
            )
        if self.council_execution is not None:
            object.__setattr__(
                self,
                "council_execution",
                CouncilExecutionConfig.coerce(self.council_execution),
            )
        for provider in (self.upstream_provider, self.downstream_provider):
            if provider is not None and provider not in {"anthropic", "openai", "ollama"}:
                raise ValueError(f"Unsupported model provider: {provider}")

    @classmethod
    def coerce(
        cls,
        value: "PipelineRuntimeOptions | dict[str, Any] | None",
    ) -> "PipelineRuntimeOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        data = dict(value)
        data["exclude_record_ids"] = frozenset(
            str(item) for item in data.get("exclude_record_ids") or () if item
        )
        data["hard_constraints"] = tuple(
            str(item) for item in data.get("hard_constraints") or () if item
        )
        data["model_endpoints"] = {
            str(model): str(base_url)
            for model, base_url in (data.get("model_endpoints") or {}).items()
            if model and base_url
        }
        return cls(**data)

    def provenance(self) -> dict[str, Any]:
        """Serializable execution metadata stored with every result."""

        return {
            "design_policy": self.design_policy,
            "pipeline_entry_point": "flora_translate.main.translate",
            "exclude_record_ids": sorted(self.exclude_record_ids),
            "retrieval_mode": self.retrieval_mode,
            "candidate_budget": self.candidate_budget,
            "council_backflow_review": self.council_backflow_review,
            "council_physics_profile": self.council_physics_profile,
            "objective_override": self.objective_override,
            "hard_constraints": list(self.hard_constraints),
            "benchmark_strict_scoring": self.benchmark_strict_scoring,
            "benchmark_scoring_batch_size": self.benchmark_scoring_batch_size,
            "benchmark_strong_revision_mode": self.benchmark_strong_revision_mode,
            "benchmark_branching_revision_mode": (
                self.benchmark_branching_revision_mode
            ),
            "benchmark_max_descendants_per_candidate": (
                self.benchmark_max_descendants_per_candidate
            ),
            "benchmark_max_total_revised_candidates": (
                self.benchmark_max_total_revised_candidates
            ),
            "council_execution": (
                self.council_execution.provenance()
                if isinstance(self.council_execution, CouncilExecutionConfig)
                else None
            ),
            "model_routing": {
                "upstream_model": self.upstream_model,
                "upstream_provider": self.upstream_provider,
                "downstream_model": self.downstream_model,
                "downstream_provider": self.downstream_provider,
                "model_endpoints": dict(self.model_endpoints),
            },
        }


_MODEL_ROUTING_LOCK = threading.RLock()


@contextmanager
def runtime_model_routing(runtime: PipelineRuntimeOptions):
    """Apply one run's model choices and restore process defaults afterward."""

    import flora_translate.config as cfg
    from flora_translate.engine import llm_agents

    with _MODEL_ROUTING_LOCK:
        config_names = (
            "MODEL_INPUT_PARSER",
            "MODEL_CHEMISTRY_AGENT",
            "MODEL_TRANSLATION",
            "MODEL_OUTPUT_FORMATTER",
            "MODEL_CONVERSATION_AGENT",
            "ENGINE_PROVIDER",
            "ENGINE_MODEL_ANTHROPIC",
            "ENGINE_MODEL_OPENAI",
            "ENGINE_MODEL_OLLAMA",
        )
        saved_config = {name: getattr(cfg, name) for name in config_names}
        saved_engine = {
            name: getattr(llm_agents, name)
            for name in (
                "ENGINE_PROVIDER",
                "ENGINE_MODEL_ANTHROPIC",
                "ENGINE_MODEL_OPENAI",
                "ENGINE_MODEL_OLLAMA",
            )
        }
        saved_endpoints = llm_agents.get_model_endpoint_overrides()
        try:
            if runtime.upstream_model:
                cfg.MODEL_INPUT_PARSER = runtime.upstream_model
                cfg.MODEL_CHEMISTRY_AGENT = runtime.upstream_model
            if runtime.downstream_model:
                cfg.MODEL_TRANSLATION = runtime.downstream_model
                cfg.MODEL_OUTPUT_FORMATTER = runtime.downstream_model
                cfg.MODEL_CONVERSATION_AGENT = runtime.downstream_model
                provider = runtime.downstream_provider or llm_agents.infer_provider_for_model(
                    runtime.downstream_model
                )
                cfg.ENGINE_PROVIDER = provider
                llm_agents.ENGINE_PROVIDER = provider
                model_field = {
                    "anthropic": "ENGINE_MODEL_ANTHROPIC",
                    "openai": "ENGINE_MODEL_OPENAI",
                    "ollama": "ENGINE_MODEL_OLLAMA",
                }.get(provider)
                if not model_field:
                    raise ValueError(f"Unsupported downstream provider: {provider}")
                setattr(cfg, model_field, runtime.downstream_model)
                setattr(llm_agents, model_field, runtime.downstream_model)
            llm_agents.set_model_endpoint_overrides(runtime.model_endpoints)
            yield
        finally:
            for name, value in saved_config.items():
                setattr(cfg, name, value)
            for name, value in saved_engine.items():
                setattr(llm_agents, name, value)
            llm_agents.set_model_endpoint_overrides(saved_endpoints)


def with_runtime_model_routing(func):
    """Decorator that makes ``runtime_options`` model routing run-scoped."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        positional = list(args)
        raw_runtime = kwargs.get("runtime_options")
        if raw_runtime is None and len(positional) >= 4:
            raw_runtime = positional[3]
        runtime = PipelineRuntimeOptions.coerce(raw_runtime)
        if len(positional) >= 4:
            positional[3] = runtime
        else:
            kwargs["runtime_options"] = runtime
        with runtime_model_routing(runtime):
            return func(*positional, **kwargs)

    return wrapped


def merged_hard_constraints(
    runtime_constraints: Iterable[str],
    intake_constraints: Any,
) -> list[str] | Any:
    """Prefer explicit benchmark constraints, preserving GUI intake behavior."""

    values = [str(item) for item in runtime_constraints if str(item).strip()]
    return values if values else intake_constraints
