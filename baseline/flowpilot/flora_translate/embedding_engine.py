"""FLORA-Translate — Embedding engine: generates summaries and embeddings."""

import json
import logging

from openai import OpenAI

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.schemas import BatchRecord, ProcessRecord

logger = logging.getLogger("flora.embedding")
_embedding_provider_available: bool | None = None


def _get_openai():
    from flora_translate.engine.llm_agents import build_verified_httpx_client

    return OpenAI(http_client=build_verified_httpx_client())

SUMMARY_SYSTEM = (
    "You are a chemistry expert. Summarize this flow chemistry process record "
    "in 4-6 sentences of natural language. Include: reaction type, photocatalyst "
    "(if any), key conditions (temperature, residence time, solvent), reactor "
    "type, and the main reason why flow improved on batch. Be specific and "
    "include numbers. Do not say 'this paper' — write as a direct factual "
    "description."
)

QUERY_SUMMARY_SYSTEM = (
    "You are a chemistry expert. Given this batch protocol, write a 3-4 sentence "
    "natural language description for semantic search. Include: reaction type, "
    "photocatalyst name and class, solvent, temperature, reaction time, and any "
    "special features (deoxygenation, sensitizer, etc.). Be specific and include "
    "all numerical values present."
)


class EmbeddingEngine:
    """Generate text summaries and vector embeddings."""

    def generate_record_summary(self, record: ProcessRecord) -> str:
        """Generate a natural language summary of a process record for embedding."""
        result = call_model_text(
            model=cfg.MODEL_EMBEDDING_SUMMARY,
            api_name="embedding_record_summary",
            max_tokens=512,
            system=SUMMARY_SYSTEM,
            user_content=json.dumps(record.model_dump(), indent=2, default=str),
        )
        return result.text.strip()

    def generate_query_summary(self, batch_record: BatchRecord) -> str:
        """Generate a search-optimized summary of a batch protocol."""
        result = call_model_text(
            model=cfg.MODEL_EMBEDDING_SUMMARY,
            api_name="embedding_query_summary",
            max_tokens=256,
            system=QUERY_SUMMARY_SYSTEM,
            user_content=json.dumps(
                batch_record.model_dump(exclude_none=True), indent=2
            ),
        )
        return result.text.strip()

    def embed(self, text: str) -> list[float]:
        """Generate a vector embedding using OpenAI text-embedding-3-small."""
        global _embedding_provider_available
        if _embedding_provider_available is False:
            raise RuntimeError(
                "embedding provider disabled after an earlier request failure"
            )
        try:
            resp = _get_openai().embeddings.create(
                input=text,
                model=cfg.EMBEDDING_MODEL,
            )
        except Exception:
            _embedding_provider_available = False
            raise
        _embedding_provider_available = True
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        global _embedding_provider_available
        if _embedding_provider_available is False:
            raise RuntimeError(
                "embedding provider disabled after an earlier request failure"
            )
        try:
            resp = _get_openai().embeddings.create(
                input=texts,
                model=cfg.EMBEDDING_MODEL,
            )
        except Exception:
            _embedding_provider_available = False
            raise
        _embedding_provider_available = True
        return [item.embedding for item in resp.data]
