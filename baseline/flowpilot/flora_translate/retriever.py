"""FLORA-Translate — Retriever: semantic search + field-based reranking."""

import logging

from flora_translate.config import (
    METAL_FAMILIES,
    PHOTOCATALYST_CLASSES,
    TOP_K_ANALOGIES,
    TOP_K_RETRIEVAL,
    W_CONCENTRATION,
    W_FIELD,
    W_PHOTOCATALYST,
    W_SEMANTIC,
    W_SOLVENT,
    W_TEMPERATURE,
    W_WAVELENGTH,
)
from flora_translate.embedding_engine import EmbeddingEngine
from flora_translate.schemas import BatchRecord, ChemistryPlan
from flora_translate.vector_store import VectorStore

logger = logging.getLogger("flora.retriever")


def _get_catalyst_class(name: str | None) -> str | None:
    """Look up photocatalyst class, trying exact then substring match."""
    if not name:
        return None
    # Exact match
    if name in PHOTOCATALYST_CLASSES:
        return PHOTOCATALYST_CLASSES[name]
    # Substring match
    name_lower = name.lower()
    for key, cls in PHOTOCATALYST_CLASSES.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return cls
    return None


def _get_metal_family(cls: str | None) -> str | None:
    if not cls:
        return None
    return METAL_FAMILIES.get(cls)


def photocatalyst_class_match(query_cat: str | None, result_cat: str | None) -> float:
    """Score photocatalyst similarity: 1.0 same class, 0.5 same family, 0.0 else."""
    q_cls = _get_catalyst_class(query_cat)
    r_cls = _get_catalyst_class(result_cat)
    if q_cls and r_cls:
        if q_cls == r_cls:
            return 1.0
        if _get_metal_family(q_cls) == _get_metal_family(r_cls):
            return 0.5
    return 0.0


def solvent_match(query_solvent: str | None, result_solvent: str | None) -> float:
    """1.0 if same solvent, 0.0 otherwise."""
    if not query_solvent or not result_solvent:
        return 0.0
    return 1.0 if query_solvent.lower() == result_solvent.lower() else 0.0


def wavelength_match(
    query_wl: float | None, result_wl: float | None
) -> float:
    """1.0 if within 30nm, linear decay to 0 at 100nm difference."""
    if not query_wl or not result_wl:
        return 0.0
    diff = abs(query_wl - result_wl)
    if diff <= 30:
        return 1.0
    if diff >= 100:
        return 0.0
    return 1.0 - (diff - 30) / 70


def temperature_proximity(
    query_t: float | None, result_t: float | None
) -> float:
    """1.0 if within 10°C, linear decay to 0 at 50°C difference."""
    if query_t is None or result_t is None:
        return 0.0
    diff = abs(query_t - result_t)
    if diff <= 10:
        return 1.0
    if diff >= 50:
        return 0.0
    return 1.0 - (diff - 10) / 40


def concentration_proximity(
    query_c: float | None, result_c: float | None
) -> float:
    """1.0 if within 2x, 0.0 if beyond 5x ratio."""
    if not query_c or not result_c or result_c == 0:
        return 0.0
    ratio = max(query_c, result_c) / min(query_c, result_c)
    if ratio <= 2:
        return 1.0
    if ratio >= 5:
        return 0.0
    return 1.0 - (ratio - 2) / 3


def compute_field_similarity(query: BatchRecord, metadata: dict) -> float:
    """Compute weighted field similarity score."""
    return (
        W_PHOTOCATALYST
        * photocatalyst_class_match(query.photocatalyst, metadata.get("photocatalyst"))
        + W_SOLVENT * solvent_match(query.solvent, metadata.get("solvent"))
        + W_WAVELENGTH * wavelength_match(query.wavelength_nm, metadata.get("wavelength_nm"))
        + W_TEMPERATURE * temperature_proximity(query.temperature_C, metadata.get("temperature_C"))
        + W_CONCENTRATION * concentration_proximity(query.concentration_M, metadata.get("concentration_M"))
    )


class VectorRetriever:
    """Two-stage retriever: semantic search → field-based reranking."""

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.engine = EmbeddingEngine()

    def retrieve(
        self,
        batch_record: BatchRecord,
        top_k: int = TOP_K_ANALOGIES,
        chemistry_plan: ChemistryPlan | None = None,
        exclude_record_ids: set[str] | None = None,
        retrieval_mode: str = "semantic",
    ) -> list[dict]:
        """Retrieve top-k analogies for a batch record.

        If a ChemistryPlan is provided (plan-aware retrieval), the query
        is enriched with mechanism keywords, reaction class, and retrieval
        hints — producing much better semantic matches.

        Returns list of dicts with keys:
            record_id, metadata, summary, semantic_score,
            field_score, final_score
        """
        if retrieval_mode not in {"semantic", "lexical"}:
            raise ValueError("retrieval_mode must be 'semantic' or 'lexical'")

        # Build query text — explicit lexical retrieval must remain local even
        # when the caller has not supplied an upstream chemistry plan.
        if chemistry_plan:
            query_text = self._build_plan_aware_query(batch_record, chemistry_plan)
            logger.info("Using plan-aware retrieval query")
        elif retrieval_mode == "lexical":
            query_text = batch_record.model_dump_json(exclude_none=True)
        else:
            query_text = self.engine.generate_query_summary(batch_record)
        logger.info(f"Query summary: {query_text[:120]}...")
        query_vector = None
        if retrieval_mode == "semantic":
            try:
                query_vector = self.engine.embed(query_text)
            except Exception as exc:
                logger.warning(
                    "Embedding provider unavailable; using deterministic lexical "
                    "retrieval fallback: %s",
                    exc,
                )
        else:
            logger.info("Using deterministic lexical retrieval (remote embeddings disabled)")

        def query_store(
            *,
            pairs_only: bool,
            mechanism_type: str = "",
            phase_regime: str = "",
        ) -> dict:
            if query_vector is not None:
                return self.store.query(
                    query_embedding=query_vector,
                    n_results=TOP_K_RETRIEVAL,
                    pairs_only=pairs_only,
                    mechanism_type=mechanism_type,
                    phase_regime=phase_regime,
                )
            return self.store.query_lexical(
                query_text=query_text,
                n_results=TOP_K_RETRIEVAL,
                pairs_only=pairs_only,
                mechanism_type=mechanism_type,
                phase_regime=phase_regime,
            )

        # Stage A: semantic search with hard metadata filters
        # Extract hard filters from ChemistryPlan if available
        hard_mechanism = ""
        hard_phase = ""
        if chemistry_plan:
            # Normalise mechanism_type to match indexed values
            mech = (chemistry_plan.mechanism_type or "").lower().strip()
            if mech:
                hard_mechanism = mech
            phase = getattr(chemistry_plan, "phase_regime", "") or ""
            if not phase:
                # Infer from stream_logic or other signals
                pass
            else:
                hard_phase = phase

        # Try 1: hard filters + pairs only
        results = query_store(
            pairs_only=True,
            mechanism_type=hard_mechanism,
            phase_regime=hard_phase,
        )

        n_hard = len(results["ids"][0]) if results["ids"] and results["ids"][0] else 0

        # Fallback: if hard filters returned < 3 results, relax filters
        if n_hard < 3:
            if hard_mechanism or hard_phase:
                logger.info(
                    f"  Hard filters returned {n_hard} results "
                    f"(mechanism={hard_mechanism}, phase={hard_phase}). "
                    f"Relaxing to soft filtering."
                )
            # Try 2: no hard filters, pairs only
            results = query_store(pairs_only=True)

        if not results["ids"] or not results["ids"][0]:
            logger.warning("No results from pairs search. Trying all records.")
            # Try 3: no filters, all records
            results = query_store(pairs_only=False)

        if not results["ids"] or not results["ids"][0]:
            logger.warning("No records in vector store")
            return []

        # Log filter status
        if n_hard >= 3:
            logger.info(f"  Hard filter matched {n_hard} records (mechanism={hard_mechanism})")


        # Stage B: rerank by field similarity
        candidates = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for i, rid in enumerate(ids):
            # ChromaDB returns L2 (Euclidean) distances.
            # text-embedding-3-small embeddings are L2-normalised (unit vectors),
            # so: cosine_similarity = 1 − L2² / 2.
            # This is a much more accurate similarity measure than 1/(1+L2),
            # which severely underestimates similarity for close matches.
            L2 = distances[i]
            semantic_score = max(0.0, 1.0 - (L2 ** 2) / 2.0)
            field_score = compute_field_similarity(batch_record, metadatas[i])
            final_score = W_SEMANTIC * semantic_score + W_FIELD * field_score

            candidates.append(
                {
                    "record_id": rid,
                    "metadata": metadatas[i],
                    "summary": documents[i],
                    "semantic_score": round(semantic_score, 4),
                    "field_score": round(field_score, 4),
                    "final_score": round(final_score, 4),
                    "retrieval_method": "semantic" if query_vector is not None else "lexical",
                    "quantitative_transfer": {
                        "eligible": bool(query_vector is not None and chemistry_plan
                            and chemistry_plan.reaction_class
                            and str(metadatas[i].get("chemistry_class") or metadatas[i].get("reaction_class") or "").strip().casefold() == chemistry_plan.reaction_class.strip().casefold()
                            and chemistry_plan.mechanism_type
                            and str(metadatas[i].get("mechanism_type") or "").strip().casefold() == chemistry_plan.mechanism_type.strip().casefold()),
                        "basis": "Quantitative screening priors require matched class and mechanism plus semantic relevance; lexical or broader analogies are qualitative only. No retrieval score validates kinetics.",
                    },
                }
            )

        # Leave-one-source-out evaluation can exclude the record that supplied
        # the hidden reference answer. Normal application calls omit this.
        excluded = {str(value).strip().lower() for value in (exclude_record_ids or set())}
        if excluded:
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("record_id", "")).strip().lower() not in excluded
            ]

        # Sort by final score descending
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        top = candidates[:top_k]
        for i, c in enumerate(top):
            logger.info(
                f"  Analogy {i+1}: {c['record_id']} "
                f"(score={c['final_score']:.3f}, "
                f"sem={c['semantic_score']:.3f}, "
                f"field={c['field_score']:.3f})"
            )

        return top

    def _build_plan_aware_query(
        self, batch_record: BatchRecord, plan: ChemistryPlan
    ) -> str:
        """Build a retrieval query enriched with chemistry plan insights.

        Instead of just describing the batch protocol, this query includes
        mechanism type, reaction class, key intermediate, and retrieval
        keywords — making semantic search much more targeted.
        """
        parts = []

        # Reaction identity from plan (more specific than raw batch description)
        if plan.reaction_name:
            parts.append(f"{plan.reaction_name}.")
        if plan.reaction_class:
            parts.append(f"Reaction class: {plan.reaction_class}.")
        if plan.mechanism_type:
            parts.append(f"Mechanism: {plan.mechanism_type}.")

        # Key species
        catalyst_names = [
            r.name for r in plan.reagents
            if r.role in ("photocatalyst", "sensitizer", "co-catalyst") and r.name
        ]
        if catalyst_names:
            parts.append(f"Photocatalyst: {', '.join(catalyst_names)}.")

        # Conditions from batch record
        if batch_record.solvent:
            parts.append(f"Solvent: {batch_record.solvent}.")
        if batch_record.temperature_C is not None:
            parts.append(f"Temperature: {batch_record.temperature_C}°C.")
        if batch_record.wavelength_nm:
            parts.append(f"Wavelength: {batch_record.wavelength_nm} nm.")
        elif plan.recommended_wavelength_nm:
            parts.append(f"Wavelength: {plan.recommended_wavelength_nm} nm.")

        # Key intermediate and bond formed (highly discriminative)
        if plan.key_intermediate:
            parts.append(f"Key intermediate: {plan.key_intermediate}.")
        if plan.bond_formed:
            parts.append(f"Bond formed: {plan.bond_formed}.")

        # Retrieval keywords from chemistry agent
        if plan.retrieval_keywords:
            parts.append(f"Keywords: {', '.join(plan.retrieval_keywords[:6])}.")

        # Similar reaction classes
        if plan.similar_reaction_classes:
            parts.append(
                f"Similar reactions: {', '.join(plan.similar_reaction_classes[:4])}."
            )

        # Flow-relevant notes
        if plan.oxygen_sensitive:
            parts.append("Oxygen-sensitive reaction requiring deoxygenation.")

        return " ".join(parts)
