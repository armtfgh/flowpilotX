"""FLORA-Translate — Vector store: ChromaDB wrapper for record storage/retrieval."""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

import chromadb

from flora_translate.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    PAIRS_COLLECTION_NAME,
    RECORDS_DIR,
)
from flora_translate.embedding_engine import EmbeddingEngine
from flora_translate.schemas import ProcessRecord

logger = logging.getLogger("flora.vector_store")


class VectorStore:
    """ChromaDB-backed vector store for FLORA process records."""

    def __init__(self, persist_dir: str | Path = CHROMA_DIR):
        self.chroma = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)
        self.pairs_collection = self.chroma.get_or_create_collection(
            PAIRS_COLLECTION_NAME
        )
        self.engine = EmbeddingEngine()

    @property
    def count(self) -> int:
        return self.collection.count()

    @property
    def pairs_count(self) -> int:
        return self.pairs_collection.count()

    def index_record(self, record: ProcessRecord) -> None:
        """Index a single process record (generate summary, embed, store)."""
        # Generate embedding summary if not present
        if not record.embedding_summary:
            record.embedding_summary = self.engine.generate_record_summary(record)

        vector = self.engine.embed(record.embedding_summary)
        metadata = self._build_metadata(record)

        # Add to main collection
        self.collection.upsert(
            ids=[record.record_id],
            embeddings=[vector],
            documents=[record.embedding_summary],
            metadatas=[metadata],
        )

        # Add to pairs collection if has translation data
        if metadata["has_translation"]:
            self.pairs_collection.upsert(
                ids=[record.record_id],
                embeddings=[vector],
                documents=[record.embedding_summary],
                metadatas=[metadata],
            )

    def index_folder(self, folder: str | Path = RECORDS_DIR) -> int:
        """Index all JSON records in a folder. Returns count of indexed records."""
        folder = Path(folder)
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {folder}")
            return 0

        logger.info(f"Indexing {len(json_files)} records from {folder}")
        count = 0
        for f in json_files:
            try:
                data = json.loads(f.read_text())
                record = self._normalize_extraction_record(data, f.stem)
                self.index_record(record)
                count += 1
                logger.info(f"  Indexed: {record.record_id}")
            except Exception as e:
                logger.warning(f"  Failed to index {f.name}: {e}")

        logger.info(f"Indexed {count}/{len(json_files)} records")
        return count

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        pairs_only: bool = True,
        min_confidence: int = 2,
        mechanism_type: str = "",
        phase_regime: str = "",
    ) -> dict:
        """Query the vector store with optional hard metadata filters.

        Args:
            mechanism_type: If set, only return records with this mechanism type.
            phase_regime: If set, only return records with this phase regime.
        """
        collection = self.pairs_collection if pairs_only else self.collection
        count = collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Build compound where filter
        filters = [{"confidence": {"$gte": min_confidence}}]
        if mechanism_type:
            filters.append({"mechanism_type": mechanism_type})
        if phase_regime:
            filters.append({"phase_regime": phase_regime})

        if len(filters) > 1:
            where_filter = {"$and": filters}
        else:
            where_filter = filters[0]

        return collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def query_lexical(
        self,
        query_text: str,
        n_results: int = 20,
        pairs_only: bool = True,
        min_confidence: int = 2,
        mechanism_type: str = "",
        phase_regime: str = "",
    ) -> dict:
        """Deterministic TF-IDF fallback when remote embeddings are unavailable."""
        collection = self.pairs_collection if pairs_only else self.collection
        count = collection.count()
        empty = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        if count == 0:
            return empty

        filters = [{"confidence": {"$gte": min_confidence}}]
        if mechanism_type:
            filters.append({"mechanism_type": mechanism_type})
        if phase_regime:
            filters.append({"phase_regime": phase_regime})
        where_filter = (
            {"$and": filters}
            if len(filters) > 1
            else filters[0]
        )
        records = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        ids = list(records.get("ids") or [])
        documents = list(records.get("documents") or [])
        metadatas = list(records.get("metadatas") or [])
        if not ids:
            return empty

        token_pattern = re.compile(r"[a-z0-9]+")

        def tokens(text: str) -> list[str]:
            return token_pattern.findall((text or "").lower())

        document_terms = [Counter(tokens(document)) for document in documents]
        query_terms = Counter(tokens(query_text))
        document_frequency: Counter[str] = Counter()
        for terms in document_terms:
            document_frequency.update(terms.keys())
        document_count = len(document_terms)

        def weighted_vector(terms: Counter[str]) -> dict[str, float]:
            return {
                term: count_value
                * (math.log((document_count + 1) / (document_frequency[term] + 1)) + 1.0)
                for term, count_value in terms.items()
            }

        query_vector = weighted_vector(query_terms)
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        ranked = []
        for record_id, document, metadata, terms in zip(
            ids,
            documents,
            metadatas,
            document_terms,
        ):
            document_vector = weighted_vector(terms)
            document_norm = math.sqrt(
                sum(value * value for value in document_vector.values())
            )
            dot_product = sum(
                query_vector.get(term, 0.0) * value
                for term, value in document_vector.items()
            )
            denominator = query_norm * document_norm
            cosine = dot_product / denominator if denominator else 0.0
            distance = math.sqrt(max(0.0, 2.0 * (1.0 - cosine)))
            ranked.append(
                (distance, record_id, document, metadata)
            )

        ranked.sort(key=lambda item: (item[0], item[1]))
        selected = ranked[: min(n_results, len(ranked))]
        return {
            "ids": [[item[1] for item in selected]],
            "documents": [[item[2] for item in selected]],
            "metadatas": [[item[3] for item in selected]],
            "distances": [[item[0] for item in selected]],
        }

    def get_record_data(self, record_id: str) -> dict | None:
        """Retrieve the full metadata for a record by ID."""
        results = self.collection.get(ids=[record_id], include=["metadatas", "documents"])
        if results and results["ids"]:
            meta = results["metadatas"][0] if results["metadatas"] else {}
            meta["embedding_summary"] = (
                results["documents"][0] if results["documents"] else ""
            )
            return meta
        return None

    def _build_metadata(self, record: ProcessRecord) -> dict:
        """Build filterable metadata dict for ChromaDB."""
        has_translation = (
            record.translation_record is not None
            and record.translation_record.batch_baseline is not None
        )
        return {
            "doi": record.doi or "",
            "title": record.title or "",
            "year": record.year or 0,
            "chemistry_class": record.chemistry_class or "",
            "process_mode": record.process_mode or "",
            "photocatalyst": (
                record.conditions.photocatalyst or ""
                if record.conditions
                else ""
            ),
            "wavelength_nm": (
                record.conditions.wavelength_nm or 0
                if record.conditions
                else 0
            ),
            "solvent": (
                record.conditions.solvent or "" if record.conditions else ""
            ),
            "reactor_type": (
                record.process_design.reactor_type or ""
                if record.process_design
                else ""
            ),
            "has_translation": has_translation,
            "confidence": record.confidence,
            "mechanism_type": record.mechanism_type or "",
            "phase_regime": record.phase_regime or "single_phase_liquid",
        }

    def _normalize_extraction_record(
        self, data: dict, fallback_id: str
    ) -> ProcessRecord:
        """Convert a PRISM/extractor JSON output into a ProcessRecord.

        The extraction pipeline outputs a different schema than ProcessRecord,
        so we map the fields here.
        """
        chemistry = data.get("chemistry", {})
        batch = data.get("batch_baseline", {})
        flow = data.get("flow_optimized", {})
        reactor = data.get("reactor", {})
        light = data.get("light_source", {})
        pump = data.get("pump", {})
        translation = data.get("translation_logic", {})

        # Determine process_mode
        has_batch = bool(batch and batch.get("yield_percent"))
        has_flow = bool(flow and flow.get("yield_percent"))
        if has_batch and has_flow:
            process_mode = "both"
        elif has_flow:
            process_mode = "flow"
        else:
            process_mode = "batch"

        # Build conditions from flow_optimized (primary) or batch
        primary = flow if has_flow else batch
        conditions = {
            "photocatalyst": (
                chemistry.get("photocatalyst", {}).get("name")
                if isinstance(chemistry.get("photocatalyst"), dict)
                else chemistry.get("photocatalyst")
            ),
            "catalyst_loading_mol_pct": (
                chemistry.get("photocatalyst", {}).get("loading_mol_pct")
                if isinstance(chemistry.get("photocatalyst"), dict)
                else None
            ),
            "base": (
                chemistry.get("base", {}).get("name")
                if isinstance(chemistry.get("base"), dict)
                else chemistry.get("base")
            ),
            "solvent": primary.get("solvent") or batch.get("solvent"),
            "temperature_C": primary.get("temperature_C"),
            "reaction_time_h": (
                batch.get("reaction_time_min", 0) / 60
                if batch.get("reaction_time_min")
                else None
            ),
            "residence_time_min": flow.get("residence_time_min"),
            "flow_rate_mL_min": flow.get("flow_rate_total_mL_min"),
            "concentration_M": primary.get("concentration_M"),
            "yield_pct": primary.get("yield_percent"),
            "light_source": light.get("type"),
            "wavelength_nm": light.get("wavelength_nm"),
        }

        process_design = {
            "reactor_type": reactor.get("type"),
            "tubing_material": reactor.get("material"),
            "tubing_ID_mm": reactor.get("tubing_diameter_mm"),
            "reactor_volume_mL": reactor.get("volume_mL"),
            "pump_type": pump.get("type"),
            "BPR_bar": flow.get("back_pressure_bar"),
            "BPR_required": flow.get("back_pressure_bar") is not None,
            "light_setup": light.get("type"),
        }

        # Build translation record if both batch and flow exist
        translation_record = None
        if has_batch and has_flow:
            translation_record = {
                "batch_baseline": {
                    "photocatalyst": conditions["photocatalyst"],
                    "solvent": batch.get("solvent"),
                    "temperature_C": batch.get("temperature_C"),
                    "reaction_time_h": (
                        batch.get("reaction_time_min", 0) / 60
                        if batch.get("reaction_time_min")
                        else None
                    ),
                    "yield_pct": batch.get("yield_percent"),
                    "light_source": batch.get("light_source"),
                    "concentration_M": batch.get("concentration_M"),
                },
                "flow_optimized": {
                    "photocatalyst": conditions["photocatalyst"],
                    "solvent": flow.get("solvent") or batch.get("solvent"),
                    "temperature_C": flow.get("temperature_C"),
                    "residence_time_min": flow.get("residence_time_min"),
                    "flow_rate_mL_min": flow.get("flow_rate_total_mL_min"),
                    "yield_pct": flow.get("yield_percent"),
                    "concentration_M": flow.get("concentration_M"),
                },
                "flow_process_design": process_design,
                "reasoning": translation.get("flow_advantage", ""),
            }

        engineering_logic = None
        if translation:
            engineering_logic = {
                "batch_limitation": translation.get("batch_limitation"),
                "flow_advantage": translation.get("flow_advantage"),
                "safety_concern": translation.get("safety_improvement"),
                "intensification_factor": translation.get("time_reduction_factor"),
                "batch_to_flow_yield_delta": translation.get("yield_change_percent"),
            }

        # Extract mechanism_type from chemistry block
        mechanism_type = chemistry.get("mechanism_type", "")
        if isinstance(mechanism_type, dict):
            mechanism_type = mechanism_type.get("type", "")

        return ProcessRecord(
            record_id=data.get("source_pdf", fallback_id),
            doi=data.get("doi", ""),
            title=data.get("title", ""),
            year=data.get("year", 0),
            chemistry_class=chemistry.get("reaction_class", ""),
            mechanism_type=mechanism_type,
            process_mode=process_mode,
            conditions=conditions,
            process_design=process_design,
            engineering_logic=engineering_logic,
            translation_record=translation_record,
            confidence=data.get("confidence_overall", 1),
        )
