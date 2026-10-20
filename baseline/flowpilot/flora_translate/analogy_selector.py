"""FLORA-Translate — Analogy selector: picks best batch→flow pairs from retrieval."""

import json
import logging
from pathlib import Path

from flora_translate.config import RECORDS_DIR
from flora_translate.schemas import ProcessRecord

logger = logging.getLogger("flora.analogy_selector")


class AnalogySelector:
    """Load full record data for retrieved analogies."""

    def __init__(self, records_dir: str | Path = RECORDS_DIR):
        self.records_dir = Path(records_dir)
        self._cache: dict[str, dict] = {}

    def load_full_record(self, record_id: str) -> dict | None:
        """Load the full JSON record from disk by record_id."""
        if record_id in self._cache:
            return self._cache[record_id]

        # record_id from ChromaDB may carry a .pdf extension — strip it
        stem = record_id.removesuffix(".pdf") if record_id.endswith(".pdf") else record_id

        candidates = [
            self.records_dir / f"{stem}.json",
            self.records_dir / f"{record_id}.json",
            self.records_dir / f"{record_id}",
        ]
        for path in candidates:
            if path.exists():
                data = json.loads(path.read_text())
                self._cache[record_id] = data
                return data

        # Fuzzy match: check both stem-in-filename and filename-in-stem
        for f in self.records_dir.glob("*.json"):
            if stem in f.stem or f.stem in stem:
                data = json.loads(f.read_text())
                self._cache[record_id] = data
                return data

        logger.warning(f"Could not load full record for {record_id}")
        return None

    def select(self, retrieval_results: list[dict]) -> list[dict]:
        """Enrich retrieval results with full record data.

        Returns the same list with a 'full_record' key added to each entry.
        """
        enriched = []
        for result in retrieval_results:
            record_data = self.load_full_record(result["record_id"])
            result["full_record"] = record_data
            enriched.append(result)
        return enriched
