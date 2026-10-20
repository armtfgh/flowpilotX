"""FLORA-Prism — Wrapper around paper_knowledge_extractor for Streamlit.

Delegates to the existing 5-pass extraction pipeline.
"""

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("flora.prism")


class Prism:
    """PDF extraction wrapper for the Streamlit GUI."""

    def __init__(
        self,
        provider: str = "anthropic",
        passes: list[str] | None = None,
    ):
        self.provider = provider
        self.passes = passes or [
            "document_map", "chemistry", "process", "figures", "synthesis"
        ]

    def extract(self, uploaded_file) -> dict:
        """Extract structured data from an uploaded PDF file.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Extracted JSON record dict
        """
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        logger.info(f"Extracting: {uploaded_file.name}")

        # Import and run the existing extractor
        import paper_knowledge_extractor as pke

        # Set provider (anthropic / openai)
        pke.set_provider(self.provider)

        try:
            # Run extraction (saves to disk automatically)
            output_dir = Path("extraction_results")
            result = pke.extract_paper(tmp_path, output_dir)
            if result is None:
                # Already extracted — load from disk
                stem = Path(uploaded_file.name).stem
                result_path = output_dir / f"{stem}.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    result = {"error": "Extraction returned None", "source_pdf": uploaded_file.name}
            return result
        finally:
            pke.set_provider("anthropic")   # reset to default
            Path(tmp_path).unlink(missing_ok=True)


def index_records(records: list[dict], records_dir: str = "flora_translate/data/records") -> int:
    """Save records to disk and index into ChromaDB."""
    records_path = Path(records_dir)
    records_path.mkdir(parents=True, exist_ok=True)

    # Save each record as JSON
    count = 0
    for record in records:
        if not record or "error" in record:
            continue
        name = record.get("source_pdf", f"record_{count}").replace(".pdf", "")
        out = records_path / f"{name}.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        count += 1

    # Index into ChromaDB
    from flora_translate.vector_store import VectorStore
    store = VectorStore()
    indexed = store.index_folder(str(records_path))
    logger.info(f"Saved {count} records, indexed {indexed} into ChromaDB")
    return indexed
