"""FLORA-Design — Topology Agent: RAG-based topology generation."""

import json
import logging
import re
from pathlib import Path

import anthropic

from flora_translate.config import TRANSLATION_MODEL
from flora_translate.embedding_engine import EmbeddingEngine
from flora_translate.schemas import (
    ChemFeatures,
    ProcessTopology,
    UnitOperation,
)
from flora_translate.vector_store import VectorStore

logger = logging.getLogger("flora.design.topology")

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


def _format_records(results: dict) -> str:
    """Format ChromaDB results into text for the prompt."""
    if not results or not results.get("ids") or not results["ids"][0]:
        return "No similar records found in corpus."

    blocks = []
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i, rid in enumerate(ids):
        meta = metadatas[i]
        summary = documents[i]
        blocks.append(
            f"Record {i+1}: {rid}\n"
            f"  DOI: {meta.get('doi', 'N/A')} | "
            f"Year: {meta.get('year', 'N/A')}\n"
            f"  Photocatalyst: {meta.get('photocatalyst', 'N/A')}\n"
            f"  Reactor: {meta.get('reactor_type', 'N/A')}\n"
            f"  Solvent: {meta.get('solvent', 'N/A')}\n"
            f"  Summary: {summary[:200]}"
        )
    return "\n\n".join(blocks)


def _normalize_topology_data(data: dict) -> dict:
    """Fix common LLM output shape deviations before Pydantic validation."""
    # literature_support: expected list[str] but LLM may return list[dict]
    support = data.get("literature_support", [])
    if support and isinstance(support[0], dict):
        normalized = []
        for item in support:
            if isinstance(item, dict):
                # Extract DOI or any string value from the dict
                doi = item.get("doi") or item.get("DOI") or item.get("record")
                if doi:
                    normalized.append(str(doi))
                else:
                    # Fall back to first string value found
                    for v in item.values():
                        if isinstance(v, str):
                            normalized.append(v)
                            break
            else:
                normalized.append(str(item))
        data["literature_support"] = normalized

    # streams: StreamConnection fields may come with wrong key names
    streams = data.get("streams", [])
    for s in streams:
        if isinstance(s, dict):
            # LLM sometimes uses "source"/"target" instead of "from_op"/"to_op"
            if "source" in s and "from_op" not in s:
                s["from_op"] = s.pop("source")
            if "target" in s and "to_op" not in s:
                s["to_op"] = s.pop("target")
            if "from" in s and "from_op" not in s:
                s["from_op"] = s.pop("from")
            if "to" in s and "to_op" not in s:
                s["to_op"] = s.pop("to")

    return data


class TopologyAgent:
    """Use RAG to refine unit operations and generate process topology."""

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.engine = EmbeddingEngine()

    def run(
        self,
        features: ChemFeatures,
        ops: list[UnitOperation],
    ) -> tuple[ProcessTopology, ProcessTopology | None]:
        """Generate primary and alternative topologies.

        Returns (primary_topology, alternative_topology).
        """
        # Retrieve similar records
        records = self.retrieve(features)

        # Generate primary topology
        logger.info("  Generating primary topology")
        primary = self._generate_topology(features, ops, records)

        # Generate alternative (optional, may fail)
        alternative = None
        try:
            logger.info("  Generating alternative topology")
            alternative = self._generate_topology(
                features, ops, records, variant="alternative"
            )
        except Exception as e:
            logger.warning(f"  Alternative topology failed: {e}")

        return primary, alternative

    def retrieve(self, features: ChemFeatures) -> dict:
        """Retrieve similar flow chemistry records from the corpus."""
        query = (
            f"{features.reaction_class.replace('_', ' ')} "
            f"photocatalyst {features.photocatalyst or features.photocatalyst_class or ''} "
            f"flow chemistry {features.solvent or ''} "
            f"{features.wavelength_nm or ''}nm LED"
        )
        query_vector = self.engine.embed(query)

        results = self.store.query(
            query_embedding=query_vector,
            n_results=5,
            pairs_only=False,
            min_confidence=1,
        )
        return results

    def _generate_topology(
        self,
        features: ChemFeatures,
        ops: list[UnitOperation],
        records: dict,
        variant: str = "primary",
    ) -> ProcessTopology:
        system = (PROMPTS_DIR / "topology_system.txt").read_text()

        extra = ""
        if variant == "alternative":
            extra = (
                "\n\nFor this ALTERNATIVE topology: propose a meaningfully "
                "different option — e.g., chip reactor instead of coil, "
                "different solvent, or omit deoxygenation if literature supports it. "
                'Set topology_id to "alternative".'
            )
            system += extra

        user_template = (PROMPTS_DIR / "topology_user.txt").read_text()
        user = user_template.format(
            chem_features_json=json.dumps(
                features.model_dump(exclude_none=True), indent=2
            ),
            unit_ops_json=json.dumps(
                [op.model_dump() for op in ops], indent=2
            ),
            records_text=_format_records(records),
        )

        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        data = _parse_json(resp.content[0].text)
        data = _normalize_topology_data(data)

        # Ensure consistency
        rt = data.get("residence_time_min", 0)
        fr = data.get("total_flow_rate_mL_min", 0)
        if rt > 0 and fr > 0:
            data["reactor_volume_mL"] = round(rt * fr, 2)

        topo = ProcessTopology(**data)
        logger.info(
            f"    {variant}: {len(topo.unit_operations)} ops, "
            f"tau={topo.residence_time_min}min, Q={topo.total_flow_rate_mL_min}mL/min"
        )
        return topo
