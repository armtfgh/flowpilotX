"""
FLORA — Topology Polisher.

A lightweight LLM agent (Claude Haiku) that reviews the ProcessTopology
for common-sense logical errors BEFORE rendering, and returns a cleaned
version. Fast and cheap — Haiku processes a typical topology in <2s.

Issues it detects and fixes:
  - Pass-through mixers (single input, no new injection → redundant)
  - Two consecutive mixers that a single cross-mixer could replace
  - Mixers immediately following another mixer with no intervening reactor
  - Any other logically absurd sequential arrangement

Usage (called automatically by FlowsheetBuilder):
    from flora_translate.topology_polisher import polish
    clean_topology = polish(topology)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_text

logger = logging.getLogger("flora.topology_polisher")


# ── Topology → human-readable description ─────────────────────────────────────

def _describe_topology(topology) -> str:
    """Convert ProcessTopology into a plain-English numbered node list."""
    ops    = topology.unit_operations
    streams = topology.streams

    out_edges: dict[str, list[str]] = defaultdict(list)
    in_edges:  dict[str, list[str]] = defaultdict(list)
    for s in streams:
        out_edges[s.from_op].append(s.to_op)
        in_edges[s.to_op].append(s.from_op)

    op_map = {o.op_id: o for o in ops}

    lines = ["NODES:"]
    for i, op in enumerate(ops, 1):
        p = op.parameters or {}
        details = []
        if p.get("temperature_C"):
            details.append(f"{p['temperature_C']}°C")
        if p.get("residence_time_min"):
            details.append(f"{p['residence_time_min']:.0f} min")
        if p.get("wavelength_nm"):
            details.append(f"λ={p['wavelength_nm']:.0f}nm")
        if p.get("pressure_bar"):
            details.append(f"{p['pressure_bar']} bar")
        if p.get("contents"):
            contents = p["contents"]
            if isinstance(contents, list):
                details.append("carries: " + ", ".join(str(c).split("(")[0].strip()
                                                         for c in contents[:2]))
        detail_str = f" [{', '.join(details)}]" if details else ""
        lines.append(f"  {i}. [{op.op_id}] {op.op_type}: {op.label}{detail_str}")

    lines.append("\nCONNECTIONS (from → to):")
    for s in streams:
        src = op_map.get(s.from_op)
        dst = op_map.get(s.to_op)
        src_lbl = f"{src.op_type}({s.from_op})" if src else s.from_op
        dst_lbl = f"{dst.op_type}({s.to_op})" if dst else s.to_op
        lines.append(f"  {src_lbl} → {dst_lbl}")

    return "\n".join(lines)


# ── Deterministic redundancy check (fast path, no LLM) ────────────────────────

def _find_passthrough_mixers(topology) -> set[str]:
    """
    Find mixers that have exactly 1 input and 0 pump inputs — they do nothing
    except pass the flow through. These are safe to remove deterministically.
    """
    ops    = topology.unit_operations
    streams = topology.streams

    in_edges: dict[str, list[str]] = defaultdict(list)
    for s in streams:
        in_edges[s.to_op].append(s.from_op)

    op_map   = {o.op_id: o for o in ops}
    pump_ids = {o.op_id for o in ops if o.op_type == "pump"}
    led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}
    MIXER_TYPES = {"mixer", "t_mixer", "y_mixer", "quench_mixer"}

    to_remove = set()
    for op in ops:
        if op.op_type not in MIXER_TYPES:
            continue
        inputs     = in_edges.get(op.op_id, [])
        real_inputs = [i for i in inputs
                       if i not in led_ids and i in op_map]
        pump_inputs = [i for i in real_inputs if i in pump_ids]
        main_inputs = [i for i in real_inputs if i not in pump_ids]

        # Pass-through: only 1 input, no pump injection
        if len(real_inputs) == 1 and len(pump_inputs) == 0:
            to_remove.add(op.op_id)
            logger.info(f"  Polisher: pass-through mixer '{op.op_id}' → removing")

    return to_remove


# ── LLM review (catches more complex issues) ───────────────────────────────────

_SYSTEM = """\
You are a flow chemistry process engineer reviewing an automated process topology
for logical errors BEFORE it is rendered as a diagram.

Your job: identify unit operations that are LOGICALLY REDUNDANT and should be
removed. Focus on:

1. Mixers with a single input and no pump injection — they pass flow through
   without mixing anything (redundant).
2. Two consecutive mixers (A → B) where mixer A has only one non-pump input —
   it could be eliminated by connecting its input directly to mixer B.
3. Any other obvious common-sense errors.

DO NOT remove:
- Reactors, BPRs, collectors, separators, filters
- Mixers that genuinely combine multiple streams
- The first mixer that combines initial pump feeds

Return ONLY valid JSON:
{
  "remove": ["op_id_1", "op_id_2"],
  "reasoning": "one sentence explaining what was wrong"
}

If nothing needs removal, return: {"remove": [], "reasoning": "topology is clean"}
"""


def _llm_review(topology) -> list[str]:
    """Ask Claude Haiku which op_ids to remove. Returns list of op_ids."""
    description = _describe_topology(topology)
    try:
        result = call_model_text(
            model=cfg.MODEL_TOPOLOGY_POLISHER,
            api_name="topology_polisher",
            max_tokens=256,
            system=_SYSTEM,
            user_content=description,
        )
        raw = result.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        ids_to_remove = result.get("remove", [])
        reasoning = result.get("reasoning", "")
        if ids_to_remove:
            logger.info(f"  Polisher (LLM): removing {ids_to_remove} — {reasoning}")
        else:
            logger.debug(f"  Polisher (LLM): topology clean — {reasoning}")
        return ids_to_remove
    except Exception as e:
        logger.warning(f"  Polisher LLM call failed ({e}) — skipping LLM review")
        return []


def _protected_removal_ids(topology) -> set[str]:
    """Return operation ids that must not be removed by cleanup passes."""
    streams = topology.streams
    op_map = {o.op_id: o for o in topology.unit_operations}
    protected: set[str] = set()

    def _is_gas_source(op_id: str) -> bool:
        op = op_map.get(op_id)
        if not op:
            return False
        params = op.parameters or {}
        return (
            op.op_type == "mfc"
            or str(params.get("phase") or "").lower() == "gas"
            or params.get("gas_flow_sccm") is not None
            or params.get("gas_flow_actual_mL_min") is not None
        )

    for op in topology.unit_operations:
        if op.op_type not in {"mixer", "t_mixer", "y_mixer", "quench_mixer"}:
            continue
        inputs = [s.from_op for s in streams if s.to_op == op.op_id]
        outputs = [s.to_op for s in streams if s.from_op == op.op_id]
        downstream_gas_reactor = any(
            (op_map.get(dst) and (op_map[dst].parameters or {}).get("gas_holdup"))
            for dst in outputs
        )
        if any(_is_gas_source(src) for src in inputs) or downstream_gas_reactor:
            protected.add(op.op_id)

    return protected


# ── Apply removals ─────────────────────────────────────────────────────────────

def _apply_removals(topology, remove_ids: set[str]):
    """
    Remove the specified op_ids from the topology and reconnect streams
    so that each removed node's inputs connect directly to its outputs.
    """
    if not remove_ids:
        return topology

    ops     = topology.unit_operations
    streams = topology.streams

    out_edges: dict[str, list[str]] = defaultdict(list)
    in_edges:  dict[str, list[str]] = defaultdict(list)
    for s in streams:
        out_edges[s.from_op].append(s.to_op)
        in_edges[s.to_op].append(s.from_op)

    # Build bypass connections around removed nodes
    from flora_translate.schemas import StreamConnection
    sc_counter = [len(streams) + 100]

    new_streams = []
    for s in streams:
        # Skip streams involving removed nodes
        if s.from_op in remove_ids or s.to_op in remove_ids:
            continue
        new_streams.append(s)

    # For each removed node, connect its inputs directly to its outputs
    for removed_id in remove_ids:
        srcs = [s.from_op for s in streams
                if s.to_op == removed_id and s.from_op not in remove_ids]
        dsts = [s.to_op for s in streams
                if s.from_op == removed_id and s.to_op not in remove_ids]
        for src in srcs:
            for dst in dsts:
                sc_counter[0] += 1
                incoming_phases = {
                    s.stream_type
                    for s in streams
                    if s.to_op == removed_id and s.from_op == src
                }
                outgoing_phases = {
                    s.stream_type
                    for s in streams
                    if s.from_op == removed_id and s.to_op == dst
                }
                phase = next(iter(outgoing_phases or incoming_phases or {"liquid"}))
                new_streams.append(StreamConnection(
                    stream_id=f"bypass_{sc_counter[0]}",
                    from_op=src, to_op=dst, stream_type=phase,
                ))
                logger.info(f"  Polisher: reconnected {src} → {dst} "
                            f"(bypassing {removed_id})")

    new_ops = [o for o in ops if o.op_id not in remove_ids]

    # Return a new topology with updated ops and streams
    import copy
    new_topo = copy.copy(topology)
    object.__setattr__(new_topo, "unit_operations", new_ops)
    object.__setattr__(new_topo, "streams", new_streams)
    from flora_translate.topology_semantics import normalize_topology_semantics

    return normalize_topology_semantics(new_topo)


# ── Public API ─────────────────────────────────────────────────────────────────

def polish(topology, use_llm: bool = True) -> object:
    """
    Review and clean the topology before rendering.

    Steps:
      1. Deterministic pass: remove obvious pass-through mixers (fast, free)
      2. LLM pass (Claude Haiku): catch more complex redundancies (optional)

    Parameters
    ----------
    topology    : ProcessTopology
    use_llm     : bool — set False to skip the LLM call (e.g. in tests)

    Returns
    -------
    Cleaned ProcessTopology (may be the same object if nothing changed)
    """
    # Step 1: fast deterministic check
    remove = _find_passthrough_mixers(topology)
    protected = _protected_removal_ids(topology)
    remove -= protected

    # Step 2: LLM review (adds ~1-2s, costs ~$0.0001 per call)
    if use_llm:
        llm_remove = set(_llm_review(topology))
        # Only remove ops that are actually mixers or clearly redundant
        # (guard against hallucination removing important nodes)
        op_map = {o.op_id: o for o in topology.unit_operations}
        SAFE_TO_REMOVE = {"mixer", "t_mixer", "y_mixer", "led_module",
                          "deoxygenation_unit"}
        llm_remove = {
            op_id for op_id in llm_remove
            if op_id in op_map and op_map[op_id].op_type in SAFE_TO_REMOVE
        }
        llm_remove -= protected
        remove |= llm_remove

    if not remove:
        return topology

    return _apply_removals(topology, remove)
