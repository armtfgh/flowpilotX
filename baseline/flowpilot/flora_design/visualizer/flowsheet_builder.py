"""
FLORA — Graphviz-based Flowsheet Builder.

Replaces the old hand-drawn SVG approach with a professional Graphviz diagram
using equipment icons (pump, coil reactor, BPR, vial, etc.).

Icon files live in:  flora_design/visualizer/icons/

Features:
  - Pump nodes: generic icon + reagent list + solvent + flow rate (fully readable)
  - Reactor nodes: coil image + temp / residence time / wavelength / ID / volume
  - Mixer nodes: clean grey box with correct in/out ports
  - BPR / collector: icon + label
  - LED module: SKIPPED — wavelength shown under reactor
  - Main process: horizontal icon-center connections; side feeds branch in
  - Mixer connections: preserved exactly; no invented upstream mixers
  - Optional check-valve icon: icons/check_valve.png, otherwise a labeled box
  - Unknown op types: plain text box

Falls back silently to the legacy SVG builder if graphviz executable
is not found on the system.

Public API (unchanged from legacy):
    FlowsheetBuilder().build(topology, title, output_svg, output_png)
    → (svg_path, png_path)
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
from copy import deepcopy
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flora.flowsheet")

# ── Paths ─────────────────────────────────────────────────────────────────────
ICONS_DIR = Path(__file__).parent / "icons"

ASSETS = {
    "pump":         ICONS_DIR / "syringe_pump.png",
    "reactor":      ICONS_DIR / "coil_reactor.png",
    "photoreactor": ICONS_DIR / "photoreactor.png",
    "microchannel": ICONS_DIR / "microchannel.png",
    "bpr":          ICONS_DIR / "bpr.png",
    "vial":         ICONS_DIR / "vial.png",
    "mixer2":       ICONS_DIR / "mixer2.png",
    "mixer3":       ICONS_DIR / "mixer3.png",
    "degasser":     ICONS_DIR / "degasser.png",
    "mfc":          ICONS_DIR / "mfc.png",
    "g_l_separator": ICONS_DIR / "g_l_sep.png",
    "l_l_separator": ICONS_DIR / "l_l_sep.png",
}

SEPARATOR_TYPES = {
    "separator",
    "phase_separator",
    "gas_liquid_separator",
    "gas_liquid_separation",
    "liquid_liquid_separator",
    "liquid_liquid_extraction",
    "liq_liq_extraction",
    "lle",
}


def _ensure_graphviz_on_path() -> str | None:
    """Resolve Graphviz from PATH, Python, Conda, or a portable bundle."""

    # Graphviz/fontconfig may run under a service account without a writable
    # home cache. Give it a process-local cache so rendering remains reliable.
    cache_root = Path(
        os.environ.setdefault(
            "XDG_CACHE_HOME",
            str(Path(tempfile.gettempdir()) / "flowpilot-font-cache"),
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("dot")
    if executable:
        return executable

    environment_bin = Path(sys.executable).resolve().parent
    home = Path.home()
    candidates = [
        environment_bin / "dot",
        environment_bin / "dot.exe",
        environment_bin / "graphviz" / "bin" / "dot",
        environment_bin / "graphviz" / "bin" / "dot.exe",
        # FlowPilot may run from its own virtualenv while Graphviz is supplied
        # by a neighbouring Conda environment.
        home / "anaconda3" / "envs" / "flent" / "bin" / "dot",
        home / "miniconda3" / "envs" / "flent" / "bin" / "dot",
    ]
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parent.parent
        candidates.extend(
            [
                conda_root / "bin" / "dot",
                conda_root / "envs" / "flent" / "bin" / "dot",
                conda_root / "Library" / "bin" / "dot.exe",
            ]
        )
    candidate = next((path for path in candidates if path.is_file()), None)
    if candidate is None:
        return None

    graphviz_bin = str(candidate.parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if graphviz_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([graphviz_bin, *path_entries])
    return str(candidate)

# ── Graph-level style ─────────────────────────────────────────────────────────
GRAPH_ATTR = {
    "rankdir":     "LR",
    "splines":     "polyline",    # respects headport corners — clean nw/sw routing
    "nodesep":     "0.35",        # compact: less vertical gap between nodes
    "ranksep":     "0.6",         # compact: less horizontal gap between columns
    "pad":         "0.2",
    "dpi":         "180",
    "bgcolor":     "white",
    "forcelabels": "true",
    "fontname":    "Arial",
}

EDGE_ATTR = {
    "color":     "royalblue4",
    "penwidth":  "2.0",
    "arrowsize": "0.55",          # compact: slightly smaller arrow heads
    "fontname":  "Arial",
    "fontsize":  "9",
    "fontcolor": "royalblue4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _esc(s) -> str:
    return escape(str(s)) if s is not None else ""


def _html_lines(*lines: str, sizes: list[int] | None = None,
                colors: list[str] | None = None) -> str:
    """Build multi-line HTML label content as <BR/>-joined FONT tags."""
    parts = []
    for i, line in enumerate(lines):
        if not line:
            continue
        sz    = sizes[i] if sizes and i < len(sizes) else 10
        color = colors[i] if colors and i < len(colors) else "#111827"
        parts.append(
            f'<FONT POINT-SIZE="{sz}" COLOR="{color}">{_esc(line)}</FONT>'
        )
    return "<BR/>".join(parts)


def _wrap(s: str, n: int = 32) -> str:
    """Wrap captions without dropping chemical names or equipment qualifiers."""
    return "\n".join(line for paragraph in str(s).splitlines()
                     for line in textwrap.wrap(paragraph, width=n, break_long_words=True,
                                               break_on_hyphens=False))


def _font_lines(text, size=10, color="#111827", bold=False):
    return "<BR/>".join(
        f'<FONT POINT-SIZE="{size}" COLOR="{color}">'
        + (f"<B>{_esc(line)}</B>" if bold else _esc(line)) + "</FONT>"
        for line in _wrap(text).splitlines()
    )


# ── Gas stream detection ──────────────────────────────────────────────────────

_GAS_IDENTITIES = {
    "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
    "h2", "h₂", "hydrogen", "ar", "argon", "he", "helium", "air",
    "compressed air", "co", "carbon monoxide", "cl2", "chlorine",
    "nh3", "ammonia", "so2", "sulfur dioxide", "n2o", "nitrous oxide",
}
_GAS_PHASE_WORDS = {"gas", "gaseous", "vapor", "vapour", "mfc"}
_LIQUID_PHASE_WORDS = {
    "liquid", "solution", "solvent", "dissolved", "degassed", "sparged",
    "reagent", "substrate",
}

def _normalized_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9₂]+", " ", str(text).lower()).split()


def _has_explicit_gas_identity(text: str) -> bool:
    normalized = " " + " ".join(_normalized_words(text)) + " "
    for kw in _GAS_IDENTITIES:
        key = re.sub(r"[^a-z0-9₂]+", " ", kw.lower()).strip()
        if key and f" {key} " in normalized:
            return True
    return False


def _is_gas_phase_value(value) -> bool:
    words = set(_normalized_words(value))
    return bool(words & _GAS_PHASE_WORDS) and not bool(words & _LIQUID_PHASE_WORDS)


def _is_pure_gas_text(text: str) -> bool:
    words = set(_normalized_words(text))
    if not _has_explicit_gas_identity(text):
        return False
    if words & _LIQUID_PHASE_WORDS:
        return False
    return True


def _is_gas_contents(contents) -> bool:
    if isinstance(contents, str):
        contents = [contents]
    items = [str(c).strip() for c in (contents or []) if str(c).strip()]
    return bool(items) and all(_is_pure_gas_text(item) for item in items)


def _is_gas_pump(op) -> bool:
    """Return True if this pump carries a gas stream (use MFC node instead)."""
    p   = op.parameters or {}
    phase = p.get("phase") or p.get("state") or p.get("stream_type")
    if phase and _is_gas_phase_value(phase):
        return True
    if phase and set(_normalized_words(phase)) & _LIQUID_PHASE_WORDS:
        return False

    if p.get("solvent"):
        return False

    contents = p.get("contents") or p.get("components") or p.get("reagents") or []
    if _is_gas_contents(contents):
        return True

    role = p.get("role") or p.get("pump_role") or ""
    if _is_pure_gas_text(role):
        return True

    return not contents and _is_pure_gas_text(op.label or "")


def _enforce_mfc_for_gas_streams(ops) -> None:
    """Final GUI safety check: gas feeds must render as MFCs, not pumps."""
    for op in ops:
        if op.op_type != "pump" or not _is_gas_pump(op):
            continue

        op.op_type = "mfc"
        p = op.parameters or {}
        p["delivery_hardware"] = "MFC"
        op.parameters = p

        label = str(op.label or "").strip()
        if label:
            label = label.replace("Pump", "MFC", 1)
            op.label = label if label.lower().startswith("mfc") else f"MFC — {label}"
        else:
            op.label = "MFC"


# ── Label generators ──────────────────────────────────────────────────────────

def _unresolved_inventory_label(op) -> str:
    p = op.parameters or {}
    status = str(
        p.get("inventory_assignment_status")
        or getattr(op, "assignment_status", "")
    ).lower()
    if status != "unresolved":
        return ""
    category = str(
        p.get("unresolved_inventory_category")
        or getattr(op, "inventory_category", "equipment")
    ).replace("_", " ")
    return f"UNRESOLVED: {category}"

def _pump_label(op) -> str:
    """
    Clean pump label: Pump letter, materials, solvent/conc,
    then flow rate in blue on its own line for visibility.
    """
    p      = op.parameters or {}
    stream = p.get("stream", "?")
    instrument = p.get("instrument_name") or getattr(op, "instrument_name", "")
    title = f"Pump {stream}\n{instrument}" if instrument else f"Pump {stream}"

    # Materials line
    contents = p.get("contents") or []
    if isinstance(contents, str):
        contents = [contents]
    material_parts = []
    for item in contents:
        material_parts.append(str(item).strip())
    materials_line = "\n".join(material_parts)

    # Solvent · concentration line
    cond_parts = []
    if p.get("solvent"):
        cond_parts.append(str(p["solvent"]))
    if p.get("concentration_M") is not None:
        cond_parts.append(f"{float(p['concentration_M']):.2f} M")
    cond_line = "  ·  ".join(cond_parts)

    rows = [
        _font_lines(title, bold=True),
    ]
    if materials_line:
        rows.append(_font_lines(materials_line, 10, "#1E40AF"))
    if cond_line:
        rows.append(_font_lines(cond_line, 10, "#374151"))
    # Flow rate: separate blue line for visibility
    if p.get("flow_rate_mL_min") is not None:
        rows.append(
            f'<FONT POINT-SIZE="9" COLOR="#2563EB"><B>'
            f'{float(p["flow_rate_mL_min"]):.6g} mL/min</B></FONT>'
        )
    unresolved = _unresolved_inventory_label(op)
    if unresolved:
        rows.append(f'<FONT POINT-SIZE="8" COLOR="#B91C1C"><B>{_esc(unresolved)}</B></FONT>')
    return "<BR/>".join(rows)


def _mfc_label(op) -> str:
    """Label for gas MFC node."""
    p      = op.parameters or {}
    stream = p.get("stream", "?")
    instrument = p.get("instrument_name") or getattr(op, "instrument_name", "")
    contents = p.get("contents") or []
    if isinstance(contents, str):
        contents = [contents]
    gas_name = ", ".join(map(str, contents)) if contents else "Gas"
    sccm = p.get("gas_flow_sccm")
    rows = [
        _font_lines(f"Stream {stream}\n{instrument or 'MFC'}", bold=True),
        _font_lines(gas_name, 10, "#DC2626"),
    ]
    if sccm is not None:
        rows.append(_font_lines(f"{float(sccm):.6g} mL/min at STP", 10))
    else:
        rows.append(_font_lines("Inlet/STP flow not recorded", 10, "#B91C1C"))
    if p.get("molar_equiv") is not None:
        species = " O2" if gas_name.strip().lower() == "air" else ""
        rows.append(_font_lines(f"{float(p['molar_equiv']):.4g} equiv{species} (inlet/STP)", 10))
    unresolved = _unresolved_inventory_label(op)
    if unresolved:
        rows.append(f'<FONT POINT-SIZE="8" COLOR="#B91C1C"><B>{_esc(unresolved)}</B></FONT>')
    return "<BR/>".join(rows)


def _reactor_label(op) -> str:
    """Compact reactor label: temperature · wavelength · residence time only.
    No verbose description — the icon already communicates 'reactor'.
    """
    p = op.parameters or {}
    parts = []
    instrument = p.get("instrument_name") or getattr(op, "instrument_name", "")
    light_instrument = p.get("light_instrument_name") or ""
    temperature_controller = p.get("temperature_controller_name") or ""

    if p.get("temperature_C") is not None:
        parts.append(f"{p['temperature_C']}°C")
    if p.get("wavelength_nm") is not None:
        parts.append(f"λ={p['wavelength_nm']:.0f} nm")
    tau_inlet = p.get("residence_time_inlet_min")
    if tau_inlet is not None:
        basis = "liquid" if not (p.get("gas_flow_sccm") or p.get("Q_gas_sccm")) and "liquid" in str(p.get("residence_time_basis")) else "inlet/STP"
        parts.append(f"Time ({basis}): {float(tau_inlet):.6g} min")
    elif p.get("residence_time_min") is not None and not (p.get("gas_flow_sccm") or p.get("Q_gas_sccm")):
        rt = float(p["residence_time_min"])
        parts.append(f"τ={rt:.1f} min" if rt >= 1 else f"τ={rt*60:.0f} s")
    volume = p.get("volume_mL") if p.get("volume_mL") is not None else p.get("reactor_volume_mL")
    if volume is not None:
        parts.append(f"V={float(volume):.1f} mL")

    diameter = p.get("ID_mm") or p.get("tubing_ID_mm") or p.get("d_mm")
    if diameter is not None:
        parts.append(f"ID={float(diameter):g} mm")
    settings = "\n".join(parts)
    identity_lines = []
    if instrument:
        identity_lines.append(instrument)
    if light_instrument:
        identity_lines.append(f"Light: {light_instrument}")
    if temperature_controller:
        identity_lines.append(f"Temp: {temperature_controller}")
    if settings:
        identity_lines.append(settings)
    serial_status = p.get("serial_connection_status")
    if serial_status:
        identity_lines.append(str(serial_status))
    unresolved = _unresolved_inventory_label(op)
    if unresolved:
        identity_lines.append(unresolved)
    if identity_lines:
        return "\n".join(identity_lines)
    return settings


def _bpr_label(op) -> str:
    p = op.parameters or {}
    bar = p.get("pressure_bar") or p.get("BPR_bar")
    instrument = p.get("instrument_name") or getattr(op, "instrument_name", "") or "BPR"
    label = f"{instrument}\n{bar:g} bar" if bar else str(instrument)
    unresolved = _unresolved_inventory_label(op)
    return f"{label}\n{unresolved}" if unresolved else label


_SHORT_LABELS = {
    "liq_liq_extraction": "L-L Separator",
    "liquid_liquid_extraction": "L-L Separator",
    "lle": "L-L Separator",
    "separator": "Separator",
    "phase_separator": "Phase Sep.",
    "deoxygenation_unit": "Degasser",
    "degas": "Degasser",
    "inline_filter": "Filter",
    "filter": "Filter",
    "quench_mixer": "Quench",
}

def _textbox_label(op) -> str:
    p   = op.parameters or {}
    ot  = op.op_type.lower().replace(" ", "_")
    lbl = _SHORT_LABELS.get(ot) or _SHORT_LABELS.get(op.op_type.lower()) or \
          op.label or op.op_type.replace("_", " ").title()
    instrument = p.get("instrument_name") or getattr(op, "instrument_name", "")
    if instrument:
        lbl = f"{lbl}\n{instrument}"
    unresolved = _unresolved_inventory_label(op)
    if unresolved:
        lbl += f"\n{unresolved}"

    # Add one short detail if meaningful
    detail = p.get("method") or p.get("reagent") or ""
    if detail:
        return lbl + "\n" + str(detail)
    return lbl


def _separator_kind(op, adjacent_stream_types=()) -> str:
    """Classify a separator from explicit phase metadata before label hints."""
    p = op.parameters or {}
    op_type = str(op.op_type).strip().lower().replace("-", "_").replace(" ", "_")

    if op_type in {"gas_liquid_separator", "gas_liquid_separation"}:
        return "gas_liquid"
    if op_type in {
        "liquid_liquid_separator",
        "liquid_liquid_extraction",
        "liq_liq_extraction",
        "lle",
    }:
        return "liquid_liquid"

    phases = p.get("phases") or p.get("outlet_phases") or p.get("phase_system") or []
    if isinstance(phases, dict):
        phase_values = list(phases.keys()) + list(phases.values())
    elif isinstance(phases, (list, tuple, set)):
        phase_values = list(phases)
    else:
        phase_values = [phases]

    phase_words = [_normalized_words(value) for value in phase_values]
    flat_phase_words = [word for words in phase_words for word in words]
    if any(word in _GAS_PHASE_WORDS for word in flat_phase_words):
        return "gas_liquid"
    if sum("liquid" in words for words in phase_words) >= 2:
        return "liquid_liquid"

    context_values = [
        op.label,
        p.get("method"),
        p.get("separator_type"),
        p.get("instrument_name"),
        getattr(op, "instrument_name", ""),
        *adjacent_stream_types,
    ]
    context = " ".join(str(value) for value in context_values if value).lower()
    normalized_context = " ".join(_normalized_words(context))
    if (
        any(word in _GAS_PHASE_WORDS for word in _normalized_words(context))
        or "gas liquid" in normalized_context
    ):
        return "gas_liquid"
    if (
        "liquid liquid" in normalized_context
        or "aqueous organic" in normalized_context
        or "organic aqueous" in normalized_context
        or "biphasic liquid" in normalized_context
    ):
        return "liquid_liquid"

    # A generic downstream phase separator is conventionally a liquid-liquid
    # workup unless the topology declares a gas phase.
    return "liquid_liquid"


def _separator_asset(op, adjacent_stream_types=()) -> Path:
    kind = _separator_kind(op, adjacent_stream_types)
    return ASSETS["g_l_separator" if kind == "gas_liquid" else "l_l_separator"]


def _separator_label(op, adjacent_stream_types=()) -> str:
    p = op.parameters or {}
    kind = _separator_kind(op, adjacent_stream_types)
    fallback = "G-L Separator" if kind == "gas_liquid" else "L-L Separator"
    instrument = (
        p.get("instrument_name")
        or getattr(op, "instrument_name", "")
        or op.label
        or fallback
    )
    label = str(instrument)
    unresolved = _unresolved_inventory_label(op)
    return f"{label}\n{unresolved}" if unresolved else label


# ── Node builders ─────────────────────────────────────────────────────────────

NODE_ICON_SIZE = 110   # uniform icon size for all components


def _main_process_path(ops, streams):
    """Choose an existing feed-to-product path; never add or remove connections."""
    from graphlib import TopologicalSorter, CycleError
    nodes = {op.op_id: op for op in ops if op.op_type != "led_module"}
    predecessors = {key: [] for key in nodes}
    successors = {key: [] for key in nodes}
    for stream in streams:
        if stream.from_op in nodes and stream.to_op in nodes:
            predecessors[stream.to_op].append(stream.from_op)
            successors[stream.from_op].append(stream.to_op)
    try:
        order = list(TopologicalSorter(predecessors).static_order())
    except CycleError:
        return []  # Recycle graphs retain their graph layout, not a fictitious linear path.
    reactor_types = {"reactor", "coil_reactor", "photoreactor", "heated_coil", "packed_bed",
                     "packed_bed_reactor", "microchannel", "microreactor", "chip", "chip_reactor", "microfluidic"}
    paths = {}
    def score(path):
        return (sum(nodes[key].op_type in reactor_types for key in path), len(path))
    for key in order:
        previous = [paths[p] for p in sorted(predecessors[key]) if p in paths]
        if previous:
            paths[key] = max(previous, key=score) + [key]
        elif not predecessors[key] and nodes[key].op_type != "mfc":
            paths[key] = [key]
    destinations = [key for key in paths if nodes[key].op_type == "collector"]
    destinations = destinations or [key for key in paths if not successors[key]]
    return max((paths[key] for key in sorted(destinations)), key=score, default=[])


def _add_pump(dot, node_id: str, op, pump_img: Path):
    """Pump symbol with assigned instrument identity and complete feed label."""
    image_w   = NODE_ICON_SIZE
    image_h   = NODE_ICON_SIZE
    half_h    = image_h // 2
    port_h    = 2      # tiny port cell (graphviz attaches the edge here)

    label_html = _pump_label(op)
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{image_w}" CELLPADDING="1">{part}</TD></TR>'
        for part in label_html.split("<BR/>")
    )

    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        # Row 1: image (spans 3 rows) + space above needle
        "<TR>"
        f'<TD ROWSPAN="3" PORT="img" FIXEDSIZE="TRUE" WIDTH="{image_w}" HEIGHT="{image_h}">'
        f'<IMG SRC="{escape(pump_img.name)}" SCALE="TRUE"/></TD>'
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        # Row 2: needle port (right side, vertically centred)
        "<TR>"
        f'<TD PORT="needle" WIDTH="0" HEIGHT="{port_h}"></TD>'
        "</TR>"
        # Row 3: space below needle
        "<TR>"
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        # Label rows below the image
        f"{label_rows}"
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mixer(dot, node_id: str, label: str,
               w_inch: float = 0.9, h_inch: float = 0.85):
    """T-mixer box with in_top / in_bottom / out ports."""
    w_pt = int(w_inch * 72)
    h_pt = int(h_inch * 72)
    half = h_pt // 2
    port_w = 10

    html = (
        '<<TABLE BORDER="1" COLOR="grey40" CELLBORDER="0" CELLSPACING="0"'
        ' CELLPADDING="3" BGCOLOR="#F1F5F9">'
        "<TR>"
        f'<TD PORT="in_top" WIDTH="{port_w}" HEIGHT="{half}"></TD>'
        f'<TD ROWSPAN="2" FIXEDSIZE="TRUE" WIDTH="{w_pt}" HEIGHT="{h_pt}">'
        f'<FONT POINT-SIZE="10" COLOR="#1E293B"><B>{_esc(label)}</B></FONT></TD>'
        f'<TD PORT="out" ROWSPAN="2" WIDTH="{port_w}" HEIGHT="{h_pt}"></TD>'
        "</TR><TR>"
        f'<TD PORT="in_bottom" WIDTH="{port_w}" HEIGHT="{half}"></TD>'
        "</TR></TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_image_node(dot, node_id: str, label: str, img_path: Path,
                    img_w: int = 125, img_h: int = 125):
    """Image node (reactor / BPR / vial / mixer) with caption centered below."""
    # Each label line gets its own row, centered, with explicit width = image width
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{img_w}"><FONT POINT-SIZE="9" COLOR="#111827">'
        f'{_esc(ln)}</FONT></TD></TR>'
        for ln in _wrap(label).split("\n") if ln.strip()
    )
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
        f'<TR><TD PORT="img" FIXEDSIZE="TRUE" WIDTH="{img_w}" HEIGHT="{img_h}">'
        f'<IMG SRC="{escape(img_path.name)}" SCALE="TRUE"/></TD></TR>'
        f'{label_rows}'
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mixer_image(dot, node_id: str, n_inputs: int, op=None):
    """Mixer image node: mixer2 (T-mixer) for ≤2 inputs, mixer3 (cross) for 3+."""
    asset = ASSETS["mixer3"] if n_inputs >= 3 else ASSETS["mixer2"]
    sz = NODE_ICON_SIZE
    instrument = ""
    if op is not None:
        parameters = op.parameters or {}
        instrument = parameters.get("instrument_name") or getattr(op, "instrument_name", "")
    label_row = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{sz}"><FONT POINT-SIZE="9" COLOR="#111827">'
        f'{_esc(line)}</FONT></TD></TR>'
        for line in _wrap(instrument).splitlines()
    )
    unresolved = _unresolved_inventory_label(op) if op is not None else ""
    if unresolved:
        label_row += (
            f'<TR><TD ALIGN="CENTER" WIDTH="{sz}"><FONT POINT-SIZE="8" '
            f'COLOR="#B91C1C"><B>{_esc(unresolved)}</B></FONT></TD></TR>'
        )
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD PORT="img" FIXEDSIZE="TRUE" WIDTH="{sz}" HEIGHT="{sz}">'
        f'<IMG SRC="{escape(asset.name)}" SCALE="TRUE"/></TD></TR>'
        f'{label_row}'
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_textbox(dot, node_id: str, op):
    """Styled text box with explicit left/right ports for clean arrow routing."""
    label = _textbox_label(op)
    content_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="145"><FONT POINT-SIZE="9" '
        f'COLOR="#1E293B">{_esc(ln)}</FONT></TD></TR>'
        for ln in _wrap(label).split("\n")
    )
    # Outer table: [left-port cell | content cell | right-port cell]
    # Named ports ensure arrows attach exactly to the left/right border edge.
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        "<TR>"
        '<TD PORT="inp" WIDTH="1" HEIGHT="1"></TD>'
        '<TD>'
        '<TABLE BORDER="1" COLOR="#94A3B8" CELLBORDER="0" CELLSPACING="0"'
        ' CELLPADDING="8" BGCOLOR="#F8FAFC">'
        f'{content_rows}'
        '</TABLE>'
        '</TD>'
        '<TD PORT="out" WIDTH="1" HEIGHT="1"></TD>'
        "</TR>"
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mfc_node(dot, node_id: str, op):
    """Mass Flow Controller image node for gas streams."""
    image_w = NODE_ICON_SIZE
    image_h = NODE_ICON_SIZE
    half_h = image_h // 2
    port_h = 2

    label_html = _mfc_label(op)
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{image_w}" CELLPADDING="1">{part}</TD></TR>'
        for part in label_html.split("<BR/>")
    )
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        "<TR>"
        f'<TD ROWSPAN="3" PORT="img" FIXEDSIZE="TRUE" WIDTH="{image_w}" HEIGHT="{image_h}">'
        f'<IMG SRC="{escape(ASSETS["mfc"].name)}" SCALE="TRUE"/></TD>'
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        "<TR>"
        f'<TD PORT="needle" WIDTH="0" HEIGHT="{port_h}"></TD>'
        "</TR>"
        "<TR>"
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        f"{label_rows}"
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


# ── Pump image prep ───────────────────────────────────────────────────────────

def _prepare_pump_img() -> Path:
    """Render the neutral pump symbol; model identity belongs in the label."""
    import cairosvg
    generic = ICONS_DIR / "pump_generic.svg"
    rendered = ICONS_DIR / "_pump_generic.png"
    if not rendered.exists() or rendered.stat().st_mtime < generic.stat().st_mtime:
        cairosvg.svg2png(url=str(generic), write_to=str(rendered), output_width=480, output_height=480)
    return rendered


def _prepare_legacy_pump_img() -> Path:
    """Legacy image conversion retained for archived callers only."""
    src = ASSETS["pump"]
    out = ICONS_DIR / "_pump_trimmed.png"
    if out.exists():
        return out
    try:
        from PIL import Image
        img = Image.open(src).convert("RGBA")
        coords = [
            (x, y)
            for y in range(img.height) for x in range(img.width)
            if img.getpixel((x, y))[3] > 0
            and not all(c > 245 for c in img.getpixel((x, y))[:3])
        ]
        if not coords:
            return src
        margin = 4
        min_x = max(min(x for x, _ in coords) - margin, 0)
        max_x = min(max(x for x, _ in coords) + margin + 1, img.width)
        min_y = max(min(y for _, y in coords) - margin, 0)
        max_y = min(max(y for _, y in coords) + margin + 1, img.height)
        img.crop((min_x, min_y, max_x, max_y)).save(out)
        return out
    except Exception as e:
        logger.warning(f"Pump trim failed: {e} — using original")
        return src


# ── Main builder ──────────────────────────────────────────────────────────────

class FlowsheetBuilder:
    """
    Builds a publication-quality process flow diagram from a ProcessTopology.

    Usage (identical API to legacy builder):
        svg_path, png_path = FlowsheetBuilder().build(topology, title=...,
                                                       output_svg=..., output_png=...)
    """

    def build(
        self,
        topology,
        title: str = "",
        output_svg: str = "flora_process.svg",
        output_png: str = "flora_process.png",
    ) -> tuple[str, str]:
        """Build and save the diagram. Returns (svg_path, png_path)."""
        self.last_render_info = {"renderer": "graphviz", "warnings": []}
        # Graceful fallback to legacy builder if graphviz not available
        if not _ensure_graphviz_on_path():
            logger.warning("Graphviz 'dot' not found — falling back to legacy SVG builder")
            self.last_render_info["warnings"].append("Graphviz executable not available.")
            return self._legacy_fallback(topology, title, output_svg, output_png)

        try:
            return self._build_graphviz(topology, title, output_svg, output_png)
        except Exception as e:
            logger.error(f"Graphviz build failed: {e} — falling back to legacy builder", exc_info=True)
            self.last_render_info["warnings"].append(f"Graphviz renderer failed: {e}")
            return self._legacy_fallback(topology, title, output_svg, output_png)

    def _build_graphviz(self, topology, title, output_svg, output_png):
        import graphviz

        # Rendering is a pure projection. Some display helpers normalize node
        # types, so operate on a deep copy and never mutate the saved topology.
        topology = deepcopy(topology)
        ops    = self._clean_ops(topology)
        _enforce_mfc_for_gas_streams(ops)
        if not ops:
            return "", ""

        pump_img = _prepare_pump_img()

        active_ids = {o.op_id for o in ops}

        # ── Adjacency (filter streams to active ops only) ─────────────────────
        out_edges: dict[str, list[str]] = defaultdict(list)
        in_edges:  dict[str, list[str]] = defaultdict(list)
        incident_stream_types: dict[str, list[str]] = defaultdict(list)
        for s in topology.streams:
            if s.from_op in active_ids and s.to_op in active_ids:
                out_edges[s.from_op].append(s.to_op)
                in_edges[s.to_op].append(s.from_op)
                incident_stream_types[s.from_op].append(str(s.stream_type))
                incident_stream_types[s.to_op].append(str(s.stream_type))

        op_map   = {o.op_id: o for o in ops}
        pump_ids = {o.op_id for o in ops if o.op_type in ("pump", "mfc")}
        led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}

        # ── Graph init ────────────────────────────────────────────────────────
        stem = str(Path(output_png).with_suffix(""))
        dot  = graphviz.Digraph(comment=title or "FlowPilot Process", format="png")
        dot.attr(**GRAPH_ATTR)
        dot.attr(imagepath=str(ICONS_DIR))
        dot.attr("node",  fontname="Arial")
        dot.attr("edge", **EDGE_ATTR)

        if title:
            title_html = '<BR/>'.join(_esc(line) for line in _wrap(title, 80).splitlines())
            dot.attr(label=f'<<FONT POINT-SIZE="13" COLOR="#111827"><B>{title_html}</B></FONT>>',
                     labelloc="t", labeljust="c")

        # ── Build graphviz nodes ──────────────────────────────────────────────
        # node_id → graphviz node name (same as op_id, sanitised)
        gv_id = {op.op_id: op.op_id.replace("-", "_").replace(" ", "_")
                 for op in ops}

        # Pre-compute input counts for all mixers (needed to pick mixer2 vs mixer3)
        MIXER_TYPES_SET = {"mixer", "t_mixer", "y_mixer", "quench_mixer"}
        mixer_input_counts: dict[str, int] = {}
        for op in ops:
            if op.op_type in MIXER_TYPES_SET:
                all_inp = in_edges.get(op.op_id, [])
                # Exclude led_module from count
                real_inp = [i for i in all_inp if op_map.get(i) and
                            op_map[i].op_type != "led_module"]
                mixer_input_counts[op.op_id] = len(real_inp)

        for op in ops:
            if op.op_type == "led_module":
                continue  # skip — wavelength shown under reactor label

            vid = gv_id[op.op_id]

            if op.op_type == "mfc":
                _add_mfc_node(dot, vid, op)

            elif op.op_type == "pump":
                _add_pump(dot, vid, op, pump_img)

            elif op.op_type in MIXER_TYPES_SET:
                n_inp = mixer_input_counts.get(op.op_id, 2)
                _add_mixer_image(dot, vid, n_inp, op)

            elif op.op_type in ("coil_reactor", "reactor", "heated_coil"):
                is_photo = bool((op.parameters or {}).get("wavelength_nm"))
                img = ASSETS["photoreactor"] if is_photo else ASSETS["reactor"]
                _add_image_node(dot, vid, _reactor_label(op), img, NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "photoreactor":
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["photoreactor"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type in ("packed_bed", "packed_bed_reactor"):
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["reactor"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type in ("microchannel", "microreactor", "chip", "chip_reactor",
                                 "microfluidic"):
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["microchannel"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type in ("deoxygenation_unit", "degas", "degasser"):
                _add_image_node(dot, vid, _textbox_label(op),
                                ASSETS["degasser"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type.lower().replace("-", "_").replace(" ", "_") in SEPARATOR_TYPES:
                stream_types = incident_stream_types.get(op.op_id, [])
                _add_image_node(
                    dot,
                    vid,
                    _separator_label(op, stream_types),
                    _separator_asset(op, stream_types),
                    NODE_ICON_SIZE,
                    NODE_ICON_SIZE,
                )

            elif op.op_type == "bpr":
                _add_image_node(dot, vid, _bpr_label(op), ASSETS["bpr"],
                                NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "check_valve" and (ICONS_DIR / "check_valve.png").is_file():
                _add_image_node(dot, vid, _textbox_label(op), ICONS_DIR / "check_valve.png",
                                NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "collector":
                _add_image_node(dot, vid, op.label or "Product",
                                ASSETS["vial"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            else:
                _add_textbox(dot, vid, op)

        # ── Edge routing ──────────────────────────────────────────────────────
        # All op types that expose an :img port (image-based nodes)
        IMAGE_TYPES = {
            "coil_reactor", "reactor", "heated_coil", "photoreactor",
            "packed_bed", "packed_bed_reactor", "bpr", "collector",
            "microchannel", "microreactor", "chip", "chip_reactor",
            "deoxygenation_unit", "degas", "degasser",
            *SEPARATOR_TYPES,
            # Mixers are now image-based too
            "mixer", "t_mixer", "y_mixer", "quench_mixer",
        }
        if (ICONS_DIR / "check_valve.png").is_file():
            IMAGE_TYPES.add("check_valve")

        main_path = _main_process_path(ops, topology.streams)
        main_edges = set(zip(main_path, main_path[1:]))
        for key in main_path:
            dot.node(gv_id[key], group="main_process")
        self.last_render_info["main_process_path"] = main_path

        def output_port(op_id: str) -> str:
            """Graphviz port string for the OUTPUT (right side) of a node."""
            op = op_map.get(op_id)
            if op is None:
                return gv_id.get(op_id, op_id)
            vid = gv_id[op_id]
            ot = op.op_type.lower().replace("-", "_").replace(" ", "_")
            if ot in ("pump", "mfc"):
                return f"{vid}:needle:e"
            if ot == "led_module":
                return vid
            if ot in IMAGE_TYPES:
                return f"{vid}:img:e"
            # textbox: use named right port
            return f"{vid}:out:e"

        def input_target(op_id: str) -> str:
            """Base port string for connecting INTO a node."""
            op = op_map.get(op_id)
            if op is None:
                return gv_id.get(op_id, op_id)
            vid = gv_id[op_id]
            ot = op.op_type.lower().replace("-", "_").replace(" ", "_")
            if ot in IMAGE_TYPES:
                return f"{vid}:img"
            # textbox: use named left port
            return f"{vid}:inp:w"

        # Draw exactly the recorded material connections. The main process uses
        # center ports; additional feeds enter separately without synthetic mixers.
        branch_counts = defaultdict(int)
        for stream in topology.streams:
            source, target = stream.from_op, stream.to_op
            if source not in active_ids or target not in active_ids or source in led_ids or target in led_ids:
                continue
            is_main = (source, target) in main_edges
            port = "w"
            if not is_main and target in main_path and op_map[target].op_type in IMAGE_TYPES:
                port = "n" if branch_counts[target] % 2 == 0 else "s"
                branch_counts[target] += 1
            destination = (f"{gv_id[target]}:img:{port}"
                           if op_map[target].op_type in IMAGE_TYPES else input_target(target))
            dot.edge(output_port(source), destination, weight="1000" if is_main else "1")

        # ── Render ────────────────────────────────────────────────────────────
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

        # PNG
        dot.format = "png"
        rendered_png = dot.render(stem, cleanup=True)
        if not Path(rendered_png).exists() and Path(stem + ".png").exists():
            rendered_png = stem + ".png"
        if Path(rendered_png).exists() and rendered_png != output_png:
            Path(rendered_png).rename(output_png)
        logger.info(f"PNG saved: {output_png}")

        # SVG
        dot.format = "svg"
        rendered_svg = dot.render(stem, cleanup=True)
        if not Path(rendered_svg).exists() and Path(stem + ".svg").exists():
            rendered_svg = stem + ".svg"
        if Path(rendered_svg).exists() and rendered_svg != output_svg:
            Path(rendered_svg).rename(output_svg)
        logger.info(f"SVG saved: {output_svg}")

        return output_svg, output_png

    def _clean_ops(self, topology) -> list:
        """
        Remove logically redundant unit operations before rendering:
          - LED modules (shown as reactor icon + wavelength label instead)
          - Consecutive duplicate op types (e.g. two BPRs back to back)
        """
        ops     = list(topology.unit_operations)
        streams = topology.streams

        # Build sequential order from streams (main lane only)
        out_edges: dict[str, list[str]] = defaultdict(list)
        in_edges:  dict[str, list[str]] = defaultdict(list)
        for s in streams:
            out_edges[s.from_op].append(s.to_op)
            in_edges[s.to_op].append(s.from_op)

        op_map   = {o.op_id: o for o in ops}
        pump_ids = {o.op_id for o in ops if o.op_type in ("pump", "mfc")}
        led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}

        # Walk main lane (non-pump, non-led) and find consecutive duplicates
        auxiliary_ids = {
            o.op_id for o in ops
            if o.op_type in {"heater", "chiller", "heat_exchanger"}
            and not in_edges.get(o.op_id)
            and not out_edges.get(o.op_id)
        }
        remove_ids = set(led_ids) | auxiliary_ids

        DEDUP_TYPES = {"bpr", "inline_filter", "filter"}

        # Check every non-pump op: if its only non-pump predecessor has the same type → remove
        for op in ops:
            if op.op_id in remove_ids or op.op_type == "pump":
                continue
            preds = [
                op_map[pid] for pid in in_edges.get(op.op_id, [])
                if pid in op_map and pid not in pump_ids and pid not in remove_ids
            ]
            for pred in preds:
                if pred.op_type == op.op_type and op.op_type in DEDUP_TYPES:
                    remove_ids.add(op.op_id)
                    break

        if not remove_ids:
            return ops

        # Return ops with removed ones filtered out
        return [o for o in ops if o.op_id not in remove_ids]

    def _legacy_fallback(self, topology, title, output_svg, output_png):
        """Use the legacy hand-drawn SVG builder as fallback."""
        try:
            from flora_design.visualizer.flowsheet_builder_legacy import (
                FlowsheetBuilder as LegacyBuilder,
            )
            result = LegacyBuilder().build(topology, title, output_svg, output_png)
            self.last_render_info["renderer"] = "legacy_svg"
            return result
        except Exception as e:
            logger.error(f"Legacy builder also failed: {e}")
            self.last_render_info["renderer"] = "failed"
            self.last_render_info["warnings"].append(f"Legacy renderer failed: {e}")
            return "", ""
