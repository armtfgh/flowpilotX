"""FLORA — Process flowsheet builder with graph-aware layout.

Supports both single-step and multi-step topologies.
Layout algorithm:
  1. Build adjacency graph from StreamConnection list
  2. Find the main flow lane (longest path from first non-pump to collector)
  3. Pumps feeding the FIRST node go on the left (stacked vertically)
  4. Pumps feeding MID-STREAM nodes appear ABOVE their injection mixer
  5. LED badges always float above their paired reactor
  6. All icons use P&ID-style engineering symbols
"""

import logging
import math
from collections import defaultdict
from pathlib import Path

from flora_translate.schemas import ProcessTopology

logger = logging.getLogger("flora.design.visualizer")

# ── Palette ────────────────────────────────────────────────────────────────
PAL = {
    "pump":               "#1d4ed8",
    "mfc":                "#dc2626",
    "mixer":              "#7c3aed",
    "deoxygenation_unit": "#059669",
    "coil_reactor":       "#d97706",
    "chip_reactor":       "#d97706",
    "photoreactor":       "#d97706",
    "reactor":            "#d97706",
    "heated_coil":        "#d97706",
    "packed_bed":         "#d97706",
    "packed_bed_reactor": "#d97706",
    "led_module":         "#ca8a04",
    "bpr":                "#dc2626",
    "inline_filter":      "#4b5563",
    "quench_mixer":       "#9333ea",
    "collector":          "#16a34a",
    "heat_exchanger":     "#ea580c",
}
DEF_COLOR  = "#64748b"
TEXT_DARK  = "#1e293b"
TEXT_MID   = "#475569"
TEXT_LIGHT = "#94a3b8"
LINE_COLOR = "#475569"
BG         = "#ffffff"

# ── Geometry ───────────────────────────────────────────────────────────────
ICON_W   = 72
ICON_H   = 60
LABEL_H  = 18
PARAM_H  = 15
MAX_PARAMS = 3
MAX_CHARS  = 22
HGAP     = 52   # horizontal gap between sequential nodes
PUMP_VGAP = 14  # vertical gap between stacked left-side pumps
MARGIN   = 28
SIDE_PUMP_ABOVE = 90   # how far above main lane mid-stream pumps appear


# ── SVG helpers ───────────────────────────────────────────────────────────

def _e(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def _t(s: str, n: int = MAX_CHARS) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[:n-1] + "…"

def _font(size=10, weight="normal", color=TEXT_DARK):
    return (f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}"')


# ── P&ID icon renderers ───────────────────────────────────────────────────

def _icon_pump(cx, cy, color):
    r = 24
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" '
            f'stroke="{color}" stroke-width="2.5"/>'
            f'<polygon points="{cx-9},{cy-15} {cx-9},{cy+15} {cx+16},{cy}" '
            f'fill="{color}"/>')

def _icon_mixer(cx, cy, color):
    h, s = 24, 22
    return (
        f'<line x1="{cx-h}" y1="{cy}" x2="{cx+h}" y2="{cy}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<line x1="{cx}" y1="{cy-s}" x2="{cx}" y2="{cy}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}"/>'
        f'<polygon points="{cx-h},{cy-5} {cx-h},{cy+5} {cx-13},{cy}" fill="{color}"/>'
        f'<polygon points="{cx},{cy-s} {cx-5},{cy-14} {cx+5},{cy-14}" fill="{color}"/>'
    )

def _icon_coil(cx, cy, color, w=66, h=50):
    x0, y0 = cx - w/2, cy - h/2
    pts = []
    for i in range(61):
        t = i / 60
        px = x0 + 7 + t * (w - 14)
        py = cy - 4 + 9 * math.sin(t * 4 * math.pi)
        pts.append(f"{px:.1f},{py:.1f}")
    d = "M " + " L ".join(pts)
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="8" '
        f'fill="#fffbeb" stroke="{color}" stroke-width="2.5"/>'
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linecap="round"/>'
    )


def _icon_photoreactor(cx, cy, color):
    """Coil reactor with compact irradiation marks."""
    coil = _icon_coil(cx, cy + 4, color, w=66, h=44)
    rays = "".join(
        f'<line x1="{cx + dx}" y1="{cy - 27}" x2="{cx + dx}" y2="{cy - 19}" '
        f'stroke="#ca8a04" stroke-width="2" stroke-linecap="round"/>'
        for dx in (-20, 0, 20)
    )
    return coil + rays

def _icon_led(cx, cy, color):
    r = 13
    rays = "".join(
        f'<line x1="{cx+(r+2)*math.cos(a*math.pi/4):.1f}" '
        f'y1="{cy+(r+2)*math.sin(a*math.pi/4):.1f}" '
        f'x2="{cx+(r+8)*math.cos(a*math.pi/4):.1f}" '
        f'y2="{cy+(r+8)*math.sin(a*math.pi/4):.1f}" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
        for a in range(8)
    )
    return (rays +
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fef9c3" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" '
            f'{_font(8,"700",color)}>hν</text>')

def _icon_bpr(cx, cy, color):
    s = 24
    return (
        f'<polygon points="{cx},{cy-s} {cx+s},{cy} {cx},{cy+s} {cx-s},{cy}" '
        f'fill="white" stroke="{color}" stroke-width="2.5"/>'
        f'<text x="{cx}" y="{cy+4}" text-anchor="middle" {_font(8,"700",color)}>P</text>'
    )

def _icon_degas(cx, cy, color):
    w, h = 42, 36
    x0, y0 = cx-w/2, cy-h/2
    arrows = "".join(
        f'<line x1="{bx}" y1="{cy+8}" x2="{bx}" y2="{cy-6}" '
        f'stroke="{color}" stroke-width="1.5"/>'
        f'<polygon points="{bx-4},{cy-4} {bx+4},{cy-4} {bx},{cy-10}" fill="{color}"/>'
        for bx in [cx-12, cx, cx+12]
    )
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="6" '
        f'fill="#ecfdf5" stroke="{color}" stroke-width="2"/>'
        f'<ellipse cx="{cx}" cy="{y0}" rx="{w/2}" ry="5" '
        f'fill="#ecfdf5" stroke="{color}" stroke-width="2"/>'
        + arrows
    )

def _icon_filter(cx, cy, color):
    w, h = 46, 34
    x0, y0 = cx-w/2, cy-h/2
    lines = "".join(
        f'<line x1="{max(x0,x0+i):.0f}" y1="{y0+max(0,-i):.0f}" '
        f'x2="{min(x0+w,x0+i+h):.0f}" y2="{y0+h-max(0,i+h-w):.0f}" '
        f'stroke="{color}" stroke-width="1" opacity="0.5"/>'
        for i in range(-int(h), int(w), 9)
        if min(x0+w, x0+i+h) > max(x0, x0+i)
    )
    return (
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="4" '
        f'fill="#f9fafb" stroke="{color}" stroke-width="2"/>'
        f'<clipPath id="cf{int(cx)}">'
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="4"/>'
        f'</clipPath>'
        f'<g clip-path="url(#cf{int(cx)})">{lines}</g>'
    )

def _icon_quench(cx, cy, color):
    h, s = 22, 20
    return (
        f'<line x1="{cx-h}" y1="{cy}" x2="{cx+h}" y2="{cy}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<line x1="{cx}" y1="{cy-s}" x2="{cx}" y2="{cy+5}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="7" fill="{color}"/>'
        f'<text x="{cx}" y="{cy+4}" text-anchor="middle" {_font(10,"700","white")}>+</text>'
    )

def _icon_collector(cx, cy, color):
    nw, bw, h = 13, 46, 48
    top = cy - h/2
    bot = cy + h/2
    nb = top + h*0.28
    d = (f"M {cx-nw/2} {top} L {cx+nw/2} {top} "
         f"L {cx+bw/2-1} {nb} Q {cx+bw/2+3} {nb+5} {cx+bw/2+3} {nb+10} "
         f"Q {cx+bw/2+3} {bot} {cx+bw/2-5} {bot} "
         f"L {cx-bw/2+5} {bot} Q {cx-bw/2-3} {bot} {cx-bw/2-3} {nb+10} "
         f"Q {cx-bw/2-3} {nb+5} {cx-bw/2+1} {nb} L {cx-nw/2} {top} Z")
    return (f'<path d="{d}" fill="#f0fdf4" stroke="{color}" stroke-width="2"/>'
            f'<line x1="{cx-nw/2-4}" y1="{top}" x2="{cx+nw/2+4}" y2="{top}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')

ICON_RENDERERS = {
    "pump": _icon_pump, "mfc": _icon_pump, "mixer": _icon_mixer,
    "t_mixer": _icon_mixer, "y_mixer": _icon_mixer,
    "deoxygenation_unit": _icon_degas, "degas": _icon_degas,
    "degasser": _icon_degas,
    "coil_reactor": _icon_coil, "chip_reactor": _icon_coil,
    "reactor": _icon_coil, "heated_coil": _icon_coil,
    "packed_bed": _icon_coil, "packed_bed_reactor": _icon_coil,
    "photoreactor": _icon_photoreactor,
    "led_module": _icon_led, "bpr": _icon_bpr,
    "inline_filter": _icon_filter, "filter": _icon_filter,
    "quench_mixer": _icon_quench, "collector": _icon_collector,
}

# ── Block renderer ────────────────────────────────────────────────────────

def _param_lines(op) -> list[str]:
    p = op.parameters
    ot = op.op_type
    lines = []
    if ot in ("pump", "mfc"):
        for c in (p.get("contents") or [])[:2]:
            lines.append(c)
        if p.get("solvent"): lines.append(f"in {p['solvent']}")
        if p.get("flow_rate_mL_min"): lines.append(f"{p['flow_rate_mL_min']} mL/min")
    elif ot in (
        "coil_reactor", "chip_reactor", "photoreactor", "reactor",
        "heated_coil", "packed_bed", "packed_bed_reactor",
    ):
        if p.get("material") and p.get("ID_mm"):
            lines.append(f"{p['material']} {p['ID_mm']}mm")
        if p.get("volume_mL"): lines.append(f"V={p['volume_mL']}mL")
        if p.get("temperature_C") is not None: lines.append(f"T={p['temperature_C']}°C")
        if p.get("wavelength_nm"): lines.append(f"λ={p['wavelength_nm']}nm")
        if p.get("reactor_type"): lines.append(f"({p['reactor_type']})")
    elif ot == "bpr":
        if p.get("pressure_bar"): lines.append(f"{p['pressure_bar']} bar")
    elif ot in ("deoxygenation_unit", "degas", "degasser"):
        lines.append(p.get("method","N₂ sparging")[:20])
    elif ot in ("mixer", "t_mixer", "y_mixer", "quench_mixer"):
        if p.get("type"): lines.append(p["type"])
        if p.get("reagent"): lines.append(f"+{p['reagent'][:16]}")
        if p.get("details"): lines.append(p["details"][:18])
    elif ot in ("inline_filter", "filter"):
        if p.get("pore_size_um"): lines.append(f"{p['pore_size_um']}μm")
        if p.get("details"): lines.append(p.get("details","")[:18])
    elif ot == "led_module":
        if p.get("wavelength_nm"): lines.append(f"{p['wavelength_nm']}nm")
    elif ot == "collector":
        lines.append("Product")
    return lines[:MAX_PARAMS]

def _block_h(op) -> float:
    return ICON_H + LABEL_H + max(len(_param_lines(op)),1)*PARAM_H + 8

def _render_block(op, cx, top_y) -> str:
    color = PAL.get(op.op_type, DEF_COLOR)
    icon_cy = top_y + ICON_H/2
    renderer = ICON_RENDERERS.get(op.op_type)
    try:
        icon_svg = renderer(cx, icon_cy, color) if renderer else ""
    except Exception:
        icon_svg = ""

    label_y = top_y + ICON_H + LABEL_H - 2
    label_svg = (f'<text x="{cx}" y="{label_y}" text-anchor="middle" '
                 f'{_font(10,"700",TEXT_DARK)}>{_e(_t(op.label,22))}</text>')
    params_svg = ""
    for i, line in enumerate(_param_lines(op)):
        py = label_y + PARAM_H*(i+1)
        params_svg += (f'<text x="{cx}" y="{py}" text-anchor="middle" '
                       f'{_font(9,"normal",TEXT_MID)}>{_e(_t(line,26))}</text>')
    return icon_svg + label_svg + params_svg

# ── Arrow helpers ─────────────────────────────────────────────────────────

def _arrow_h(x1, y, x2) -> str:
    return (f'<line x1="{x1}" y1="{y}" x2="{x2-6}" y2="{y}" '
            f'stroke="{LINE_COLOR}" stroke-width="1.8"/>'
            f'<polygon points="{x2-7},{y-4} {x2},{y} {x2-7},{y+4}" fill="{LINE_COLOR}"/>')

def _arrow_diag(x1, y1, x2, y2) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{LINE_COLOR}" stroke-width="1.5" stroke-dasharray="5,3"/>'
            f'<polygon points="{x2-5},{y2-3} {x2},{y2} {x2-5},{y2+3}" fill="{LINE_COLOR}"/>')

def _arrow_down(x, y1, y2) -> str:
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2-5}" '
            f'stroke="{LINE_COLOR}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<polygon points="{x-4},{y2-6} {x},{y2} {x+4},{y2-6}" fill="{LINE_COLOR}"/>')


# ── Main builder ──────────────────────────────────────────────────────────

class FlowsheetBuilder:
    """Build a graph-aware process flow diagram."""

    def build(self, topology: ProcessTopology, title: str = "",
              output_svg: str = "flora_process.svg",
              output_png: str = "flora_process.png") -> tuple[str, str]:

        ops = topology.unit_operations
        if not ops:
            return "", ""

        # ── Build adjacency from stream connections ────────────────────────
        # out_edges[op_id] → list of op_ids it flows into
        # in_edges[op_id]  → list of op_ids flowing into it
        out_edges = defaultdict(list)
        in_edges  = defaultdict(list)
        for sc in topology.streams:
            out_edges[sc.from_op].append(sc.to_op)
            in_edges[sc.to_op].append(sc.from_op)

        op_map = {o.op_id: o for o in ops}
        pump_ids = {o.op_id for o in ops if o.op_type in ("pump", "mfc")}
        led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}

        # ── Find main sequential lane ──────────────────────────────────────
        # Start from first non-pump, non-LED node that has no non-pump inputs
        # OR just trace from the pump outputs through the graph
        def find_main_lane():
            """Return ordered list of op_ids on the main flow path."""
            # Find the 'collector' as the end node
            end = next((o.op_id for o in ops if o.op_type == "collector"), None)
            if not end:
                # Fallback: last non-pump non-LED op
                main_ops = [o for o in ops if o.op_type not in ("pump", "mfc", "led_module")]
                return [o.op_id for o in main_ops]

            # Walk backward from collector using in_edges
            lane = [end]
            current = end
            for _ in range(len(ops)):
                parents = [p for p in in_edges[current]
                           if p not in pump_ids and p not in led_ids]
                if not parents:
                    break
                # Pick the parent that has the most upstream connections
                # (i.e., the main trunk, not a side branch)
                best = max(parents, key=lambda p: len(in_edges[p]))
                if best in lane:
                    break
                lane.insert(0, best)
                current = best
            return lane

        main_lane = find_main_lane()

        # For each main-lane node, which pumps feed directly into it?
        def pumps_feeding(node_id):
            return [p for p in in_edges[node_id] if p in pump_ids]

        # ── Classify pumps as "left pumps" (feed first main-lane node)
        # vs "mid pumps" (feed a later main-lane node)
        first_main = main_lane[0] if main_lane else None
        left_pump_ids = pumps_feeding(first_main) if first_main else list(pump_ids)
        mid_pump_map = {}   # mid_pump_id → main_lane_node_id it feeds
        for node_id in main_lane[1:]:
            for pid in pumps_feeding(node_id):
                mid_pump_map[pid] = node_id

        left_pumps = [op_map[p] for p in left_pump_ids if p in op_map]
        # Main lane ops (excluding LEDs)
        main_ops = [op_map[n] for n in main_lane if n in op_map]

        # ── Find reactor→LED pairings ──────────────────────────────────────
        # LED nodes connect to reactors via stream or are implied
        reactor_led = {}  # reactor_id → led_id
        for lid in led_ids:
            led_op = op_map.get(lid)
            if led_op:
                # Find the reactor in the same stage (same prefix)
                prefix = lid.rsplit("_", 1)[0]
                for n in main_lane:
                    if n.startswith(prefix) and op_map[n].op_type in (
                        "coil_reactor", "chip_reactor", "photoreactor", "reactor",
                    ):
                        reactor_led[n] = lid

        # ── Measure heights ────────────────────────────────────────────────
        def measure(op): return ICON_H + LABEL_H + max(len(_param_lines(op)),1)*PARAM_H + 8

        left_heights  = [measure(p) for p in left_pumps]
        main_heights  = [measure(o) for o in main_ops]
        max_main_h    = max(main_heights) if main_heights else 100
        total_left_h  = sum(left_heights) + PUMP_VGAP*max(len(left_pumps)-1,0)

        # How many mid-pump groups are there? (to reserve vertical space above lane)
        max_mid_pumps_at_node = max(
            (len([p for p in mid_pump_map if mid_pump_map[p] == n]) for n in main_lane),
            default=0
        )
        # Reserve the pump body as well as its connecting drop. Without this,
        # a mid-stage pump can overlap the diagram title.
        mid_zone_h = SIDE_PUMP_ABOVE + 60 if max_mid_pumps_at_node > 0 else 0

        # ── Canvas ─────────────────────────────────────────────────────────
        title_h = 36 if title else 0
        led_zone = 70 if reactor_led else 0
        pump_col_w = ICON_W + 20
        seq_w = len(main_ops) * (ICON_W + HGAP) - HGAP if main_ops else 0

        canvas_w = MARGIN*2 + pump_col_w + HGAP + seq_w
        canvas_h = (MARGIN*2 + title_h + led_zone + mid_zone_h
                    + max(total_left_h, max_main_h) + 50)

        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'width="{canvas_w}" height="{canvas_h}" '
                   f'viewBox="0 0 {canvas_w} {canvas_h}">')
        svg.append(f'<rect width="{canvas_w}" height="{canvas_h}" fill="{BG}"/>')

        if title:
            svg.append(f'<text x="{canvas_w/2}" y="{MARGIN+20}" text-anchor="middle" '
                       f'{_font(13,"700",TEXT_DARK)}>{_e(title[:90])}</text>')

        # ── Main lane Y position ───────────────────────────────────────────
        base_y = MARGIN + title_h + led_zone + mid_zone_h
        # Vertical centre of sequential icons
        seq_icon_cy = base_y + max_main_h/2 - (LABEL_H + MAX_PARAMS*PARAM_H)/2

        # ── Left pumps (stacked vertically) ───────────────────────────────
        pump_col_cx = MARGIN + pump_col_w/2
        pump_total_h = sum(left_heights) + PUMP_VGAP*max(len(left_pumps)-1,0)
        pump_start_y = seq_icon_cy - pump_total_h/2

        pump_out_ports = []
        py = pump_start_y
        for i, pump in enumerate(left_pumps):
            h = left_heights[i]
            svg.append(_render_block(pump, pump_col_cx, py))
            pump_out_ports.append((pump_col_cx + ICON_W/2, py + ICON_H/2))
            py += h + PUMP_VGAP

        # ── Sequential main lane nodes ─────────────────────────────────────
        seq_start_x = MARGIN + pump_col_w + HGAP + ICON_W/2
        node_positions = {}   # op_id → (cx, top_y)

        for i, op in enumerate(main_ops):
            h = main_heights[i]
            cx = seq_start_x + i*(ICON_W + HGAP)
            top = seq_icon_cy
            svg.append(_render_block(op, cx, top))
            node_positions[op.op_id] = (cx, top)

            # LED above reactor
            if op.op_id in reactor_led:
                led_id = reactor_led[op.op_id]
                led_op = op_map.get(led_id)
                if led_op:
                    led_cy = top - 35
                    svg.append(_icon_led(cx, led_cy, PAL["led_module"]))
                    wl = led_op.parameters.get("wavelength_nm","")
                    if wl:
                        svg.append(f'<text x="{cx}" y="{led_cy+28}" '
                                   f'text-anchor="middle" {_font(8,"normal",TEXT_MID)}>'
                                   f'{wl}nm</text>')
                    # Dashed line from LED down to reactor
                    svg.append(f'<line x1="{cx}" y1="{led_cy+20}" x2="{cx}" y2="{top}" '
                               f'stroke="{PAL["led_module"]}" stroke-width="1.5" '
                               f'stroke-dasharray="4,3"/>')

        # ── Mid-stream pumps (above their injection mixer) ─────────────────
        for pid, target_id in mid_pump_map.items():
            if pid not in op_map or target_id not in node_positions:
                continue
            pump_op = op_map[pid]
            target_cx, target_top = node_positions[target_id]
            # Place pump above the target node
            pump_h = measure(pump_op)
            pump_top = target_top - SIDE_PUMP_ABOVE - pump_h/2
            pump_cx  = target_cx
            svg.append(_render_block(pump_op, pump_cx, pump_top))
            # Arrow: straight down from pump to target mixer
            pump_bot = pump_top + pump_h
            target_top_edge = target_top
            svg.append(_arrow_down(pump_cx, pump_bot, target_top_edge))

        # ── Connect left pumps → first sequential node ─────────────────────
        # Connections enter equipment at icon center. Using total block height
        # here routed arrows through labels whenever parameter text was present.
        flow_lane_y = seq_icon_cy + ICON_H/2
        if main_ops and pump_out_ports:
            first_cx = seq_start_x
            manifold_x = pump_col_cx + ICON_W/2 + 12
            top_port = min(p[1] for p in pump_out_ports)
            bot_port = max(p[1] for p in pump_out_ports)
            if len(pump_out_ports) > 1:
                svg.append(f'<line x1="{manifold_x}" y1="{top_port}" '
                           f'x2="{manifold_x}" y2="{bot_port}" '
                           f'stroke="{LINE_COLOR}" stroke-width="1.8"/>')
            for px_r, py_m in pump_out_ports:
                svg.append(f'<line x1="{px_r}" y1="{py_m}" x2="{manifold_x}" y2="{py_m}" '
                           f'stroke="{LINE_COLOR}" stroke-width="1.5"/>')
            # Manifold to first node
            manifold_mid = (top_port + bot_port)/2
            if abs(manifold_mid - flow_lane_y) > 2:
                svg.append(f'<line x1="{manifold_x}" y1="{manifold_mid}" '
                           f'x2="{manifold_x}" y2="{flow_lane_y}" '
                           f'stroke="{LINE_COLOR}" stroke-width="1.5"/>')
            svg.append(_arrow_h(manifold_x, flow_lane_y, first_cx - ICON_W/2))

        # ── Connect sequential nodes ───────────────────────────────────────
        for i in range(len(main_ops)-1):
            cx_a, _ = node_positions[main_ops[i].op_id]
            cx_b, _ = node_positions[main_ops[i+1].op_id]
            x1, x2 = cx_a + ICON_W/2, cx_b - ICON_W/2
            # Check if next node has a mid-stream pump above it
            # If so, shift the arrow to avoid overlap
            svg.append(_arrow_h(x1, flow_lane_y, x2))

        # ── Footer metrics ─────────────────────────────────────────────────
        parts = []
        if topology.residence_time_min:
            parts.append(f"τ = {topology.residence_time_min} min")
        if topology.total_flow_rate_mL_min:
            parts.append(f"Q = {topology.total_flow_rate_mL_min} mL/min")
        if topology.reactor_volume_mL:
            parts.append(f"V = {topology.reactor_volume_mL} mL")
        if parts:
            svg.append(f'<text x="{canvas_w/2}" y="{canvas_h-14}" '
                       f'text-anchor="middle" {_font(10,"normal",TEXT_LIGHT)}>'
                       f'{" · ".join(parts)}</text>')

        svg.append("</svg>")
        svg_content = "\n".join(svg)

        Path(output_svg).parent.mkdir(parents=True, exist_ok=True)
        Path(output_svg).write_text(svg_content, encoding="utf-8")
        logger.info(f"SVG saved: {output_svg}")

        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg_content.encode(), write_to=output_png, scale=2.0)
            logger.info(f"PNG saved: {output_png}")
        except ImportError:
            logger.warning("cairosvg not installed — PNG skipped")
            output_png = ""
        except Exception as e:
            logger.warning(f"PNG failed: {e}")
            output_png = ""

        return output_svg, output_png
