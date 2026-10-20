"""Non-destructive styling and export helpers for benchmark figures."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deliverables/manuscript_benchmark_visualizations_20260825"
REVISED = PACKAGE / "figures_revised"
MAIN = REVISED / "main"
DEFAULT_EXPORT_ROOT = ROOT / "outputs/benchmark_figure_studio"


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    description: str
    svg_path: Path
    preview_path: Path
    semantic_colors: tuple[tuple[str, str, str], ...]
    raster_palette: str | None = None


FIGURE_SPECS: dict[str, FigureSpec] = {
    "figs5-1": FigureSpec(
        figure_id="figs5-1",
        title="Case-level benchmark scores",
        description="Sorted one-shot and FlowPilot scores for each chemistry.",
        svg_path=REVISED / "fig01c_case_panels_scores_sorted.svg",
        preview_path=MAIN / "figs5-1.png",
        semantic_colors=(
            ("one_shot", "One-shot", "#D46459"),
            ("flowpilot", "FlowPilot", "#167C80"),
        ),
    ),
    "figs5-2": FigureSpec(
        figure_id="figs5-2",
        title="Criterion-level gain heatmaps",
        description="FlowPilot-minus-one-shot gain separated by chemistry.",
        svg_path=REVISED / "fig07_criterion_gain_heatmap_revised.svg",
        preview_path=MAIN / "figs5-2.png",
        semantic_colors=(
            ("negative_gain", "Negative gain", "#B6403A"),
            ("zero_gain", "No change", "#F7F7F5"),
            ("positive_gain", "Positive gain", "#117A65"),
            ("not_available", "Not available", "#E4E7E9"),
        ),
        raster_palette="gain",
    ),
    "figs5-3": FigureSpec(
        figure_id="figs5-3",
        title="Campaign critical-error map",
        description="Campaign-level critical flags and judge agreement.",
        svg_path=REVISED / "fig08a_campaign_error_map.svg",
        preview_path=MAIN / "figs5-3.png",
        semantic_colors=(
            ("not_applicable", "Not applicable", "#D9DEE2"),
            ("no_error", "No critical error", "#EEF5F0"),
            ("one_judge", "One judge", "#F5C4BF"),
            ("two_judges", "Two judges", "#DF756B"),
            ("three_judges", "Three judges", "#A92F2B"),
        ),
        raster_palette="errors",
    ),
    "figs5-4": FigureSpec(
        figure_id="figs5-4",
        title="FlowPilot cost efficiency",
        description="Token use, cost, runtime, and chemistry-level value.",
        svg_path=REVISED / "fig18a_flowpilot_campaign_cost_efficiency.svg",
        preview_path=MAIN / "figs5-4.png",
        semantic_colors=(
            ("qwen36", "Qwen3.6", "#376AA0"),
            ("qwen38", "Qwen3.8", "#4C956C"),
            ("gpt4o", "GPT-4o", "#D58B32"),
            ("sonnet46", "Sonnet 4.6", "#8C61A8"),
            ("opus46", "Opus 4.6", "#C55467"),
        ),
    ),
}


COMMON_DEFAULTS = {
    "canvas": "#FFFFFF",
    "primary_text": "#111111",
    "secondary_text": "#59636B",
    "guides": "#AAB2B8",
}

PRESETS = {
    "Original": {},
    "Colorblind safe": {
        "one_shot": "#D55E00",
        "flowpilot": "#0072B2",
        "negative_gain": "#D55E00",
        "zero_gain": "#F5F5F3",
        "positive_gain": "#009E73",
        "not_available": "#D7DCE0",
        "not_applicable": "#D7DCE0",
        "no_error": "#EAF4EF",
        "one_judge": "#F6D6B8",
        "two_judges": "#E69F00",
        "three_judges": "#A44200",
        "qwen36": "#0072B2",
        "qwen38": "#009E73",
        "gpt4o": "#E69F00",
        "sonnet46": "#CC79A7",
        "opus46": "#D55E00",
    },
    "Print grayscale": {
        "canvas": "#FFFFFF",
        "primary_text": "#111111",
        "secondary_text": "#555555",
        "guides": "#A0A0A0",
        "one_shot": "#777777",
        "flowpilot": "#111111",
        "negative_gain": "#444444",
        "zero_gain": "#F4F4F4",
        "positive_gain": "#A8A8A8",
        "not_available": "#D8D8D8",
        "not_applicable": "#D8D8D8",
        "no_error": "#F3F3F3",
        "one_judge": "#C7C7C7",
        "two_judges": "#777777",
        "three_judges": "#222222",
        "qwen36": "#D0D0D0",
        "qwen38": "#A8A8A8",
        "gpt4o": "#808080",
        "sonnet46": "#555555",
        "opus46": "#202020",
    },
}


FONT_SIZE_RE = re.compile(r"font-size:\s*([0-9]+(?:\.[0-9]+)?)px")
STROKE_WIDTH_RE = re.compile(r"stroke-width:\s*([0-9]+(?:\.[0-9]+)?)")
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")
PNG_DATA_RE = re.compile(
    r'(xlink:href="data:image/png;base64,\s*)([^\"]+)(\")',
    flags=re.DOTALL,
)


def figure_spec(figure_id: str) -> FigureSpec:
    try:
        spec = FIGURE_SPECS[figure_id]
    except KeyError as exc:
        raise ValueError(f"Unknown figure: {figure_id}") from exc
    if not spec.svg_path.is_file():
        raise FileNotFoundError(spec.svg_path)
    return spec


def palette_defaults(figure_id: str, preset: str = "Original") -> dict[str, str]:
    spec = figure_spec(figure_id)
    if preset not in PRESETS:
        raise ValueError(f"Unknown palette preset: {preset}")
    colors = dict(COMMON_DEFAULTS)
    colors.update({key: default for key, _, default in spec.semantic_colors})
    colors.update({key: value for key, value in PRESETS[preset].items() if key in colors})
    return colors


def detected_vector_colors(figure_id: str) -> list[str]:
    """Return vector colors available for optional exact overrides."""
    text = figure_spec(figure_id).svg_path.read_text(encoding="utf-8")
    return sorted({value.upper() for value in HEX_RE.findall(text)})


def _hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {value!r}")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)])


def _replace_svg_colors(svg: str, replacements: Mapping[str, str]) -> str:
    normalized = {
        source.lower(): target.upper() for source, target in replacements.items()
    }
    return HEX_RE.sub(lambda match: normalized.get(match.group(0).lower(), match.group(0)), svg)


def _replace_canvas(svg: str, canvas: str) -> str:
    # Matplotlib draws canvas and axes backgrounds as paths. Text that is
    # intentionally white is not touched by this path-specific replacement.
    return re.sub(
        r'(<path\b[\s\S]*?style="[^"]*?fill:\s*)#ffffff(?=[;\"])([^>]*?/>)',
        lambda match: match.group(1) + canvas.upper() + match.group(2),
        svg,
        flags=re.IGNORECASE,
    )


def _scale_svg_metrics(svg: str, font_scale: float, line_scale: float) -> str:
    def font_replacement(match: re.Match[str]) -> str:
        return f"font-size: {float(match.group(1)) * font_scale:.3f}px"

    def line_replacement(match: re.Match[str]) -> str:
        return f"stroke-width: {float(match.group(1)) * line_scale:.3f}"

    svg = FONT_SIZE_RE.sub(font_replacement, svg)
    return STROKE_WIDTH_RE.sub(line_replacement, svg)


def _replace_exact_pixels(
    array: np.ndarray, replacements: Mapping[str, str], tolerance: float = 2.0
) -> np.ndarray:
    output = array.copy()
    rgb = array[..., :3].astype(float)
    for source, target in replacements.items():
        source_rgb = _hex_rgb(source).astype(float)
        mask = np.linalg.norm(rgb - source_rgb, axis=-1) <= tolerance
        output[mask, :3] = _hex_rgb(target)
    return output


def _project_segment(
    pixels: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    position = np.clip(((pixels - start) @ vector) / denominator, 0.0, 1.0)
    projected = start + position[:, None] * vector
    distance = np.linalg.norm(pixels - projected, axis=1)
    return position, distance


def _recolor_gain_raster(array: np.ndarray, palette: Mapping[str, str]) -> np.ndarray:
    output = array.copy()
    flat = array[..., :3].reshape(-1, 3).astype(float)
    old_negative = _hex_rgb("#B6403A").astype(float)
    old_zero = _hex_rgb("#F7F7F5").astype(float)
    old_positive = _hex_rgb("#117A65").astype(float)
    new_negative = _hex_rgb(palette["negative_gain"]).astype(float)
    new_zero = _hex_rgb(palette["zero_gain"]).astype(float)
    new_positive = _hex_rgb(palette["positive_gain"]).astype(float)

    negative_t, negative_distance = _project_segment(flat, old_negative, old_zero)
    positive_t, positive_distance = _project_segment(flat, old_zero, old_positive)
    use_negative = negative_distance <= positive_distance
    best_distance = np.minimum(negative_distance, positive_distance)
    mapped = flat.copy()
    mapped[use_negative] = (
        new_negative
        + negative_t[use_negative, None] * (new_zero - new_negative)
    )
    mapped[~use_negative] = (
        new_zero
        + positive_t[~use_negative, None] * (new_positive - new_zero)
    )
    colormap_mask = best_distance <= 7.0
    mapped[~colormap_mask] = flat[~colormap_mask]

    unavailable = _hex_rgb("#E4E7E9").astype(float)
    unavailable_mask = np.linalg.norm(flat - unavailable, axis=1) <= 3.0
    mapped[unavailable_mask] = _hex_rgb(palette["not_available"])
    output[..., :3] = np.clip(mapped.reshape(array.shape[:2] + (3,)), 0, 255).astype(
        np.uint8
    )
    return output


def _transform_embedded_pngs(
    svg: str, spec: FigureSpec, palette: Mapping[str, str]
) -> str:
    if not spec.raster_palette:
        return svg

    def replacement(match: re.Match[str]) -> str:
        encoded = re.sub(r"\s+", "", match.group(2))
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
        array = np.asarray(image).copy()
        if spec.raster_palette == "gain":
            array = _recolor_gain_raster(array, palette)
        elif spec.raster_palette == "errors":
            array = _replace_exact_pixels(
                array,
                {
                    "#D9DEE2": palette["not_applicable"],
                    "#EEF5F0": palette["no_error"],
                    "#F5C4BF": palette["one_judge"],
                    "#DF756B": palette["two_judges"],
                    "#A92F2B": palette["three_judges"],
                },
            )
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGBA").save(buffer, format="PNG", optimize=True)
        transformed = base64.b64encode(buffer.getvalue()).decode("ascii")
        return match.group(1) + transformed + match.group(3)

    return PNG_DATA_RE.sub(replacement, svg)


def transform_figure_svg(
    figure_id: str,
    *,
    palette: Mapping[str, str],
    font_scale: float = 1.0,
    line_scale: float = 1.0,
    advanced_overrides: Mapping[str, str] | None = None,
) -> bytes:
    """Create an edited SVG in memory without modifying its source file."""
    if not 0.5 <= font_scale <= 2.5:
        raise ValueError("font_scale must be between 0.5 and 2.5")
    if not 0.5 <= line_scale <= 3.0:
        raise ValueError("line_scale must be between 0.5 and 3.0")
    spec = figure_spec(figure_id)
    required = set(COMMON_DEFAULTS) | {key for key, _, _ in spec.semantic_colors}
    missing = required - set(palette)
    if missing:
        raise ValueError(f"Palette is missing required colors: {sorted(missing)}")

    svg = spec.svg_path.read_text(encoding="utf-8")
    svg = _transform_embedded_pngs(svg, spec, palette)
    semantic_sources = {
        default: palette[key] for key, _, default in spec.semantic_colors
    }
    common_sources = {
        "#000000": palette["primary_text"],
        "#111111": palette["primary_text"],
        "#20272C": palette["primary_text"],
        "#59636B": palette["secondary_text"],
        "#7A858D": palette["secondary_text"],
        "#B0B0B0": palette["guides"],
        "#AAB2B8": palette["guides"],
        "#D8DDE1": palette["guides"],
        "#7C878F": palette["guides"],
        "#4F5961": palette["guides"],
        "#5E6870": palette["guides"],
        "#4B555C": palette["guides"],
    }
    replacements = {**common_sources, **semantic_sources}
    if advanced_overrides:
        replacements.update(advanced_overrides)
    svg = _replace_svg_colors(svg, replacements)
    svg = _replace_canvas(svg, palette["canvas"])
    svg = _scale_svg_metrics(svg, font_scale, line_scale)
    return svg.encode("utf-8")


def render_svg_png(svg: bytes, width_px: int = 2400) -> bytes:
    """Rasterize SVG bytes using the local ImageMagick installation."""
    if not 800 <= width_px <= 8000:
        raise ValueError("width_px must be between 800 and 8000")
    executable = shutil.which("magick") or shutil.which("convert")
    if not executable:
        raise RuntimeError("ImageMagick is required for PNG preview and export")
    with tempfile.TemporaryDirectory(prefix="flowpilot-figure-studio-") as directory:
        source = Path(directory) / "figure.svg"
        destination = Path(directory) / "figure.png"
        source.write_bytes(svg)
        command = [
            executable,
            str(source),
            "-resize",
            f"{width_px}x",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if completed.returncode != 0 or not destination.is_file():
            raise RuntimeError(
                "PNG rendering failed: " + (completed.stderr.strip() or "unknown error")
            )
        return destination.read_bytes()


def save_figure_export(
    figure_id: str,
    svg: bytes,
    png: bytes,
    settings: Mapping[str, object],
    export_root: Path = DEFAULT_EXPORT_ROOT,
) -> Path:
    """Save one immutable studio export with its reproducibility settings."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = export_root / f"{timestamp}_{figure_id}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / f"{figure_id}.svg").write_bytes(svg)
    (destination / f"{figure_id}.png").write_bytes(png)
    (destination / "settings.json").write_text(
        json.dumps(dict(settings), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def list_exports(export_root: Path = DEFAULT_EXPORT_ROOT) -> list[Path]:
    if not export_root.is_dir():
        return []
    return sorted(
        (path for path in export_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
