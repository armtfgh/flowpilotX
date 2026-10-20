"""Interactive, non-destructive editor for manuscript benchmark figures."""

from __future__ import annotations

import json

import streamlit as st

_STUDIO_IMPORT_ERROR = None
try:
    from visualization.benchmark_figure_studio import (
        COMMON_DEFAULTS,
        FIGURE_SPECS,
        PRESETS,
        detected_vector_colors,
        figure_spec,
        list_exports,
        palette_defaults,
        render_svg_png,
        save_figure_export,
        transform_figure_svg,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"visualization", "visualization.benchmark_figure_studio"}:
        raise
    _STUDIO_IMPORT_ERROR = exc.name



SELECTED_KEY = "benchmark_figure_studio_selected"


@st.cache_data(show_spinner=False)
def _render_preview(
    figure_id: str,
    palette_items: tuple[tuple[str, str], ...],
    font_scale: float,
    line_scale: float,
    width_px: int,
    override_items: tuple[tuple[str, str], ...],
) -> tuple[bytes, bytes]:
    palette = dict(palette_items)
    overrides = dict(override_items)
    svg = transform_figure_svg(
        figure_id,
        palette=palette,
        font_scale=font_scale,
        line_scale=line_scale,
        advanced_overrides=overrides,
    )
    return svg, render_svg_png(svg, width_px=width_px)


def render() -> None:
    st.title("Benchmark Figure Studio")
    if _STUDIO_IMPORT_ERROR is not None:
        st.info(
            "Benchmark Figure Studio is unavailable in this installation: its optional "
            "visualization package was not included in the supplied source. "
            "Install that package and restart the app to enable the studio. "
            "FlowPilot topology rendering remains available."
        )
        return
    missing_assets = [
        spec.figure_id for spec in FIGURE_SPECS.values()
        if not spec.svg_path.is_file() or not spec.preview_path.is_file()
    ]
    if missing_assets:
        st.info(
            "Benchmark Figure Studio is installed, but its private benchmark figure "
            "assets are not present in this workspace. Add your authorized figure "
            "assets at the configured paths to enable the gallery and editor. "
            "Missing figure assets: " + ", ".join(missing_assets)
        )
        return
    st.caption("Manuscript originals remain unchanged. Studio exports are stored separately.")

    view = st.segmented_control(
        "View",
        ["Figure Gallery", "Edit Figure", "Saved Exports"],
        default="Figure Gallery",
        key="benchmark_figure_studio_view",
        label_visibility="collapsed",
    )
    if view == "Figure Gallery":
        _render_gallery()
    elif view == "Edit Figure":
        _render_editor()
    else:
        _render_exports()


def _open_editor(figure_id: str) -> None:
    st.session_state[SELECTED_KEY] = figure_id
    st.session_state["benchmark_figure_studio_view"] = "Edit Figure"


def _render_gallery() -> None:
    columns = st.columns(2)
    for index, spec in enumerate(FIGURE_SPECS.values()):
        with columns[index % 2]:
            with st.container(border=True):
                st.subheader(spec.figure_id)
                st.image(str(spec.preview_path), use_container_width=True)
                st.markdown(f"**{spec.title}**")
                st.button(
                    "Edit figure",
                    key=f"studio_gallery_edit_{spec.figure_id}",
                    use_container_width=True,
                    on_click=_open_editor,
                    args=(spec.figure_id,),
                )


def _initialize_palette(figure_id: str) -> None:
    defaults = palette_defaults(figure_id)
    for key, value in defaults.items():
        st.session_state.setdefault(f"studio_{figure_id}_{key}", value)


def _apply_preset(figure_id: str, preset: str) -> None:
    for key, value in palette_defaults(figure_id, preset).items():
        st.session_state[f"studio_{figure_id}_{key}"] = value


def _reset_style(figure_id: str) -> None:
    _apply_preset(figure_id, "Original")
    st.session_state[f"studio_{figure_id}_preset"] = "Original"
    st.session_state[f"studio_{figure_id}_font_scale_pct"] = 100
    st.session_state[f"studio_{figure_id}_line_scale_pct"] = 100
    st.session_state[f"studio_{figure_id}_width_px"] = 2600
    st.session_state[f"studio_{figure_id}_advanced_selected"] = []


def _render_editor() -> None:
    selected = st.session_state.get(SELECTED_KEY, next(iter(FIGURE_SPECS)))
    figure_ids = list(FIGURE_SPECS)
    selected_index = figure_ids.index(selected) if selected in figure_ids else 0
    figure_id = st.selectbox(
        "Figure",
        figure_ids,
        index=selected_index,
        format_func=lambda value: f"{value} | {FIGURE_SPECS[value].title}",
        key="studio_figure_selector",
    )
    st.session_state[SELECTED_KEY] = figure_id
    spec = figure_spec(figure_id)
    _initialize_palette(figure_id)

    controls, preview = st.columns([1.05, 2.35], gap="large")
    with controls:
        _render_controls(spec)
    with preview:
        _render_selected_preview(spec)


def _render_controls(spec) -> None:
    figure_id = spec.figure_id
    st.subheader("Style")
    preset = st.selectbox(
        "Palette preset",
        list(PRESETS),
        key=f"studio_{figure_id}_preset",
    )
    st.button(
        "Apply preset",
        key=f"studio_{figure_id}_apply_preset",
        use_container_width=True,
        on_click=_apply_preset,
        args=(figure_id, preset),
    )

    st.markdown("##### Typography")
    st.slider(
        "Font size",
        min_value=60,
        max_value=180,
        value=100,
        step=5,
        format="%d%%",
        key=f"studio_{figure_id}_font_scale_pct",
    )
    st.slider(
        "Line and border weight",
        min_value=60,
        max_value=200,
        value=100,
        step=5,
        format="%d%%",
        key=f"studio_{figure_id}_line_scale_pct",
    )
    st.slider(
        "Export width",
        min_value=1200,
        max_value=6000,
        value=2600,
        step=200,
        format="%d px",
        key=f"studio_{figure_id}_width_px",
    )

    st.markdown("##### Common colors")
    for key, label in [
        ("canvas", "Canvas"),
        ("primary_text", "Primary text and axes"),
        ("secondary_text", "Secondary text"),
        ("guides", "Grid lines and grouping guides"),
    ]:
        st.color_picker(
            label,
            key=f"studio_{figure_id}_{key}",
        )

    st.markdown("##### Data colors")
    for key, label, _ in spec.semantic_colors:
        st.color_picker(label, key=f"studio_{figure_id}_{key}")

    with st.expander("Advanced exact-color overrides", expanded=False):
        controlled = {
            default.upper() for _, _, default in spec.semantic_colors
        } | {
            "#000000",
            "#111111",
            "#20272C",
            "#59636B",
            "#7A858D",
            "#B0B0B0",
            "#AAB2B8",
            "#D8DDE1",
            "#7C878F",
            "#4F5961",
            "#5E6870",
            "#4B555C",
            "#FFFFFF",
        }
        available = [
            color for color in detected_vector_colors(figure_id) if color not in controlled
        ]
        selected = st.multiselect(
            "Source colors",
            available,
            key=f"studio_{figure_id}_advanced_selected",
        )
        for source in selected:
            st.color_picker(
                f"Replace {source}",
                value=source,
                key=f"studio_{figure_id}_advanced_{source}",
            )

    st.button(
        "Reset figure style",
        key=f"studio_{figure_id}_reset",
        use_container_width=True,
        on_click=_reset_style,
        args=(figure_id,),
    )


def _settings(spec) -> dict[str, object]:
    figure_id = spec.figure_id
    palette_keys = list(COMMON_DEFAULTS) + [key for key, _, _ in spec.semantic_colors]
    palette = {
        key: st.session_state[f"studio_{figure_id}_{key}"] for key in palette_keys
    }
    advanced = {
        source: st.session_state[f"studio_{figure_id}_advanced_{source}"]
        for source in st.session_state.get(
            f"studio_{figure_id}_advanced_selected", []
        )
    }
    return {
        "schema_version": "flowpilot_benchmark_figure_studio_v1.0",
        "figure_id": figure_id,
        "source_svg": str(spec.svg_path),
        "font_scale": st.session_state[f"studio_{figure_id}_font_scale_pct"] / 100,
        "line_scale": st.session_state[f"studio_{figure_id}_line_scale_pct"] / 100,
        "width_px": st.session_state[f"studio_{figure_id}_width_px"],
        "palette": palette,
        "advanced_overrides": advanced,
    }


def _render_selected_preview(spec) -> None:
    settings = _settings(spec)
    try:
        with st.spinner("Rendering figure..."):
            svg, png = _render_preview(
                spec.figure_id,
                tuple(sorted(settings["palette"].items())),
                settings["font_scale"],
                settings["line_scale"],
                settings["width_px"],
                tuple(sorted(settings["advanced_overrides"].items())),
            )
    except Exception as exc:
        st.error(f"Figure rendering failed: {exc}")
        return

    st.subheader(spec.figure_id)
    st.image(png, use_container_width=True)
    st.caption(spec.description)

    download_svg, download_png, save = st.columns(3)
    download_svg.download_button(
        "Download SVG",
        data=svg,
        file_name=f"{spec.figure_id}_edited.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )
    download_png.download_button(
        "Download PNG",
        data=png,
        file_name=f"{spec.figure_id}_edited.png",
        mime="image/png",
        use_container_width=True,
    )
    if save.button(
        "Save export",
        type="primary",
        key=f"studio_save_{spec.figure_id}",
        use_container_width=True,
    ):
        destination = save_figure_export(spec.figure_id, svg, png, settings)
        st.success(f"Saved: {destination}")

    with st.expander("Reproducibility settings", expanded=False):
        st.code(json.dumps(settings, indent=2), language="json")


def _render_exports() -> None:
    exports = list_exports()
    if not exports:
        st.info("No Figure Studio exports have been saved yet.")
        return
    for destination in exports[:30]:
        settings_path = destination / "settings.json"
        settings = (
            json.loads(settings_path.read_text(encoding="utf-8"))
            if settings_path.is_file()
            else {}
        )
        figure_id = str(settings.get("figure_id") or destination.name.split("_")[-1])
        png_path = destination / f"{figure_id}.png"
        with st.expander(destination.name, expanded=False):
            if png_path.is_file():
                st.image(str(png_path), use_container_width=True)
            st.code(str(destination), language=None)
            if settings:
                st.json(settings)
