"""FlowPilot Streamlit dashboard."""

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="FlowPilot",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load custom CSS
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Sidebar navigation
from components.sidebar import render_sidebar
page = render_sidebar()

# Route to pages
if page == "flora_design":
    from pages.flora_design_unified import render
    render()

elif page == "inventory":
    from pages.inventory_manager import render
    render()

elif page == "diagnose":
    from pages.diagnose import render
    render()

elif page == "optimize":
    from pages.optimize import render
    render()

elif page == "figure_studio":
    from pages.benchmark_figure_studio import render
    render()

elif page == "fundamentals":
    from pages.fundamentals import render
    render()

elif page == "scout":
    from pages.scout import render
    render()

elif page == "prism":
    from pages.prism import render
    render()

elif page == "corpus":
    from pages.corpus import render
    render()

else:
    from pages.flora_design_unified import render
    render()
