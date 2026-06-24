"""
Metadata Hierarchy Explorer — TFM 2026
Navigation router (Streamlit st.navigation).

Sidebar layout:
    Metadata Hierarchy Explorer / TFM 2026   (branding, top)
    Demo View                                (pre-built results viewer)
    Build hierarchy  (collapsible)           (upload a CSV and run a method)
         • the three methods (descriptive names from methods.py)
"""
import sys
from pathlib import Path

import streamlit as st

# Shared method names live in views/methods.py — make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "views"))
from methods import METHODS  # noqa: E402

st.set_page_config(
    page_title="Metadata Hierarchy Explorer",
    layout="wide",
)

# ── Pages ────────────────────────────────────────────────────────────────────
viewer = st.Page("views/viewer.py",         title="Demo View", default=True)
base   = st.Page("views/run_baseline.py",   title=METHODS["Baseline"]["title"])
appr1  = st.Page("views/run_approach_1.py", title=METHODS["Approach 1"]["title"])
appr2  = st.Page("views/run_approach_2.py", title=METHODS["Approach 2"]["title"])

# Hidden default nav — we render our own links so we control the order.
pg = st.navigation([viewer, base, appr1, appr2], position="hidden")

# ── Sidebar: branding + navigation (Built Hierarchy above Demo View) ─────────
with st.sidebar:
    st.title("Metadata Hierarchy Explorer")
    st.caption("TFM 2026 — Metadata hierarchy construction")
    st.markdown("---")
    with st.expander("Built Hierarchy", expanded=False):
        st.caption("Upload a CSV and run a method live.")
        st.page_link(base,  label=METHODS["Baseline"]["title"])
        st.page_link(appr1, label=METHODS["Approach 1"]["title"])
        st.page_link(appr2, label=METHODS["Approach 2"]["title"])
    st.page_link(viewer, label="Demo View")

# ── The selected page renders here (its own controls included) ───────────────
pg.run()
