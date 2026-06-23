"""
Metadata Hierarchy Explorer — TFM 2026
Navigation router (Streamlit st.navigation).

Sidebar layout:
    Hierarchy Explorer / TFM 2026      (branding, top)
    Demo View                          (pre-built results viewer)
    … the Demo View's own controls …      (Select Approach / Dataset, etc.)
    Build hierarchy  (collapsible)     (upload a CSV and run an app)
         • Baseline  • Approach 1  • Approach 2
"""
import streamlit as st

st.set_page_config(
    page_title="Metadata Hierarchy Explorer",
    layout="wide",
)

# ── Pages ────────────────────────────────────────────────────────────────────
viewer = st.Page("views/viewer.py",        title="Demo View", default=True)
base   = st.Page("views/run_baseline.py",  title="Baseline")
appr1  = st.Page("views/run_approach_1.py", title="Approach 1")
appr2  = st.Page("views/run_approach_2.py", title="Approach 2")

# Hidden default nav — we render our own links so we control the order.
pg = st.navigation([viewer, base, appr1, appr2], position="hidden")

# ── Sidebar TOP: branding + Demo View link ──────────────────────────────────
with st.sidebar:
    st.title("Hierarchy Explorer")
    st.caption("TFM 2026 — Metadata hierarchy construction")
    st.markdown("---")
    st.page_link(viewer, label="Demo View")

# ── The selected page renders here (its own sidebar controls included) ───────
pg.run()

# ── Sidebar BOTTOM: collapsible "Build hierarchy" group ─────────────────────
with st.sidebar:
    st.markdown("---")
    with st.expander("Build hierarchy", expanded=False):
        st.caption("Upload your own CSV and run an algorithm live.")
        st.page_link(base,  label="Baseline")
        st.page_link(appr1, label="Approach 1")
        st.page_link(appr2, label="Approach 2")
