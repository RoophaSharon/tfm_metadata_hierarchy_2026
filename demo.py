"""
Metadata Hierarchy Explorer — TFM 2026
Pre-built results viewer for Baseline, Approach 1, and Approach 2.

Rendering faithfully replicates each app's display pipeline:
  - Baseline    : raw tree, Greens, Sunburst + Treemap
  - Approach 1  : raw tree, Blues,  Sunburst + Treemap + Node-link + Facets
  - Approach 2  : compress one-child chains, Viridis, Sunburst + Treemap + Node-link

Level-of-Detail controls (depth, leaf labels, hidden nodes, compress chains)
match the controls in the individual apps.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metadata Hierarchy Explorer",
    page_icon="🌿",
    layout="wide",
)

ROOT = Path(__file__).parent / "outputs"

DEFAULT_DEPTH = 7

# ─────────────────────────────────────────────────────────────────────────────
# PRE-BUILT OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PREBUILT = {
    "Baseline": {
        "AI-MIND": {"hierarchy": ROOT / "baseline" / "ai-mind-variable-descriptions_in__baseline_hierarchy.json"},
        "HCP":     {"hierarchy": ROOT / "baseline" / "HCP_S1200_DataDictionary_Oct_30_2023_baseline_hierarchy.json"},
    },
    "Approach 1": {
        "AI-MIND": {
            "hierarchy": ROOT / "approach_1" / "ai-mind-variable-descriptions_in__approach1_hierarchy.json",
            "facets":    ROOT / "approach_1" / "ai-mind-variable-descriptions_in__approach1_facets.json",
        },
        "HCP": {
            "hierarchy": ROOT / "approach_1" / "keybert" / "HCP_S1200_DataDictionary_Oct_30_2023_approach1_hierarchy.json",
            "facets":    ROOT / "approach_1" / "keybert" / "HCP_S1200_DataDictionary_Oct_30_2023_approach1_facets.json",
        },
    },
    "Approach 2": {
        "AI-MIND": {"hierarchy": ROOT / "approach_2" / "ai-mind-variable-descriptions_in__approach2_lod.json"},
        "HCP":     {"hierarchy": ROOT / "approach_2" / "HCP_S1200_DataDictionary_Oct_30_2023_approach2_lod.json"},
    },
}

# Per-approach rendering config (matches each source app)
CONFIG = {
    "Baseline":   {"color": "Greens",  "compress": False, "node_link": False},
    "Approach 1": {"color": "Blues",   "compress": False, "node_link": True},
    "Approach 2": {"color": "Viridis", "compress": True,  "node_link": True},
}

APPROACH_DESC = {
    "Baseline": (
        "Pure clustering baseline — TF-IDF representation + recursive agglomerative "
        "(cosine) clustering, number of clusters chosen by silhouette. No external APIs, "
        "no neural embeddings. Node labels are the most discriminative terms per cluster."
    ),
    "Approach 1": (
        "Global embedding pipeline — SBERT + N×M concept-table alignment (Gonçalves 2019) "
        "+ HiExpan refinement (Shen et al. KDD 2018) + Castanet parallel facets. Optionally "
        "retrieves concept context from Wikidata / Wikipedia / WordNet / BioPortal."
    ),
    "Approach 2": (
        "Dataset-constrained multi-aspect hierarchy — group-anchored L1/L2 → phrase-slot "
        "mining → FASTopic semantic aspect discovery (Wu et al. NeurIPS 2024) → GMM/KMeans "
        "clustering → deterministic 5-stage label generation. Optional local-LLM refinement."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# TREE TRANSFORMS  (copied from approach_2.py — display-only, exact behaviour)
# ─────────────────────────────────────────────────────────────────────────────
def _filter_dissolved(nodes: list) -> list:
    drop_ids = {int(n["id"]) for n in nodes
                if n.get("type") == "dissolved" or n.get("isShown") is False}
    if not drop_ids:
        return nodes
    out = []
    for n in nodes:
        if int(n["id"]) in drop_ids:
            continue
        m = dict(n)
        m["related"] = [int(c) for c in n.get("related", []) if int(c) not in drop_ids]
        out.append(m)
    return out

def compress_one_child_chains(nodes: list) -> list:
    """Collapse chains where an aggregation node has exactly one aggregation child
    (e.g. 'DMS → DMS Recommended Standard' becomes 'DMS / DMS Recommended Standard')."""
    nodes = _filter_dissolved(nodes)
    nm = {int(n["id"]): dict(n) for n in nodes}

    def _is_chain_link(n):
        if n.get("type") != "aggregation":
            return False
        children = n.get("related", [])
        return (len(children) == 1
                and nm.get(int(children[0]), {}).get("type") == "aggregation")

    changed = True
    while changed:
        changed = False
        for nid, n in list(nm.items()):
            if _is_chain_link(n):
                child_id = int(n["related"][0])
                child = nm[child_id]
                new_node = dict(child)
                new_node["id"] = nid
                new_node["name"] = f"{n['name']} / {child['name']}"
                new_node["desc"] = f"{n.get('desc', '')} | {child.get('desc', '')}"
                nm[nid] = new_node
                if child_id in nm:
                    del nm[child_id]
                for other in nm.values():
                    other["related"] = [nid if int(c) == child_id else int(c)
                                        for c in other.get("related", [])]
                changed = True
                break
    return list(nm.values())

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS  (DAG-safe value map — copied from approach_2.py)
# ─────────────────────────────────────────────────────────────────────────────
def _leaf_ids(nodes: list, nid: int) -> list:
    m = {int(n["id"]): n for n in nodes}
    out = []
    def rec(x):
        n = m.get(int(x))
        if not n:
            return
        if n.get("type") == "attribute":
            out.append(int(x)); return
        for c in n.get("related", []):
            rec(int(c))
    rec(nid)
    return list(dict.fromkeys(out))

def _parent_map(nodes: list) -> dict:
    pm = {}
    for n in nodes:
        for c in n.get("related", []):
            if int(c) not in pm:
                pm[int(c)] = int(n["id"])
    return pm

def _tree_value_map(nodes: list, pm: dict) -> dict:
    kids = {}
    for child, par in pm.items():
        kids.setdefault(int(par), []).append(int(child))
    nodemap = {int(n["id"]): n for n in nodes}
    memo = {}
    def count(nid: int) -> int:
        if nid in memo:
            return memo[nid]
        memo[nid] = 1
        n = nodemap.get(nid)
        if n is not None and n.get("type") == "attribute":
            memo[nid] = 1
            return 1
        ch = kids.get(nid, [])
        v = sum(count(c) for c in ch) if ch else 1
        memo[nid] = max(1, v)
        return memo[nid]
    return {nid: count(nid) for nid in nodemap}

def _wrap_hover(text: str, width: int = 80) -> str:
    import textwrap as _tw
    s = str(text or "")
    if not s:
        return ""
    lines = []
    for raw_line in s.split("\n"):
        lines.extend(_tw.wrap(raw_line, width=width) or [""])
    return "<br>".join(lines)

def plot_sunburst(nodes: list, color: str, max_depth: int = DEFAULT_DEPTH):
    nodes = _filter_dissolved(nodes)
    pm = _parent_map(nodes)
    vm = _tree_value_map(nodes, pm)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n["id"])
        lc = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get("name", ""))[:40])
        parents.append("" if nid == 0 else str(pm.get(nid, 0)))
        values.append(vm.get(nid, 1))
        hover.append(f"<b>{n.get('name', '')}</b><br>Type: {n.get('type', '')}<br>"
                     f"Variables: {lc}<br><br>{_wrap_hover(n.get('desc', ''))}")
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", hovertext=hover, hoverinfo="text",
        maxdepth=max_depth, insidetextorientation="radial",
        marker=dict(colorscale=color, line=dict(width=1, color="white"))))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text="Click sector to drill down — click centre to go back",
                                 font=dict(size=13), x=0.5))
    return fig

def plot_treemap(nodes: list, color: str):
    nodes = _filter_dissolved(nodes)
    pm = _parent_map(nodes)
    vm = _tree_value_map(nodes, pm)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n["id"])
        lc = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get("name", ""))[:40])
        parents.append("" if nid == 0 else str(pm.get(nid, 0)))
        values.append(vm.get(nid, 1))
        hover.append(f"<b>{n.get('name', '')}</b><br>Variables: {lc}<br>"
                     f"{_wrap_hover(n.get('desc', ''))}")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", hovertext=hover, hoverinfo="text",
        textinfo="label+value",
        marker=dict(colorscale=color, line=dict(width=1, color="white"))))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# NODE-LINK TREE  (Reingold-Tilford layout — copied from approach_2.py)
# ─────────────────────────────────────────────────────────────────────────────
def _node_color(n: dict) -> str:
    t = n.get("type", "")
    if t == "root":      return "#c44e52"
    if t == "attribute": return "#4C72B0"
    if t == "collapsed": return "#bbbbbb"
    return "#8C8C8C"

def _display_graph(nodes: list, max_depth: int, show_hidden: bool):
    m = {int(n["id"]): n for n in nodes}
    dnodes: dict = {}
    edges: list = []
    counter = 10 ** 9

    def rec(nid, depth):
        nonlocal counter
        n = m.get(int(nid))
        if not n:
            return
        if not show_hidden and n.get("isShown") is False and depth > 0:
            return
        dnodes[int(nid)] = n
        if depth >= max_depth and n.get("related"):
            counter += 1
            cid = counter
            n_leaves = len(_leaf_ids(nodes, nid))
            dnodes[cid] = {"id": cid, "name": f"… {n_leaves} variables",
                           "type": "collapsed", "related": [],
                           "desc": f"Collapsed: {n.get('name')}"}
            edges.append((int(nid), cid))
            return
        for c in n.get("related", []):
            ch = m.get(int(c))
            if not ch:
                continue
            if not show_hidden and ch.get("isShown") is False:
                continue
            edges.append((int(nid), int(c)))
            rec(int(c), depth + 1)

    rec(0, 0)
    return list(dnodes.values()), edges

def _positions(edges: list):
    H_SCALE, V_SPACE = 3.0, 1.8
    children: dict = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
    pos: dict = {}
    counter = {"v": 0}

    def rec(nid, depth):
        ch = children.get(nid, [])
        if not ch:
            y_pos = counter["v"] * V_SPACE
            counter["v"] += 1
            pos[nid] = (depth * H_SCALE, y_pos)
            return y_pos
        child_ys = [rec(c, depth + 1) for c in ch]
        y_pos = float(np.mean(child_ys))
        pos[nid] = (depth * H_SCALE, y_pos)
        return y_pos

    rec(0, 0)
    return pos

def plot_node_link(nodes: list, max_depth: int, show_hidden: bool, show_leaf_labels: bool):
    nodes = _filter_dissolved(nodes)
    dnodes, edges = _display_graph(nodes, max_depth, show_hidden)
    pos = _positions(edges)

    ex, ey = [], []
    for p, c in edges:
        if p not in pos or c not in pos:
            continue
        x0, y0 = pos[p]
        x1, y1 = pos[c]
        xm = (x0 + x1) / 2
        ex += [x0, xm, xm, x1, None]
        ey += [y0, y0, y1, y1, None]
    traces = [go.Scatter(x=ex, y=ey, mode="lines",
                         line=dict(width=1, color="#c8c8c8"),
                         hoverinfo="skip", showlegend=False)]

    agg_x, agg_y, agg_lab, agg_col, agg_hov = [], [], [], [], []
    lf_x, lf_y, lf_lab, lf_col, lf_hov = [], [], [], [], []
    for n in dnodes:
        nid = int(n["id"])
        if nid not in pos:
            continue
        x, y = pos[nid]
        lc = len(_leaf_ids(nodes, nid))
        lab = str(n.get("name", ""))[:32]
        hov = (f"<b>{n.get('name', '')}</b><br>Type: {n.get('type', '')}<br>"
               f"Variables: {lc}")
        if n.get("type") == "attribute":
            lf_x.append(x); lf_y.append(y); lf_col.append(_node_color(n))
            lf_lab.append(lab if show_leaf_labels else "")
            lf_hov.append(hov)
        else:
            agg_x.append(x); agg_y.append(y); agg_col.append(_node_color(n))
            agg_lab.append(lab); agg_hov.append(hov)

    traces.append(go.Scatter(
        x=lf_x, y=lf_y, mode="markers+text" if show_leaf_labels else "markers",
        text=lf_lab, textposition="middle right", textfont=dict(size=9),
        marker=dict(size=7, color=lf_col, line=dict(width=0.5, color="white")),
        hovertext=lf_hov, hoverinfo="text", showlegend=False))
    traces.append(go.Scatter(
        x=agg_x, y=agg_y, mode="markers+text", text=agg_lab,
        textposition="middle right", textfont=dict(size=10),
        marker=dict(size=13, color=agg_col, line=dict(width=1, color="white")),
        hovertext=agg_hov, hoverinfo="text", showlegend=False))

    n_rows = max(len(lf_y), len(agg_y), 1)
    fig = go.Figure(traces)
    fig.update_layout(
        height=max(600, n_rows * 16),
        margin=dict(l=10, r=140, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_json(path_str: str):
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)

def count_nodes(nodes: list) -> tuple[int, int]:
    nodes = _filter_dissolved(nodes)
    leaves = sum(1 for n in nodes if n.get("type") == "attribute")
    aggs = sum(1 for n in nodes if n.get("type") == "aggregation")
    return leaves, aggs

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 Hierarchy Explorer")
    st.caption("TFM 2026 — Metadata hierarchy construction")
    st.markdown("---")

    approach = st.radio("**Select Approach**",
                        ["Baseline", "Approach 1", "Approach 2"], index=0)
    dataset = st.radio("**Select Dataset**", ["AI-MIND", "HCP"], index=0)

    st.markdown("---")
    st.caption("Results are pre-built from the thesis experiments. To run on your "
               "own data, clone the repository and run the individual apps.")
    st.markdown("[📦 GitHub Repository]"
                "(https://github.com/RoophaSharon/tfm_metadata_hierarchy_2026)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
cfg = CONFIG[approach]
color = cfg["color"]

st.title(f"📊 {approach} — {dataset} Dataset")
st.markdown(f"> {APPROACH_DESC[approach]}")

paths = PREBUILT[approach][dataset]
hier_path = paths.get("hierarchy")
if hier_path is None or not hier_path.exists():
    st.error(f"Pre-built result not found: `{hier_path}`")
    st.stop()

raw_nodes = _load_json(str(hier_path))

leaves, aggs = count_nodes(raw_nodes)
c1, c2, c3 = st.columns(3)
c1.metric("Leaf Variables", leaves)
c2.metric("Aggregation Nodes", aggs)
c3.metric("Total Nodes", leaves + aggs)
st.markdown("---")

# ── Level-of-Detail controls (above chart — matches the apps) ────────────────
view_options = ["Sunburst (drill-down)", "Treemap"]
if cfg["node_link"]:
    view_options.append("Node-link tree")

if cfg["compress"]:
    vc1, vc2, vc3, vc4, vc5 = st.columns([2.4, 2, 1, 1, 1.2])
else:
    vc1, vc2, vc3, vc4 = st.columns([2.4, 2, 1, 1])
    vc5 = None

with vc1:
    viz_mode = st.radio("View mode", view_options, horizontal=True, index=0,
                        help="Sunburst best for large hierarchies [Taxonomizer]. "
                             "Node-link best for moderate-depth structure inspection.")
with vc2:
    depth = st.slider("Depth (Level of Detail)", 1, 8, DEFAULT_DEPTH, 1)
with vc3:
    show_leaf_labels = st.checkbox("Leaf labels", value=False)
with vc4:
    show_hidden = st.checkbox("Hidden nodes", value=False)
if vc5 is not None:
    with vc5:
        compress_chains = st.checkbox("Compress chains", value=True,
                                      help="Merge one-child aggregation chains "
                                           '(e.g. "DMS → DMS Recommended Standard") for '
                                           "display. Export JSON keeps original structure.")
else:
    compress_chains = False

st.divider()

display_nodes = compress_one_child_chains(raw_nodes) if compress_chains else raw_nodes

if viz_mode == "Sunburst (drill-down)":
    st.plotly_chart(plot_sunburst(display_nodes, color, depth), use_container_width=True)
elif viz_mode == "Treemap":
    st.plotly_chart(plot_treemap(display_nodes, color), use_container_width=True)
else:
    st.plotly_chart(plot_node_link(display_nodes, depth, show_hidden, show_leaf_labels),
                    use_container_width=True)

# ── Facets (Approach 1 only) ─────────────────────────────────────────────────
facet_path = paths.get("facets")
if facet_path is not None and facet_path.exists():
    st.markdown("---")
    st.subheader("🔀 Parallel facets")
    facets = _load_json(str(facet_path))
    names = list(facets.keys())
    if not names:
        st.info("No facets available for this dataset.")
    else:
        sel = st.selectbox("Select facet", names)
        fnodes = facets[sel]
        ft1, ft2 = st.tabs(["Sunburst", "Treemap"])
        with ft1:
            st.plotly_chart(plot_sunburst(fnodes, color, depth), use_container_width=True)
        with ft2:
            st.plotly_chart(plot_treemap(fnodes, color), use_container_width=True)
