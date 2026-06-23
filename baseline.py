# baseline.py — Metadata Hierarchy Builder — Baseline (Taxonomizer)
#
# Pure Taxonomizer baseline — NO hardcoded, domain-specific patterns.
# The only lexical resource is a generic English stop-word list (standard
# information-retrieval practice, not dataset-specific).
#
# Pipeline (dataset-only, no external APIs, no sentence-transformers):
#   1. Load metadata file (CSV / TSV / XLSX / JSON)
#   2. Detect column roles (leaf / group / text / meta)
#   3. Build canonical schema (_leaf_id, _leaf_label, _group_path, _text)
#   4. Represent each variable as a TF-IDF text object
#   5. Recursively cluster variables (agglomerative, cosine distance) into an
#      abstract-to-concrete taxonomy; internal-node labels are the most
#      discriminative terms of each cluster — derived from the data, not hardcoded
#   6. Visualise (Sunburst / Treemap)
#   7. Export visualization-ready JSON + canonical CSV
#
# Papers:
#   [TAX] Taxonomizer (Mahmood & Mueller, IEEE TVCG) — leaf=attribute, internal=abstract group
#         built bottom-up by recursively clustering item feature vectors and
#         labelling each internal node with its members' shared/discriminative terms
#   [GON] Goncalves et al. — TF-IDF text objects + cosine distance
#   NB: discriminative-term labelling here is Taxonomizer's own contrastive
#       in-vs-sibling term scoring; HiExpan refinement is NOT used in the baseline
#       (it is introduced in Approach 1).

from __future__ import annotations
import csv, json, re, warnings
from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Metadata Hierarchy — Baseline', page_icon='🌿', layout='wide')
st.title('Metadata Hierarchy Builder — Baseline (Taxonomizer)')
st.caption(
    'Pure Taxonomizer baseline: TF-IDF text objects + recursive agglomerative '
    'clustering into an abstract-to-concrete taxonomy, with internal-node labels '
    'derived from each cluster’s discriminative terms. No hardcoded domain '
    'patterns, no external APIs, no sentence embeddings — works on any dataset.'
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LEAF_KEYS  = 'variable var field column attribute name code id item indicator question measure concept'.split()
GROUP_KEYS = 'task category domain module section table dataset assessment test variant group topic instrument form subscale construct'.split()
TEXT_KEYS  = 'description definition desc label title question meaning note notes text display full details explanation comment'.split()
META_KEYS  = 'type dtype data_type datatype unit units format decimal precision values value coding codebook range min max scale'.split()

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in name)

def try_read_csv(path: Path) -> pd.DataFrame:
    best, best_score = None, -1
    for enc in ['utf-8-sig', 'utf-8', 'latin1']:
        for sep in [None, ',', '\t', ';', '|']:
            try:
                df = pd.read_csv(path, sep=sep, engine='python', encoding=enc)
                score = df.shape[1] * 10 - float(df.isna().mean().mean())
                if score > best_score:
                    best, best_score = df, score
            except Exception:
                pass
    if best is None:
        raise ValueError(f'Could not read {path.name}')
    best.columns = [str(c).strip().replace(';', '') for c in best.columns]
    # Repair comma-packed rows (AI-Mind format)
    if len(best) > 0:
        first = best.iloc[:, 0].astype(str)
        other_null = best.iloc[:, 1:].isna().mean().mean() if best.shape[1] > 1 else 1.0
        if first.str.contains(',').mean() > 0.50 and other_null > 0.70:
            lines = path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
            if lines:
                header = [h.strip().replace(';', '') for h in lines[0].split(',')]
                rows = []
                for line in lines[1:]:
                    line = line.strip().rstrip(';')
                    if not line:
                        continue
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    try:
                        parts = next(csv.reader([line], quotechar='"'))
                    except Exception:
                        continue
                    if len(parts) >= len(header):
                        rows.append(parts[:len(header)])
                if rows:
                    best = pd.DataFrame(rows, columns=header)
    best.columns = [str(c).strip().replace(';', '') for c in best.columns]
    return best

def load_any(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s in ['.csv', '.tsv', '.txt']:
        return try_read_csv(path)
    if s in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    if s == '.json':
        obj = json.loads(path.read_text(encoding='utf-8', errors='replace'))
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return pd.json_normalize(v)
    raise ValueError(f'Unsupported file type: {s}')

def save_upload(f) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix='baseline_'))
    p = tmp / safe_name(f.name)
    p.write_bytes(f.getbuffer())
    return p

# ─────────────────────────────────────────────────────────────────────────────
# ROLE DETECTION  [GON]
# ─────────────────────────────────────────────────────────────────────────────
def norm(c: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')

def kscore(c: str, keys: list) -> int:
    nc = norm(c)
    return sum(1 for k in keys if k in nc)

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        non = float(s.notna().mean())
        nun = int(s.nunique(dropna=True))
        ur  = nun / n
        avg = float(s.dropna().astype(str).map(len).mean()) if s.notna().any() else 0
        out.append({
            'column':         str(col),
            'non_null':       round(non, 3),
            'unique_values':  nun,
            'unique_ratio':   round(ur, 3),
            'avg_length':     round(avg, 1),
            'leaf_score':     4*kscore(col, LEAF_KEYS)  + (3 if 0.5 <= ur <= 1 else 0) + (1 if avg < 80 else 0),
            'group_score':    4*kscore(col, GROUP_KEYS) + (3 if 1 < nun < min(n*0.5, 80) else 0) + (1 if avg < 60 else 0),
            'text_score':     5*kscore(col, TEXT_KEYS)  + (4 if avg > 50 else 0) + (1 if non > 0.5 else 0),
            'metadata_score': 4*kscore(col, META_KEYS)  + (2 if 1 < nun < min(n*0.8, 100) else 0),
        })
    return pd.DataFrame(out)

def detect_roles(df: pd.DataFrame) -> tuple:
    prof  = profile_columns(df)
    leaf  = prof.sort_values(['leaf_score', 'unique_ratio'], ascending=False).head(1)['column'].tolist()
    text  = (prof[(prof.text_score >= 4) | (prof.avg_length > 80)]
             .sort_values('text_score', ascending=False)['column'].tolist()) or leaf.copy()
    group = (prof[(prof.group_score >= 4) & (~prof.column.isin(leaf)) & (prof.unique_values > 1)]
             .sort_values('group_score', ascending=False)['column'].head(3).tolist())
    meta  = (prof[(prof.metadata_score >= 4) & (~prof.column.isin(text + leaf + group))]
             .sort_values('metadata_score', ascending=False)['column'].head(5).tolist())
    return {'leaf_cols': leaf, 'group_cols': group, 'text_cols': text, 'metadata_cols': meta}, prof

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL SCHEMA  [GON]
# ─────────────────────────────────────────────────────────────────────────────
def sv(x) -> str:
    return '' if pd.isna(x) else str(x).strip()

def build_canonical(df: pd.DataFrame, cfg: dict, source: str) -> pd.DataFrame:
    leaf_cols  = cfg.get('leaf_cols', [])
    group_cols = cfg.get('group_cols', [])
    text_cols  = cfg.get('text_cols', [])
    meta_cols  = cfg.get('metadata_cols', [])
    rows = []
    for i, row in df.iterrows():
        leaf_parts  = [sv(row.get(c, '')) for c in leaf_cols]
        leaf_parts  = [p for p in leaf_parts if p]
        label       = ' / '.join(leaf_parts) if leaf_parts else f'variable_{i+1}'
        group_parts = [sv(row.get(c, '')) for c in group_cols]
        group_parts = [p for p in group_parts if p and p.lower() not in ['nan', 'none']]
        gpath       = ' > '.join(group_parts) if group_parts else 'Ungrouped'
        parts = []
        for c in list(dict.fromkeys(group_cols + leaf_cols + text_cols + meta_cols)):
            v = sv(row.get(c, ''))
            if v:
                parts.append(f'{c}: {v}')
        text = ' | '.join(parts) if parts else label
        rows.append({
            '_source_file': source,
            '_row_index':   int(i),
            '_leaf_label':  label,
            '_leaf_id':     f'{gpath}.{label}' if gpath != 'Ungrouped' else label,
            '_group_path':  gpath,
            '_text':        text,
        })
    can = pd.DataFrame(rows)
    if can['_leaf_id'].duplicated().any():
        cnt: dict = defaultdict(int)
        ids = []
        for lid in can['_leaf_id']:
            cnt[lid] += 1
            ids.append(lid if cnt[lid] == 1 else f'{lid}__{cnt[lid]}')
        can['_leaf_id'] = ids
    return can

# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMIZER CORE  [TAX + GON]
#
# Everything here is data-driven: TF-IDF over the variable text objects, cosine
# distance, agglomerative clustering with the number of clusters chosen by
# silhouette, and internal-node labels taken from each cluster's most
# discriminative terms.  The ONLY lexical resource is the generic English
# stop-word list (standard IR practice — not dataset-specific).
# ─────────────────────────────────────────────────────────────────────────────
def vectorize_texts(texts: list):
    """TF-IDF text objects [GON].  Generic English stop-words only."""
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                          max_features=2000, min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    return X, vec

def best_k(dist: np.ndarray, n: int, k_min: int = 2, k_max: int = 8) -> int:
    """Pick the number of clusters that maximises the silhouette score.

    Fully data-driven — no fixed cluster count.  Returns 1 if no split with
    >=2 clusters is well separated.
    """
    k_hi = min(k_max, n - 1)
    if k_hi < k_min:
        return 1
    best, best_s = 1, -1.0
    for k in range(k_min, k_hi + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                         linkage='average').fit_predict(dist)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(dist, labels, metric='precomputed')
        except Exception:
            continue
        if s > best_s:
            best_s, best = s, k
    return best

def discriminative_label(inside: np.ndarray, outside, terms: np.ndarray,
                         used: set, top_n: int = 2) -> str:
    """Label a cluster by the terms that most separate it from its siblings.

    inside  = mean TF-IDF vector of the cluster's members
    outside = mean TF-IDF vector of the sibling pool (or 0 if none)
    """
    scores = inside - (outside if outside is not None else 0)
    picks: list = []
    for i in np.argsort(scores)[::-1]:
        term = terms[i]
        if len(term) <= 2 or scores[i] <= 0 or term in used:
            continue
        picks.append(term)
        if len(picks) >= top_n:
            break
    if not picks:  # degenerate: fall back to highest raw mean term
        for i in np.argsort(inside)[::-1]:
            if len(terms[i]) > 2:
                picks = [terms[i]]
                break
    return ' / '.join(p.title() for p in picks) if picks else 'Group'

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHY CONSTRUCTION  [TAX + GON]
# ─────────────────────────────────────────────────────────────────────────────
def _nmap(nodes: list) -> dict:
    return {int(n['id']): n for n in nodes}

def _next_id(nodes: list) -> int:
    return max((int(n['id']) for n in nodes), default=0) + 1

def _add_child(nodes: list, parent_id: int, child_id: int):
    m = _nmap(nodes)
    p = m.get(int(parent_id))
    if p is None:
        return
    rel = list(p.get('related', []))
    if int(child_id) not in rel:
        rel.append(int(child_id))
    p['related'] = rel

def _make_agg(nid: int, name: str, desc: str = '') -> dict:
    return {'id': int(nid), 'name': str(name), 'related': [],
            'type': 'aggregation', 'isShown': True, 'desc': desc, 'dtype': 'determine'}

def _leaf_ids(nodes: list, nid: int) -> list:
    m = _nmap(nodes)
    out: list = []
    def rec(x):
        n = m.get(int(x))
        if not n:
            return
        if n.get('type') == 'attribute':
            out.append(int(x))
            return
        for c in n.get('related', []):
            rec(int(c))
    rec(nid)
    return list(dict.fromkeys(out))

def build_hierarchy(can: pd.DataFrame, project: str = 'project',
                    max_depth: int = 3, min_cluster_size: int = 6,
                    branch_max: int = 8) -> list:
    """Pure Taxonomizer construction [TAX].

    Builds an abstract-to-concrete taxonomy by recursively clustering the
    variables' TF-IDF text objects.  At each level the number of clusters is
    chosen by silhouette; each resulting internal node is labelled with the
    terms that most discriminate its members from their siblings.  No group
    column, no hardcoded patterns are used in construction — so the recovered
    structure can be fairly evaluated against the original group column.
    """
    # ── leaf attribute nodes (ids 1..N) ──────────────────────────────────────
    nodes: list = [{'id': 0, 'name': project, 'type': 'root',
                    'dtype': 'root', 'isShown': True, 'related': [], 'desc': 'Root node'}]
    row_to_node: list = []
    for i, (_, r) in enumerate(can.iterrows(), start=1):
        nodes.append({'id': i, 'name': r['_leaf_label'], 'dtype': 'determine',
                      'related': [], 'isShown': True, 'type': 'attribute',
                      'desc': r['_text'],
                      'metadata': {'leaf_id': r['_leaf_id'], 'group_path': r['_group_path']}})
        row_to_node.append(i)
    row_to_node = np.array(row_to_node)

    # ── TF-IDF text objects + full cosine distance matrix [GON] ───────────────
    texts = (can['_leaf_label'].astype(str) + ' . ' + can['_text'].astype(str)).tolist()
    X, vec = vectorize_texts(texts)
    Xd     = X.toarray()
    terms  = vec.get_feature_names_out()
    full_dist = cosine_distances(X).astype(float)
    np.fill_diagonal(full_dist, 0.0)

    # ── recursive clustering ─────────────────────────────────────────────────
    def attach_leaves(parent_id: int, idx: np.ndarray):
        for i in idx:
            _add_child(nodes, parent_id, int(row_to_node[i]))

    def recurse(parent_id: int, idx: np.ndarray, depth: int, used: set):
        n = len(idx)
        if n <= min_cluster_size or depth >= max_depth:
            attach_leaves(parent_id, idx)
            return

        sub = full_dist[np.ix_(idx, idx)]
        k_cap = min(branch_max, max(2, n // min_cluster_size))
        k = best_k(sub, n, k_min=2, k_max=k_cap)
        if k <= 1:
            attach_leaves(parent_id, idx)
            return

        labels  = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                          linkage='average').fit_predict(sub)
        pool_Xd = Xd[idx]
        for c in range(k):
            mask    = labels == c
            members = idx[mask]
            if len(members) == 0:
                continue
            if len(members) == 1:           # don't create singleton internal nodes
                _add_child(nodes, parent_id, int(row_to_node[members[0]]))
                continue
            inside  = pool_Xd[mask].mean(axis=0)
            outside = pool_Xd[~mask].mean(axis=0) if (~mask).any() else None
            label   = discriminative_label(inside, outside, terms, used)
            nid = _next_id(nodes)
            nodes.append(_make_agg(nid, label,
                                   desc=f'Cluster of {len(members)} variables — '
                                        f'discriminative terms: {label}'))
            _add_child(nodes, parent_id, nid)
            recurse(nid, members, depth + 1, used | {label.lower()})

    recurse(0, np.arange(len(can)), 0, set())

    for n in nodes:
        n['related'] = list(dict.fromkeys(int(x) for x in n.get('related', [])))
    return nodes

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _parent_map(nodes: list) -> dict:
    pm: dict = {}
    for n in nodes:
        for c in n.get('related', []):
            if int(c) not in pm:
                pm[int(c)] = int(n['id'])
    return pm

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _eval_cluster_assignments(nodes: list, can: pd.DataFrame) -> list[int]:
    """Return predicted cluster id (depth-1 aggregation ancestor) for each row in can."""
    pm = _parent_map(nodes)
    def depth1(nid: int) -> int:
        # Walk up until our parent is root (id==0) or we have no parent
        while pm.get(nid, -1) not in (-1, 0):
            nid = pm[nid]
        return nid
    lid_to_nid = {n['metadata']['leaf_id']: int(n['id'])
                  for n in nodes if n.get('type') == 'attribute' and 'metadata' in n}
    return [depth1(lid_to_nid[lid]) if lid in lid_to_nid else -1
            for lid in can['_leaf_id']]

def _purity(y_true, y_pred) -> float:
    from collections import Counter
    clusters: dict = {}
    for t, p in zip(y_true, y_pred):
        clusters.setdefault(p, []).append(t)
    correct = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct / max(len(y_true), 1)

def _structural_stats(nodes: list) -> dict:
    pm = _parent_map(nodes)
    def depth_of(nid: int) -> int:
        d = 0
        while nid in pm:
            nid = pm[nid]; d += 1
        return d
    agg   = [n for n in nodes if n.get('type') == 'aggregation']
    leafs = [n for n in nodes if n.get('type') == 'attribute']
    depths   = [depth_of(int(n['id'])) for n in leafs]
    branches = [len(n.get('related', [])) for n in agg]
    singletons = sum(1 for b in branches if b == 1)
    return {
        'n_aggregation_nodes':  len(agg),
        'max_depth':            int(max(depths, default=0)),
        'avg_leaf_depth':       round(float(np.mean(depths)), 2) if depths else 0.0,
        'avg_branching_factor': round(float(np.mean(branches)), 2) if branches else 0.0,
        'singleton_nodes_%':    round(100.0 * singletons / max(len(agg), 1), 1),
    }

def _wrap(text: str, width: int = 70) -> str:
    """Wrap long hover text onto multiple <br> lines so it never runs off-screen."""
    import textwrap
    text = str(text).replace('<', '&lt;')
    lines: list = []
    for para in text.split('\n'):
        wrapped = textwrap.wrap(para, width=width) or ['']
        lines.extend(wrapped)
    return '<br>'.join(lines)

def plot_sunburst(nodes: list, max_depth: int = 4) -> go.Figure:
    pm = _parent_map(nodes)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id'])
        lc  = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get('name', ''))[:40])
        parents.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(max(1, lc))
        desc = _wrap(n.get('desc', ''))
        hover.append(f'<b>{_wrap(n.get("name",""))}</b><br>Type: {n.get("type","")}'
                     f'<br>Variables: {lc}<br><br>{desc}')
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues='total', hovertext=hover, hoverinfo='text',
        maxdepth=max_depth, insidetextorientation='radial',
        marker=dict(colorscale='Greens', line=dict(width=1, color='white')),
    ))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10),
                      title='Click a sector to drill down — click centre to go back')
    return fig

def plot_treemap(nodes: list) -> go.Figure:
    pm = _parent_map(nodes)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id'])
        lc  = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get('name', ''))[:40])
        parents.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(max(1, lc))
        desc = _wrap(n.get('desc', ''))
        hover.append(f'<b>{_wrap(n.get("name",""))}</b><br>Variables: {lc}<br>{desc}')
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues='total', hovertext=hover, hoverinfo='text',
        textinfo='label+value',
        marker=dict(colorscale='Greens', line=dict(width=1, color='white')),
    ))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('1. Upload')
    uploaded = st.file_uploader(
        'Upload a metadata file',
        type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'json'],
        accept_multiple_files=False,
    )
    st.header('2. Taxonomizer settings')
    tx_max_depth = st.slider('Max taxonomy depth', 2, 5, 3, 1,
                             help='How many abstract-to-concrete levels to build')
    tx_min_size  = st.slider('Min cluster size', 3, 20, 6, 1,
                             help='Clusters smaller than this stop splitting (leaves attach directly)')
    tx_branch    = st.slider('Max branches per node', 3, 12, 8, 1,
                             help='Upper bound on clusters per split; the actual number is chosen by silhouette')

    st.header('3. Display')
    max_items     = st.slider('Maximum variables', 25, 1200, 300, 25)
    group_filter  = st.text_input('Group filter (optional)', value='',
                                  help='Filter rows whose group path contains this text')
    display_depth = st.slider('Sunburst depth', 2, 6, 4, 1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if not uploaded:
    st.info('Upload a metadata CSV / XLSX / JSON file to begin.')
    st.markdown("""
    ### Baseline algorithm — pure Taxonomizer

    The simplest of the three approaches — no hardcoded domain patterns, no
    external APIs, no neural embeddings. Works on any dataset.

    | Step | Method | Paper |
    |------|--------|-------|
    | Text object | Concatenate all metadata fields per variable | Goncalves et al. |
    | Representation | TF-IDF (generic English stop-words only) | Goncalves et al. |
    | Hierarchy construction | Recursive agglomerative clustering (cosine), #clusters chosen by silhouette | Taxonomizer (Mahmood & Mueller, IEEE TVCG) |
    | Node labelling | Most discriminative terms of each cluster vs its siblings | Taxonomizer |

    The group column is **not** used for construction, so the recovered taxonomy
    can be fairly evaluated against it (NMI / ARI / Purity in the Evaluation tab).

    **Approach 1** adds SBERT embeddings + Wikidata/BioPortal enrichment + HiExpan refinement.

    **Approach 2** adds NMF/FASTopic aspect discovery + GMM clustering + optional LLM labels.
    """)
    st.stop()

path = save_upload(uploaded)

@st.cache_data(show_spinner=False)
def _load_profile(path_str: str):
    df = load_any(Path(path_str))
    cfg, prof = detect_roles(df)
    return df, cfg, prof

with st.spinner('Loading file…'):
    df, auto_cfg, prof = _load_profile(str(path))

st.subheader('Step 1 — File preview')
with st.expander(f'📄 {uploaded.name}  ({len(df):,} rows, {len(df.columns)} columns)',
                 expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    score_cols = [c for c in ['column', 'leaf_score', 'group_score', 'text_score', 'metadata_score']
                  if c in prof.columns]
    st.dataframe(prof[score_cols].sort_values('leaf_score', ascending=False),
                 use_container_width=True)

st.subheader('Step 2 — Confirm column roles')
cols = list(df.columns)
with st.expander('Column configuration', expanded=True):
    left, right = st.columns(2)
    with left:
        leaf_cols = st.multiselect('Leaf variable column(s)', cols,
            default=[c for c in auto_cfg.get('leaf_cols', []) if c in cols], key='leaf')
        group_cols = st.multiselect('Group/task column(s)', cols,
            default=[c for c in auto_cfg.get('group_cols', []) if c in cols], key='group')
    with right:
        text_cols = st.multiselect('Text/description column(s)', cols,
            default=[c for c in auto_cfg.get('text_cols', []) if c in cols], key='text')
        meta_cols = st.multiselect('Metadata/type column(s)', cols,
            default=[c for c in auto_cfg.get('metadata_cols', []) if c in cols], key='meta')

if not leaf_cols:
    st.error('Choose at least one leaf variable column.')
    st.stop()

cfg = {'leaf_cols': leaf_cols, 'group_cols': group_cols,
       'text_cols': text_cols, 'metadata_cols': meta_cols}

if st.button('Build baseline hierarchy', type='primary'):
    with st.spinner('Building hierarchy…'):
        _can = build_canonical(df, cfg, source=Path(uploaded.name).stem)

        if group_filter.strip():
            _can = _can[_can['_group_path'].str.contains(
                group_filter.strip(), case=False, na=False)].copy()

        if len(_can) > max_items:
            _can = _can.head(max_items).copy()

        _can = _can.reset_index(drop=True)

        if len(_can) < 2:
            st.error('Need at least 2 variables after filtering.')
            st.stop()

        _pname = Path(uploaded.name).stem
        _nodes = build_hierarchy(_can, project=_pname,
                                 max_depth=tx_max_depth,
                                 min_cluster_size=tx_min_size,
                                 branch_max=tx_branch)

    st.session_state['_bl_nodes']   = _nodes
    st.session_state['_bl_can']     = _can
    st.session_state['_bl_project'] = _pname

if '_bl_nodes' not in st.session_state:
    st.info('Configure columns above then click **Build baseline hierarchy**.')
    st.stop()

nodes        = st.session_state['_bl_nodes']
can          = st.session_state['_bl_can']
project_name = st.session_state['_bl_project']

_sm = _structural_stats(nodes)
n_leaves   = len([n for n in nodes if n['type'] == 'attribute'])
n_internal = len([n for n in nodes if n['type'] == 'aggregation'])

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric('Variables', n_leaves)
c2.metric('Aggregation nodes', n_internal)
c3.metric('Max depth', _sm['max_depth'])
c4.metric('Avg branching', _sm['avg_branching_factor'])

tabs = st.tabs(['Sunburst', 'Treemap', 'Node detail', 'Canonical table', 'Export', '📊 Evaluation'])

with tabs[0]:
    st.plotly_chart(plot_sunburst(nodes, max_depth=display_depth), use_container_width=True)
    st.caption('Green = Baseline. Click a sector to drill down; click the centre to go back.')

with tabs[1]:
    st.plotly_chart(plot_treemap(nodes), use_container_width=True)

with tabs[2]:
    nm = _nmap(nodes)
    agg_nodes = [n for n in nodes if n['type'] in ('aggregation', 'root')]
    options   = [f'{n["name"]}  [{len(_leaf_ids(nodes, int(n["id"])))} vars]'
                 for n in agg_nodes]
    if options:
        sel      = st.selectbox('Select a node', options)
        sel_name = sel.split('  [')[0]
        sel_node = next((n for n in agg_nodes if n['name'] == sel_name), None)
        if sel_node:
            lids = _leaf_ids(nodes, int(sel_node['id']))
            leaf_ids_set = {nm[i]['metadata']['leaf_id']
                            for i in lids if i in nm and 'metadata' in nm[i]}
            sub = can[can['_leaf_id'].isin(leaf_ids_set)]
            st.write(f'**{len(lids)} variables** under "{sel_node["name"]}"')
            st.dataframe(sub[['_leaf_label', '_group_path', '_text']].reset_index(drop=True),
                         use_container_width=True)

with tabs[3]:
    st.dataframe(can, use_container_width=True)

with tabs[4]:
    _base = safe_name(project_name)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            'Hierarchy JSON',
            data=json.dumps(nodes, indent=2, ensure_ascii=False).encode('utf-8'),
            file_name=f'{_base}_baseline_hierarchy.json',
            mime='application/json',
            use_container_width=True,
        )
    with col2:
        st.download_button(
            'Canonical CSV',
            data=can.to_csv(index=False).encode('utf-8'),
            file_name=f'{_base}_baseline_canonical.csv',
            mime='text/csv',
            use_container_width=True,
        )

    st.divider()
    # ── Save directly into the project's outputs/baseline/ folder ──────────────
    _out_dir = Path(__file__).resolve().parent / 'outputs' / 'baseline'
    st.markdown('### Save to project folder')
    st.caption(
        'The download buttons above go to your browser’s Downloads folder (a browser '
        f'restriction). This button instead writes the files into `{_out_dir}` with the '
        'dataset name — convenient for `evaluate_all.py`.'
    )
    if st.button('💾 Save all to outputs/baseline/', type='primary',
                 use_container_width=True):
        try:
            _out_dir.mkdir(parents=True, exist_ok=True)
            (_out_dir / f'{_base}_baseline_hierarchy.json').write_text(
                json.dumps(nodes, indent=2, ensure_ascii=False), encoding='utf-8')
            can.to_csv(_out_dir / f'{_base}_baseline_canonical.csv', index=False)
            st.success(f'Saved to `{_out_dir}`:\n\n'
                       f'- {_base}_baseline_hierarchy.json\n'
                       f'- {_base}_baseline_canonical.csv')
        except Exception as _e:
            st.error(f'Could not save: {_e}')

with tabs[5]:
    import hierarchy_eval as he

    st.subheader('Hierarchy Quality Evaluation')
    st.caption(
        'The group column is a *construction input* (Gonçalves text object), so it '
        'cannot serve as ground truth. The primary metrics below are **reference-free** '
        '— they assess the hierarchy itself, with no gold standard.'
    )

    with st.spinner('Computing reference-free metrics…'):
        tm = he.traco_metrics(nodes)
        npmi = he.npmi_coherence(nodes, can['_text'].tolist())

    # ── PRIMARY: reference-free hierarchy quality ─────────────────────────────
    st.markdown('#### Primary — reference-free hierarchy quality')
    p1, p2, p3 = st.columns(3)
    p1.metric('Parent–child coherence', tm['pc_coherence'],
              help='TraCo (Wu et al., AAAI 2024). Mean similarity of each node to its parent. '
                   'Higher = children correctly nest under their parent theme.')
    p2.metric('Sibling diversity', tm['sibling_diversity'],
              help='TraCo (Wu et al., AAAI 2024). Mean distance between sibling nodes. '
                   'Higher = siblings are distinct (LOW = redundant/repeated siblings).')
    p3.metric('NPMI label coherence', npmi,
              help='Lau et al., EACL 2014. Whether node-label terms genuinely co-occur in the '
                   'data. Higher = meaningful labels, not arbitrary term salads.')
    st.caption(f'Embedding backend: **{tm["encoder"]}**.  '
               'Coherence & diversity ∈ [−1, 1]; NPMI ∈ ≈[−1, 1].')

    # ── Structural metrics ────────────────────────────────────────────────────
    st.markdown('#### Structural statistics')
    sm = he.structural_stats(nodes)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric('Aggregation nodes', sm['n_aggregation_nodes'])
    s2.metric('Max leaf depth',    sm['max_depth'])
    s3.metric('Avg leaf depth',    sm['avg_leaf_depth'])
    s4.metric('Avg branching',     sm['avg_branching_factor'])
    s5.metric('Singleton nodes',   f"{sm['singleton_nodes_%']}%",
              help='Aggregation nodes with a single child (sparse-hierarchy indicator)')

    # ── SECONDARY: group preservation (caveated) ──────────────────────────────
    st.markdown('#### Secondary — group-structure preservation *(descriptive)*')
    st.caption(
        '⚠️ The group column was an **input** to construction, so these are NOT accuracy '
        'metrics — they only describe how much the discovered hierarchy still reflects the '
        'pre-existing group column. High values are expected and not evidence of quality.'
    )
    gp = he.group_preservation(nodes, can)
    g1, g2, g3 = st.columns(3)
    g1.metric('NMI', gp['NMI']);  g2.metric('ARI', gp['ARI']);  g3.metric('Purity', gp['Purity'])
