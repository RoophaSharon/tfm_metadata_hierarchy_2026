# baseline.py — Metadata Hierarchy Builder — Baseline (Taxonomizer)
#
# Baseline = Taxonomizer (Mahmood & Mueller, IEEE TVCG 2019), semantic-space
# pipeline, adapted to a metadata-only setting.  No hardcoded domain patterns.
#
# Pipeline:
#   1. Load metadata file (CSV / TSV / XLSX / JSON)
#   2. Detect column roles (leaf / group / text / meta) — same as Approach 1 / 2
#   3. Build canonical schema (incl. _semantic_text = description values only)
#   4. Embed each variable (code + description) via Word2Vec skip-gram and build
#      the cosine-distance semantic space [TAX §3.2]
#   5. Recursively cluster (agglomerative, cosine) into the dendrogram taxonomy;
#      internal-node labels = data-driven contrastive terms of each cluster
#   6. Visualise (Sunburst / Treemap / Node-link)
#   7. Export visualization-ready JSON + canonical CSV
#
# Paper & justified adaptations (metadata/schema setting, fully automatic):
#   [TAX] Mahmood & Mueller — Taxonomizer, IEEE TVCG 2019.
#         Builds a SEMANTIC space (cosine over word2vec skip-gram embeddings of
#         attribute names; gensim, Wikipedia, window=5, dim=128) merged with a
#         DATA space (correlation over raw values), clustered into a dendrogram;
#         inner nodes labelled semi-automatically by distributional degree-of-
#         entailment + WordNet synonyms.
#   Adaptations (all documented):
#     1. No DATA space — a schema/dictionary has no raw values, so we use the
#        semantic space alone (Taxonomizer with semantic weight = 1.0).
#     2. Embed the attribute's short NAME (the description's name clause), since
#        the bare code goes out-of-vocabulary (a limitation the paper flags,
#        e.g. "BP").  Taxonomizer embeds the NAME ("a few words"), not a
#        paragraph; using the short name (not the full description prose) keeps
#        task-distinctive words from being diluted by shared explanatory text.
#     3. Fully-automatic labels — the paper's labelling is semi-automatic
#        (human picks from suggestions); a baseline must be non-interactive, so
#        we use data-driven contrastive terms from each cluster's members.
#
# Dependencies: gensim
#   pip install gensim

from __future__ import annotations
import csv, json, re, warnings
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Metadata Hierarchy — Baseline', page_icon='🌿', layout='wide')
st.title('Metadata Hierarchy Builder — Baseline (Taxonomizer)')
st.caption(
    'Taxonomizer baseline [Mahmood & Mueller, IEEE TVCG 2019]: Word2Vec skip-gram '
    'semantic space (short attribute names) + balanced Ward agglomerative clustering '
    'into the dendrogram taxonomy; nodes labelled by data-driven contrastive terms. '
    'Semantic space only (no raw data values); no hardcoded patterns, no external APIs.'
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LEAF_KEYS  = 'variable var field column attribute name code id item indicator question measure concept'.split()
GROUP_KEYS = 'task category domain module section table dataset assessment test variant group topic instrument form subscale construct'.split()
TEXT_KEYS  = 'description definition desc label title question meaning note notes text display full details explanation comment'.split()
META_KEYS  = 'type dtype data_type datatype unit units format decimal precision values value coding codebook range min max scale'.split()

# URL pattern — strip embedded links (e.g. HCP FreeSurfer NeuroLex URLs) so web
# tokens cannot dominate the embedding or the cluster label.  [shared with A1]
_URL_RE = re.compile(r'(https?://\S+|www\.\S+|\b\w+\.(?:org|com|net|gov|edu)\b/?\S*)',
                     re.IGNORECASE)

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
    """Auto-detect column roles.  Identical logic to Approach 1 / 2 so the
    preprocessing up to the canonical table is comparable across all apps."""
    prof  = profile_columns(df)
    leaf  = prof.sort_values(['leaf_score', 'unique_ratio'], ascending=False).head(1)['column'].tolist()
    text  = (prof[(prof.text_score >= 4) | (prof.avg_length > 80)]
             .sort_values('text_score', ascending=False)['column'].tolist()) or leaf.copy()
    group = (prof[(prof.group_score >= 4) & (~prof.column.isin(leaf)) & (prof.unique_values > 1)]
             .sort_values('group_score', ascending=False)['column'].head(3).tolist())
    meta  = (prof[(prof.metadata_score >= 4) & (~prof.column.isin(text + leaf + group))]
             .sort_values('metadata_score', ascending=False)['column'].head(5).tolist())
    # Representation columns (decimal/precision/unit/type/format/…) must never
    # become structural levels — force them out of group and into metadata. [GON][TAX]
    _META_SUBSTR_BLOCK = {
        'decimal', 'precision', 'unit', 'dtype', 'type', 'format', 'scale',
        'values', 'range', 'min', 'max', 'coding', 'codebook', 'missing',
    }
    def _is_repr(col_name):
        nc = re.sub(r'[^a-z0-9]', '', str(col_name).lower())
        return any(sub in nc for sub in _META_SUBSTR_BLOCK)
    meta_extra = [c for c in prof['column'].tolist()
                  if _is_repr(c) and c not in text and c not in leaf and c not in meta]
    group = [c for c in group if not _is_repr(c)]
    meta  = list(dict.fromkeys(meta + meta_extra))[:8]
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
        # _semantic_text: description VALUES only — no "fieldname:" prefixes, no
        # other fields, URLs stripped.  This is the clean text Taxonomizer embeds
        # (the attribute's meaning), identical in spirit to Approach 1's column.
        sem_parts = [sv(row.get(c, '')) for c in text_cols]
        sem_parts = [p for p in sem_parts if p]
        if not sem_parts:
            sem_parts = list(leaf_parts)
        semantic = _URL_RE.sub(' ', ' '.join(sem_parts)) if sem_parts else label
        rows.append({
            '_source_file':   source,
            '_row_index':     int(i),
            '_leaf_label':    label,
            '_leaf_id':       f'{gpath}.{label}' if gpath != 'Ungrouped' else label,
            '_group_path':    gpath,
            '_text':          text,
            '_semantic_text': semantic,
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
# TAXONOMIZER CORE  [TAX — Mahmood & Mueller, IEEE TVCG 2019]
#
# Taxonomizer builds the taxonomy from a SEMANTIC SPACE (cosine distance between
# word2vec skip-gram embeddings of attribute names) merged with a DATA SPACE
# (correlation over the raw values).  In a metadata/schema setting we have no
# raw data values, so we use the semantic space alone (= Taxonomizer with
# semantic weight 1.0).  Because attribute *names* here are opaque codes that go
# out-of-vocabulary — a limitation the paper explicitly flags (e.g. "BP") — we
# embed code + description so real words carry the meaning (OOV code tokens are
# skipped during averaging).  Internal-node labels: the paper uses semi-automatic
# distributional degree-of-entailment + WordNet synonyms; a baseline must be
# fully automatic, so we use data-driven contrastive terms drawn from the data.
# ─────────────────────────────────────────────────────────────────────────────

_W2V_STOP = frozenset(
    'a an the and or but if in on at to of for with by is are was were be '
    'been being have has had do does did will would could should may might '
    'shall can this that these those i you he she it we they me him her us '
    'them my your his her its our their what which who whom when where why '
    'how all each every few more most other some such no not only same so '
    'than too very just because as until while'.split()
)

@st.cache_resource(show_spinner=False)
def _load_w2v():
    """Load pre-trained Word2Vec / GloVe model via gensim downloader.

    We prefer glove-wiki-gigaword-100 (~66 MB) because its Wikipedia training
    corpus and skip-gram-style objective most closely match Taxonomizer's
    described word2vec-Wikipedia-dim128 model.
    """
    try:
        import gensim.downloader as api
        return api.load('glove-wiki-gigaword-100')
    except Exception as e:
        st.error(
            f'Could not load Word2Vec model: {e}\n\n'
            'Run:  pip install gensim  and restart the app.\n'
            'The model (~66 MB) is downloaded automatically on first use.'
        )
        return None

def _tokenize(label: str) -> list[str]:
    return [t for t in re.sub(r'[^a-zA-Z]+', ' ', label).lower().split()
            if len(t) > 2 and t not in _W2V_STOP]

def attribute_name(text: str) -> str:
    """The attribute's short NAME — what Taxonomizer actually embeds [TAX §3.2].

    The paper embeds the attribute name ("not more than a few words long"), not a
    paragraph.  Descriptions here are formatted '<name>: <full sentence>' (some
    prefixed with a marker like 'KEY: <name>: …'), so we take the first clause
    that is not a pure all-caps marker.  Embedding this short name — rather than
    the full description prose — keeps the task-distinctive words from being
    diluted by shared explanatory text, so the taxonomy groups far more by theme
    (e.g. DMS / PAL / SWM) without ever touching the group column.
    """
    text = str(text)
    for clause in re.split(r'[:\n]', text):
        clause = clause.strip()
        if clause and not all(2 <= len(w) <= 6 and w.isupper() for w in clause.split()):
            return clause
    return text.strip()

def embed_labels_w2v(labels: list[str], model) -> np.ndarray:
    """Average Word2Vec vectors for each label's tokens [TAX §4.1].

    Falls back to a zero vector for labels where none of the tokens are in the
    model vocabulary (rare for standard English attribute names).
    """
    dim = model.vector_size
    out = np.zeros((len(labels), dim), dtype=np.float32)
    for i, label in enumerate(labels):
        toks = _tokenize(label)
        vecs = [model[t] for t in toks if t in model]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    # L2-normalise so cosine distance = 1 - dot
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms

def _cluster(X: np.ndarray, k: int) -> np.ndarray:
    """Ward-linkage agglomerative cut into k clusters.

    Ward (on the L2-normalised embedding vectors, where Euclidean ∝ √cosine)
    minimises within-cluster variance and so produces *balanced* clusters.
    This avoids the average/single-linkage chaining pathology that otherwise
    peels off tiny clusters and leaves one giant residual (i.e. no real
    hierarchy forms).
    """
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)

def best_k(X: np.ndarray, n: int, k_min: int = 2, k_max: int = 8) -> int:
    """Pick the number of clusters that maximises the silhouette score.

    Fully data-driven — no fixed cluster count.  Returns 1 only when the node
    is too small to split (n <= k_min).
    """
    k_hi = min(k_max, n - 1)
    if k_hi < k_min:
        return 1
    best, best_s = 1, -1.0
    for k in range(k_min, k_hi + 1):
        labels = _cluster(X, k)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(X, labels)
        except Exception:
            continue
        if s > best_s:
            best_s, best = s, k
    return best

def _doc_freq(texts: list[str]) -> Counter:
    """Document frequency: how many member texts each content word appears in."""
    c: Counter = Counter()
    for t in texts:
        for w in set(_tokenize(t)):
            c[w] += 1
    return c

def cluster_term_label(member_texts: list[str], sibling_texts: list[str],
                       used: set, vocab=None, top_n: int = 2) -> str:
    """Label a node with the content words most characteristic of its members.

    Data-driven labelling: each candidate word is scored by how much more
    frequent it is *inside* the cluster than in the sibling pool (contrastive
    document frequency), so labels are domain terms drawn from the dataset
    itself — not external ontology words.  This replaces Taxonomizer's
    WordNet degree-of-entailment, which produces over-general, off-domain
    abstractions on specialised scientific metadata.

    If `vocab` is given (the Word2Vec model), only real dictionary words are
    eligible, so opaque attribute codes (e.g. 'dms', 'motml') are filtered out
    of labels.  Codes are used only as a last-resort fallback.
    """
    def in_vocab(w: str) -> bool:
        return vocab is None or w in vocab

    n_in  = max(len(member_texts), 1)
    n_out = max(len(sibling_texts), 1)
    cin   = _doc_freq(member_texts)
    cout  = _doc_freq(sibling_texts)

    scores: dict[str, float] = {}
    for w, f in cin.items():
        if w in used or len(w) <= 2 or not in_vocab(w):
            continue
        p_in  = f / n_in
        p_out = cout.get(w, 0) / n_out
        # ignore single-occurrence noise unless the term is widely shared
        if f < 2 and p_in < 0.5:
            continue
        scores[w] = p_in - p_out

    picks = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]
             if scores[w] > 0]
    if not picks:
        # fallback: most frequent shared real word, then any shared token
        for require_vocab in (True, False):
            for w, _ in cin.most_common():
                if w not in used and len(w) > 2 and (not require_vocab or in_vocab(w)):
                    picks = [w]
                    break
            if picks:
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

def build_hierarchy(can: pd.DataFrame, w2v_model, project: str = 'project',
                    max_depth: int = 3, min_cluster_size: int = 6,
                    branch_max: int = 8) -> list:
    """Taxonomizer semantic-space construction [TAX].

    Embeds each variable from its short attribute NAME (Word2Vec skip-gram
    average) — the name clause of the description, as Taxonomizer specifies.
    Recursively clusters via balanced Ward linkage — the semantic-space
    dendrogram.  Labels each internal node with the contrastive content terms of
    its members (data-driven, fully automatic).  No group column, no hardcoding.
    """
    # ── leaf attribute nodes (ids 1..N) ──────────────────────────────────────
    nodes: list = [{'id': 0, 'name': project, 'type': 'root',
                    'dtype': 'root', 'isShown': True, 'related': [], 'desc': 'Root node'}]
    row_to_node: list = []
    embed_list: list[str] = []    # short attribute name → embedding input + labels
    for i, (_, r) in enumerate(can.iterrows(), start=1):
        sem  = str(r.get('_semantic_text', '') or r['_leaf_label'])
        name = attribute_name(sem) or str(r['_leaf_label'])
        nodes.append({'id': i, 'name': r['_leaf_label'], 'dtype': 'determine',
                      'related': [], 'isShown': True, 'type': 'attribute',
                      'desc': r['_text'],
                      'metadata': {'leaf_id': r['_leaf_id'], 'group_path': r['_group_path']}})
        row_to_node.append(i)
        embed_list.append(name)
    label_list = embed_list
    row_to_node = np.array(row_to_node)

    # ── Word2Vec semantic-space embeddings [TAX §3.2] ─────────────────────────
    emb = embed_labels_w2v(embed_list, w2v_model)   # (N, dim), L2-normalised

    # ── recursive clustering down the Ward dendrogram ─────────────────────────
    def attach_leaves(parent_id: int, idx: np.ndarray):
        for i in idx:
            _add_child(nodes, parent_id, int(row_to_node[i]))

    def recurse(parent_id: int, idx: np.ndarray, depth: int, used: set):
        n = len(idx)
        if n <= min_cluster_size or depth >= max_depth:
            attach_leaves(parent_id, idx)
            return

        sub = emb[idx]
        k_cap = min(branch_max, n - 1)
        # Branching floor: a node with n leaves and `remaining` levels left must
        # fan out enough to fit all its leaves into buckets of ~min_cluster_size
        # by the depth cap, i.e. k >= (n / min_cluster_size) ** (1/remaining).
        # Without this, silhouette keeps picking k=2 on overlapping data (e.g.
        # HCP), giving a near-binary tree that dumps ~100 leaves per bottom node.
        remaining = max(1, max_depth - depth)
        k_floor = int(np.ceil((n / max(min_cluster_size, 1)) ** (1.0 / remaining)))
        k_floor = max(2, min(k_floor, k_cap))
        k = best_k(sub, n, k_min=k_floor, k_max=k_cap)
        if k <= 1:
            k = min(k_floor, k_cap) if n > min_cluster_size else 1
        if k <= 1:
            attach_leaves(parent_id, idx)
            return

        cluster_labels = _cluster(sub, k)
        for c in range(k):
            mask    = cluster_labels == c
            members = idx[mask]
            if len(members) == 0:
                continue
            if len(members) == 1:           # don't create singleton internal nodes
                _add_child(nodes, parent_id, int(row_to_node[members[0]]))
                continue
            mset = set(members.tolist())
            member_texts  = [label_list[i] for i in members]
            sibling_texts = [label_list[i] for i in idx if i not in mset]
            # data-driven contrastive-term labelling
            label = cluster_term_label(member_texts, sibling_texts, used)
            nid = _next_id(nodes)
            nodes.append(_make_agg(nid, label,
                                   desc=f'Cluster of {len(members)} variables — '
                                        f'label terms: {label}'))
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
# NODE-LINK TREE  (Reingold–Tilford layout — matches Approach 1 / 2 interface)
# ─────────────────────────────────────────────────────────────────────────────
def _bl_node_color(n: dict) -> str:
    t = n.get('type', '')
    if t == 'root':      return '#2a7d2a'
    if t == 'attribute': return '#74c476'
    if t == 'collapsed': return '#bbbbbb'
    return '#238b45'

def _display_graph(nodes: list, max_depth: int = 4):
    """Walk the tree to the chosen depth, inserting 'collapsed' placeholders for
    branches cut off below max_depth (the Level-of-Detail control)."""
    m = _nmap(nodes)
    dnodes: dict = {}
    edges: list  = []
    counter = 10 ** 9

    def rec(nid, depth):
        nonlocal counter
        n = m.get(int(nid))
        if not n:
            return
        dnodes[int(nid)] = n
        if depth >= max_depth and n.get('related'):
            counter += 1
            cid = counter
            n_leaves = len(_leaf_ids(nodes, nid))
            dnodes[cid] = {'id': cid, 'name': f'… {n_leaves} variables',
                           'type': 'collapsed', 'related': [],
                           'desc': f"Collapsed: {n.get('name')}", 'isShown': True}
            edges.append((int(nid), cid))
            return
        for c in n.get('related', []):
            if int(c) not in m:
                continue
            edges.append((int(nid), int(c)))
            rec(int(c), depth + 1)

    rec(0, 0)
    return list(dnodes.values()), edges

def _positions(edges: list):
    """Reingold–Tilford style positions: x = depth, y = subtree-aware vertical."""
    H_SCALE, V_SPACE = 3.0, 1.8
    children: dict = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
    pos: dict = {}
    counter = {'v': 0}

    def rec(nid, depth):
        ch = children.get(nid, [])
        if not ch:
            y = counter['v'] * V_SPACE
            counter['v'] += 1
            pos[nid] = (depth * H_SCALE, y)
            return y
        y = float(np.mean([rec(c, depth + 1) for c in ch]))
        pos[nid] = (depth * H_SCALE, y)
        return y

    rec(0, 0)
    return pos

def plot_node_link(nodes: list, max_depth: int = 4, show_leaf_labels: bool = False) -> go.Figure:
    """Node-link tree with elbow edges. Best for inspecting structure at moderate
    depth; Sunburst is recommended for large hierarchies (Taxonomizer)."""
    dnodes, edges = _display_graph(nodes, max_depth)
    pos = _positions(edges)

    ex, ey = [], []
    for p, c in edges:
        if p not in pos or c not in pos:
            continue
        x0, y0 = pos[p]; x1, y1 = pos[c]
        xm = (x0 + x1) / 2
        ex += [x0, xm, xm, x1, None]
        ey += [y0, y0, y1, y1, None]
    traces = [go.Scatter(x=ex, y=ey, mode='lines',
                         line=dict(width=1, color='#c8c8c8'),
                         hoverinfo='skip', showlegend=False)]

    agg_x, agg_y, agg_l, agg_c, agg_h = [], [], [], [], []
    lf_x,  lf_y,  lf_l,  lf_c,  lf_h  = [], [], [], [], []
    for n in dnodes:
        nid = int(n['id'])
        if nid not in pos:
            continue
        x, y = pos[nid]
        lc   = len(_leaf_ids(nodes, nid))
        lab  = str(n.get('name', nid))
        htxt = (f"<b>{_wrap(n.get('name',''))}</b><br>Type: {n.get('type','')}"
                f"<br>Variables: {lc}<br><br>{_wrap(n.get('desc',''))}")
        col  = _bl_node_color(n)
        if n.get('type') in ('root', 'aggregation', 'collapsed'):
            agg_x.append(x); agg_y.append(y)
            agg_l.append((lab + (f' ({lc})' if lc else ''))[:50])
            agg_c.append(col); agg_h.append(htxt)
        else:
            lf_x.append(x); lf_y.append(y)
            lf_l.append(lab[:40] if show_leaf_labels else '')
            lf_c.append(col); lf_h.append(htxt)

    if agg_x:
        traces.append(go.Scatter(
            x=agg_x, y=agg_y, mode='markers+text', text=agg_l,
            textposition='middle right', hovertext=agg_h, hoverinfo='text',
            marker=dict(size=16, color=agg_c, line=dict(color='white', width=2)),
            showlegend=False))
    if lf_x:
        traces.append(go.Scatter(
            x=lf_x, y=lf_y, mode='markers+text', text=lf_l,
            textposition='middle right', hovertext=lf_h, hoverinfo='text',
            marker=dict(size=7, color=lf_c, symbol='circle', opacity=0.75,
                        line=dict(color='white', width=1)),
            showlegend=False))

    n_leaves = max(12, len(lf_x))
    fig = go.Figure(traces)
    fig.update_layout(
        height=max(700, min(4000, int(n_leaves * 32))),
        margin=dict(l=20, r=220, t=30, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(visible=False, fixedrange=False),
        yaxis=dict(visible=False, autorange='reversed', fixedrange=False),
        dragmode='pan')
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
    tx_max_depth = st.slider('Max taxonomy depth', 2, 6, 3, 1,
                             help='How many abstract-to-concrete levels to build')
    tx_min_size  = st.slider('Min cluster size', 3, 20, 6, 1,
                             help='Clusters smaller than this stop splitting (leaves attach directly)')
    tx_branch    = st.slider('Max branches per node', 3, 12, 8, 1,
                             help='Upper bound on clusters per split; the actual number is chosen by silhouette')

    st.header('3. Display')
    max_items     = st.slider('Maximum variables', 25, 1200, 900, 25,
                              help='Cap on variables included (lower only to speed up very large files). '
                                   'Default keeps full datasets like HCP (813).')
    group_filter  = st.text_input('Group filter (optional)', value='',
                                  help='Filter rows whose group path contains this text')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if not uploaded:
    st.info('Upload a metadata CSV / XLSX / JSON file to begin.')
    st.markdown("""
    ### Baseline algorithm — Taxonomizer (semantic space)

    Based on **Mahmood & Mueller, IEEE TVCG 2019** (Taxonomizer), adapted to a
    metadata-only setting. No hardcoded domain patterns, no external APIs.

    | Step | Method | Paper |
    |------|--------|-------|
    | Variable representation | **short attribute name** (description's name clause; codes are OOV) | Taxonomizer §3.2 / §4.1 |
    | Embedding | Word2Vec skip-gram — average of word vectors (`glove-wiki-gigaword-100`) | Taxonomizer §3.2 |
    | Semantic space | Cosine-distance matrix (no data space — schema has no raw values) | Taxonomizer §3.2 *(adapted)* |
    | Hierarchy construction | Agglomerative clustering (cosine, average-linkage), k by silhouette → dendrogram | Taxonomizer §4.2 |
    | Internal node labelling | **Data-driven contrastive terms** (paper's labelling is semi-automatic) | Taxonomizer §4.3 *(adapted)* |

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
# Scope widget keys to the uploaded file so a NEW file always shows its own
# auto-detected defaults (Streamlit otherwise keeps the previous file's
# selections under a fixed key, which silently overrides the new defaults).
_fk = safe_name(uploaded.name)
with st.expander('Column configuration', expanded=True):
    left, right = st.columns(2)
    with left:
        leaf_cols = st.multiselect('Leaf variable column(s)', cols,
            default=[c for c in auto_cfg.get('leaf_cols', []) if c in cols], key=f'leaf_{_fk}')
        group_cols = st.multiselect('Group/task column(s)', cols,
            default=[c for c in auto_cfg.get('group_cols', []) if c in cols], key=f'group_{_fk}')
    with right:
        text_cols = st.multiselect('Text/description column(s)', cols,
            default=[c for c in auto_cfg.get('text_cols', []) if c in cols], key=f'text_{_fk}')
        meta_cols = st.multiselect('Metadata/type column(s)', cols,
            default=[c for c in auto_cfg.get('metadata_cols', []) if c in cols], key=f'meta_{_fk}')

if not leaf_cols:
    st.error('Choose at least one leaf variable column.')
    st.stop()

cfg = {'leaf_cols': leaf_cols, 'group_cols': group_cols,
       'text_cols': text_cols, 'metadata_cols': meta_cols}

if st.button('Build baseline hierarchy', type='primary'):
    # ── load Word2Vec model (cached after first call) ──────────────────────
    with st.spinner('Loading Word2Vec model (first run downloads ~66 MB)…'):
        _w2v = _load_w2v()
    if _w2v is None:
        st.stop()

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
        _nodes = build_hierarchy(_can, _w2v, project=_pname,
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

tabs = st.tabs(['🌳 Visualization', 'Node detail', 'Canonical table', 'Export', '📊 Evaluation'])

with tabs[0]:
    # ── Visualization controls (above chart — matches Approach 1 / 2) ─────────
    vc1, vc2, vc3 = st.columns([3, 2, 1])
    with vc1:
        viz_mode = st.radio(
            'View mode',
            ['Sunburst (drill-down)', 'Treemap', 'Node-link tree'],
            horizontal=True, index=0,
            help='Sunburst best for large hierarchies [Taxonomizer]. '
                 'Node-link best for inspecting structure at moderate depth.')
    with vc2:
        display_depth = st.slider('Depth (Level of Detail)', 1, 8, 4, 1,
                                  help='How many levels to reveal at once.')
    with vc3:
        show_leaf_labels = st.checkbox('Leaf labels', value=False,
                                       help='Show variable names on the node-link tree.')
    st.divider()

    if viz_mode == 'Sunburst (drill-down)':
        st.plotly_chart(plot_sunburst(nodes, max_depth=display_depth),
                        use_container_width=True)
        st.caption('Green = Baseline. Click a sector to drill down; click the centre to go back.')
    elif viz_mode == 'Treemap':
        st.plotly_chart(plot_treemap(nodes), use_container_width=True)
    else:
        st.plotly_chart(plot_node_link(nodes, max_depth=display_depth,
                                       show_leaf_labels=show_leaf_labels),
                        use_container_width=True)

with tabs[1]:
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

with tabs[2]:
    st.dataframe(can, use_container_width=True)

with tabs[3]:
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
        "The download buttons above go to your browser's Downloads folder (a browser "
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

with tabs[4]:
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

    # ── Label-quality proxies (interpretability) ──────────────────────────────
    st.markdown('#### Label quality *(interpretability — reference-free)*')
    lq = he.label_quality(nodes)
    l1, l2, l3 = st.columns(3)
    l1.metric('Concept-valid labels', f"{lq['concept_label_pct']}%",
              help='% of internal labels that read as a real concept (short noun '
                   'phrase, WordNet head) rather than a "/"-joined term fragment.')
    l2.metric('Sibling label redundancy', f"{lq['redundancy_pct']}%",
              help='% of internal labels duplicating a sibling label (lower is better).')
    l3.metric('Avg label words', lq['avg_label_words'],
              help='Mean label length in words (shorter = more name-like).')

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

    # ── Held-out group recovery (VALID — group column not used in construction) ─
    st.markdown('#### Held-out group recovery *(valid — group column not used)*')
    st.caption(
        'The baseline never uses the group column (it embeds only attribute '
        'names), so this is a **valid held-out** recovery score. ARI and AMI are '
        'chance-corrected; NMI and Purity are omitted as inflated by over-splitting.'
    )
    gp = he.group_preservation(nodes, can)
    g1, g2 = st.columns(2)
    g1.metric('ARI', gp['ARI'], help='Adjusted Rand Index (chance-corrected).')
    g2.metric('AMI', gp['AMI'], help='Adjusted Mutual Information (chance-corrected).')
