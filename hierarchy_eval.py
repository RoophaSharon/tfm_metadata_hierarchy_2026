"""
hierarchy_eval.py — shared, reference-free hierarchy evaluation for the TFM.

WHY REFERENCE-FREE?
-------------------
In all three approaches the dataset's group column is a *construction input*
(Gonçalves text object in baseline / Approach 1; explicit group-anchored L1/L2
in Approach 2).  An input cannot also serve as the ground truth — measuring the
hierarchy against the group column is therefore circular (for Approach 2 it is
circular by design).  The defensible evaluation is reference-free.

PRIMARY METRICS (no gold standard required) — fair cross-approach comparison
-------------------------------------------
  • Parent–child coherence   — TraCo (Wu et al., AAAI 2024, arXiv:2401.14113)
  • Sibling diversity        — TraCo (same paper)
  • NPMI label coherence     — Lau et al., EACL 2014 (aclanthology.org/E14-1056);
                               orig. Mimno et al., EMNLP 2010
  • Label quality            — interpretability proxies (concept-valid label %,
                               sibling redundancy, avg label words).  Captures the
                               dimension coherence misses (meaningful inner labels,
                               Taxonomizer's stated goal).
  • Structural statistics    — HiExpan-style reporting (Shen et al., KDD 2018)

All of the above use the SAME encoder/corpus for every approach, so the
cross-approach comparison is fair.  NOTE: coherence (TraCo/NPMI) can favour the
data-derived baseline, so interpretability + a human study are needed to show
the approaches' advantage.

GROUP-COLUMN METRICS (ARI / AMI / NMI / Purity) — meaning differs by approach
-----------------------------------------------------------------------------
  • Baseline: the group column is NOT used in construction → this is a VALID
    held-out recovery score (headline ARI / AMI; NMI & Purity inflated).
  • Approach 1 / 2: the group column is a construction input → circular, reported
    only as a self-consistency check (expected high), NOT comparable to baseline.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Tree helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_parent_map(nodes: list) -> dict:
    pm: dict = {}
    for n in nodes:
        for c in n.get('related', []):
            cid = int(c)
            if cid not in pm:
                pm[cid] = int(n['id'])
    return pm


def structural_stats(nodes: list) -> dict:
    pm = build_parent_map(nodes)

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


# ──────────────────────────────────────────────────────────────────────────────
# Encoder — SBERT if available, TF-IDF fallback.  Loaded once, reused.
# ──────────────────────────────────────────────────────────────────────────────
_SBERT = None
_SBERT_TRIED = False


def _get_sbert():
    global _SBERT, _SBERT_TRIED
    if _SBERT_TRIED:
        return _SBERT
    _SBERT_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        _SBERT = None
    return _SBERT


def encode(texts: list):
    """Return (unit-normalised vectors, backend_name)."""
    texts = [str(t) if str(t).strip() else '_' for t in texts]
    model = _get_sbert()
    if model is not None:
        v = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=float), 'SBERT (all-MiniLM-L6-v2)'
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words='english', max_features=2000,
                        min_df=1).fit_transform(texts).toarray().astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms), 'TF-IDF (SBERT unavailable)'


# ──────────────────────────────────────────────────────────────────────────────
# TraCo reference-free metrics  (Wu et al., AAAI 2024)
# ──────────────────────────────────────────────────────────────────────────────
def traco_metrics(nodes: list) -> dict:
    """Parent–child coherence and sibling diversity over node *labels*."""
    usable = [n for n in nodes if n.get('type') in ('aggregation', 'attribute')]
    if len(usable) < 2:
        return {'pc_coherence': 0.0, 'sibling_diversity': 0.0, 'encoder': 'n/a'}

    ids    = [int(n['id']) for n in usable]
    labels = [str(n.get('name', '')) for n in usable]
    vecs, backend = encode(labels)
    id2v = {i: vecs[k] for k, i in enumerate(ids)}

    pc_sims, sib_divs = [], []
    for n in nodes:
        if n.get('type') == 'root':
            continue
        pid = int(n['id'])
        if pid not in id2v:
            continue
        children = [int(c) for c in n.get('related', []) if int(c) in id2v]
        for cid in children:
            pc_sims.append(float(np.dot(id2v[pid], id2v[cid])))
        if len(children) >= 2:
            cv = np.array([id2v[c] for c in children])
            S  = cv @ cv.T
            nc = len(children)
            divs = [1.0 - float(S[i, j]) for i in range(nc) for j in range(i + 1, nc)]
            sib_divs.append(float(np.mean(divs)))

    return {
        'pc_coherence':      round(float(np.mean(pc_sims)),  4) if pc_sims  else 0.0,
        'sibling_diversity': round(float(np.mean(sib_divs)), 4) if sib_divs else 0.0,
        'encoder':           backend,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NPMI label coherence  (Lau et al., EACL 2014; Mimno et al., EMNLP 2010)
# Reference corpus = the variable descriptions themselves.
# ──────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r'[a-z][a-z]{2,}')
_STOP = set(
    'the a an and or of to in for on with by at from as is are be this that these '
    'those it its was were has have had not no than then so such can will may '
    'group description name label value type using used per each'.split()
)


def _tokens(text: str) -> set:
    return {w for w in _TOKEN_RE.findall(str(text).lower()) if w not in _STOP}


def npmi_coherence(nodes: list, corpus_texts: list, topn: int = 5) -> float:
    """Average NPMI of each aggregation node's label terms over the corpus.

    Returns a value in roughly [-1, 1]; higher = node labels use term
    combinations that genuinely co-occur in the data (meaningful, not random).
    """
    docs = [_tokens(t) for t in corpus_texts]
    docs = [d for d in docs if d]
    N = len(docs)
    if N < 2:
        return 0.0

    df: Counter = Counter()
    for d in docs:
        for w in d:
            df[w] += 1

    # Collect the term sets we actually need (node labels)
    label_termsets: list = []
    needed_terms: set = set()
    for n in nodes:
        if n.get('type') != 'aggregation':
            continue
        terms = [w for w in _tokens(n.get('name', '')) if df.get(w, 0) > 0]
        terms = sorted(terms, key=lambda w: df[w], reverse=True)[:topn]
        if len(terms) >= 2:
            label_termsets.append(terms)
            needed_terms.update(terms)

    if not label_termsets:
        return 0.0

    # Pair co-occurrence counts (only for needed pairs)
    needed_pairs = set()
    for terms in label_termsets:
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                needed_pairs.add(frozenset((terms[i], terms[j])))

    co: Counter = Counter()
    for d in docs:
        present = d & needed_terms
        if len(present) < 2:
            continue
        pl = list(present)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                pair = frozenset((pl[i], pl[j]))
                if pair in needed_pairs:
                    co[pair] += 1

    eps = 1e-12
    node_scores: list = []
    for terms in label_termsets:
        pair_npmis: list = []
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                wi, wj = terms[i], terms[j]
                c_ij = co.get(frozenset((wi, wj)), 0)
                p_ij = (c_ij + eps) / N
                p_i  = df[wi] / N
                p_j  = df[wj] / N
                pmi  = np.log(p_ij / (p_i * p_j + eps) + eps)
                npmi = pmi / (-np.log(p_ij + eps))
                pair_npmis.append(float(npmi))
        if pair_npmis:
            node_scores.append(float(np.mean(pair_npmis)))

    return round(float(np.mean(node_scores)), 4) if node_scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Secondary (descriptive, caveated): group-structure preservation
# ──────────────────────────────────────────────────────────────────────────────
def _depth1_assignments(nodes: list, can) -> list:
    pm = build_parent_map(nodes)

    def depth1(nid: int) -> int:
        while pm.get(nid, -1) not in (-1, 0):
            nid = pm[nid]
        return nid

    lid_to_nid = {n['metadata']['leaf_id']: int(n['id'])
                  for n in nodes if n.get('type') == 'attribute' and 'metadata' in n}
    return [depth1(lid_to_nid[lid]) if lid in lid_to_nid else -1
            for lid in can['_leaf_id']]


def _purity(y_true, y_pred) -> float:
    clusters: dict = {}
    for t, p in zip(y_true, y_pred):
        clusters.setdefault(p, []).append(t)
    correct = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct / max(len(y_true), 1)


def group_preservation(nodes: list, can) -> dict:
    """NMI / ARI / Purity of the depth-1 partition vs the group column.

    CAVEAT: the group column is a construction input in every approach, so this
    is a descriptive 'structure preservation' figure, NOT an accuracy metric.
    """
    from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                                 adjusted_mutual_info_score)
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    # group column robust to either canonical schema (_group_path or _group)
    gcol = '_group_path' if '_group_path' in can.columns else '_group'
    y_true_raw = can[gcol].apply(
        lambda x: str(x).split(' > ')[0].strip()
        if pd.notna(x) and str(x) not in ('', 'nan') else 'Ungrouped'
    ).tolist()
    y_pred_raw = _depth1_assignments(nodes, can)

    y_true = LabelEncoder().fit_transform(y_true_raw)
    y_pred = LabelEncoder().fit_transform(y_pred_raw)
    return {
        # ARI and AMI are chance-corrected — the trustworthy numbers.
        'ARI':    round(float(adjusted_rand_score(y_true, y_pred)), 4),
        'AMI':    round(float(adjusted_mutual_info_score(y_true, y_pred)), 4),
        # NMI and Purity are reported for completeness but are inflated by
        # over-splitting (more clusters → higher), so they are NOT headline.
        'NMI':    round(float(normalized_mutual_info_score(
                     y_true, y_pred, average_method='arithmetic')), 4),
        'Purity': round(_purity(y_true_raw, y_pred_raw), 4),
    }

def label_quality(nodes: list) -> dict:
    """Reference-free interpretability proxies for internal-node labels.

    Captures the dimension Taxonomizer is *about* — meaningful inner-node labels —
    which coherence metrics miss.  Fully automatic, no gold standard:

      • concept_label_pct  — % of internal labels that read as a real concept:
        a short phrase (<=3 words) whose head word is a known English noun
        (WordNet).  Penalises '/'-joined contrastive term fragments.
      • redundancy_pct     — % of internal labels that duplicate a sibling's
        label (same normalised text under the same parent).
      • avg_label_words    — mean label length in words (shorter = more name-like).
    """
    pm = build_parent_map(nodes)
    internal = [n for n in nodes if n.get('type') == 'aggregation']
    if not internal:
        return {'concept_label_pct': 0.0, 'redundancy_pct': 0.0, 'avg_label_words': 0.0}

    # WordNet noun check (optional; degrade gracefully if unavailable)
    try:
        from nltk.corpus import wordnet as wn
        def _is_noun(w):
            return bool(wn.synsets(w, pos=wn.NOUN))
    except Exception:
        def _is_noun(w):
            return len(w) > 2  # fallback: any real-ish word

    def _norm(s): return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

    concept = 0
    wordcounts = []
    for n in internal:
        raw = str(n.get('name', ''))
        words = _norm(raw).split()
        wordcounts.append(len(words))
        # '/'-joined fragments are NOT concept labels
        is_fragment = '/' in raw
        head = words[-1] if words else ''
        if (not is_fragment) and 1 <= len(words) <= 3 and head and _is_noun(head):
            concept += 1

    # sibling redundancy
    by_parent: dict = {}
    for n in internal:
        p = pm.get(int(n['id']), -1)
        by_parent.setdefault(p, []).append(_norm(n.get('name', '')))
    redundant = 0
    for sibs in by_parent.values():
        seen = set()
        for s in sibs:
            if s in seen:
                redundant += 1
            seen.add(s)

    n_int = len(internal)
    return {
        'concept_label_pct': round(100.0 * concept / n_int, 1),
        'redundancy_pct':    round(100.0 * redundant / n_int, 1),
        'avg_label_words':   round(float(np.mean(wordcounts)), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Gold-standard comparison — Edge-F1 / Ancestor-F1
#
# HiExpan (Shen et al., KDD 2018) scores a system taxonomy against a hand-built
# gold taxonomy with Edge-F1 (direct parent–child links) and Ancestor-F1 (all
# ancestor links).  Because our internal-node *labels* differ between the gold
# tree and each system, we use the label-free leaf-pair formulation (the
# pair-counting tradition, Fowlkes & Mallows 1983):
#
#   • Edge-F1     — over pairs of leaves that share the same IMMEDIATE parent
#                   (i.e. they are siblings).  Strict: rewards correct granularity.
#   • Ancestor-F1 — over pairs of leaves that share ANY non-root ancestor
#                   (i.e. they are grouped together somewhere).  Lenient.
#
# Leaves are matched between gold and system by their attribute-node NAME (the
# variable label) — the one field all three approaches expose for every leaf.
# Only leaves present in BOTH the gold subset and the system tree are scored, so
# a gold subset of 50–100 variables fairly evaluates a full hierarchy.
# ──────────────────────────────────────────────────────────────────────────────
def _pred_leaf_lineage(nodes: list) -> dict:
    """leaf name → list of ancestor node ids (root-most first, excl. root & leaf)."""
    pm = build_parent_map(nodes)
    id_to_node = {int(n['id']): n for n in nodes}
    lineage: dict = {}
    for n in nodes:
        if n.get('type') != 'attribute':
            continue
        name = str(n.get('name', ''))
        cur  = int(n['id'])
        anc, seen = [], set()
        while cur in pm and cur not in seen:
            seen.add(cur)
            cur = pm[cur]
            nd = id_to_node.get(cur)
            if nd is None or nd.get('type') == 'root':
                break
            anc.append(cur)
        anc.reverse()
        lineage[name] = anc
    return lineage


def _gold_leaf_lineage(gold_df) -> dict:
    """leaf name → list of cumulative path-prefix strings (the gold ancestors)."""
    lineage: dict = {}
    for _, r in gold_df.iterrows():
        name = str(r['leaf_label'])
        path = str(r.get('gold_path', '') or '')
        comps = [c.strip() for c in path.split('>')
                 if c.strip() and c.strip().lower() != 'ungrouped']
        anc, pref = [], ''
        for c in comps:
            pref = c if not pref else f'{pref} > {c}'
            anc.append(pref)
        lineage[name] = anc
    return lineage


def _sibling_pairs(lineage: dict) -> set:
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for name, anc in lineage.items():
        if anc:
            groups[anc[-1]].append(name)
    pairs: set = set()
    for members in groups.values():
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pairs.add((m[i], m[j]))
    return pairs


def _cogrouped_pairs(lineage: dict) -> set:
    from collections import defaultdict
    occ: dict = defaultdict(set)
    for name, anc in lineage.items():
        for a in anc:
            occ[a].add(name)
    pairs: set = set()
    for members in occ.values():
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pairs.add((m[i], m[j]))
    return pairs


def _prf(pred_set: set, gold_set: set) -> dict:
    if not pred_set and not gold_set:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4)}


def gold_comparison(nodes: list, gold_df) -> dict:
    """Edge-F1 and Ancestor-F1 of a system tree vs a hand-built gold tree."""
    pred = _pred_leaf_lineage(nodes)
    gold = _gold_leaf_lineage(gold_df)
    shared = set(pred) & set(gold)
    pred = {k: v for k, v in pred.items() if k in shared}
    gold = {k: v for k, v in gold.items() if k in shared}
    return {
        'n_matched_leaves': len(shared),
        'edge_f1':     _prf(_sibling_pairs(pred),   _sibling_pairs(gold)),
        'ancestor_f1': _prf(_cogrouped_pairs(pred), _cogrouped_pairs(gold)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Granularity-tolerant, label-independent structural F1  (set-overlap matching)
#
# Edge-F1 punishes a system for adding *correct* extra depth, because two leaves
# that gold lists as siblings stop being immediate siblings once the system
# refines them into sub-tiers.  That makes edge-F1 unfair to deliberately deeper
# trees (Approaches 1 & 2).  Set-overlap F1 fixes this: it matches each gold
# cluster (the set of leaves under a gold path-prefix) to the system node whose
# leaf set overlaps it most (Jaccard), regardless of that node's depth or label.
#
#   • precision — for each system aggregation node, its best Jaccard with any
#                 gold cluster, averaged.  Low when the system invents groups
#                 gold does not have (e.g. one node per delay value = over-split).
#   • recall    — for each gold cluster, its best Jaccard with any system node,
#                 averaged.  Low when the system fails to recover a gold group.
#
# This is the cluster-matching / overlap-F1 tradition (e.g. ontology alignment,
# hierarchical-clustering evaluation).  Label-free, so it compares the three
# approaches fairly even though their internal-node labels differ.
# ──────────────────────────────────────────────────────────────────────────────
def _system_clusters(nodes: list) -> list:
    """Each aggregation node → frozenset of leaf NAMES in its subtree (size ≥ 2)."""
    id_to_node = {int(n['id']): n for n in nodes}
    out: list = []
    for n in nodes:
        if n.get('type') != 'aggregation':
            continue
        leaves: list = []
        stack = [int(n['id'])]
        seen: set = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            nd = id_to_node.get(x)
            if nd is None:
                continue
            if nd.get('type') == 'attribute':
                leaves.append(str(nd.get('name', '')))
            else:
                stack.extend(int(c) for c in nd.get('related', []))
        s = frozenset(leaves)
        if len(s) >= 2:
            out.append(s)
    return out


def _gold_clusters(gold_df) -> list:
    """Each gold path-prefix → frozenset of leaf NAMES under it (size ≥ 2)."""
    from collections import defaultdict
    occ: dict = defaultdict(set)
    for name, anc in _gold_leaf_lineage(gold_df).items():
        for a in anc:
            occ[a].add(name)
    return [frozenset(v) for v in occ.values() if len(v) >= 2]


def set_overlap_f1(nodes: list, gold_df) -> dict:
    """Granularity-tolerant, label-free hierarchical F1 via best leaf-set Jaccard."""
    pred_names = set(_pred_leaf_lineage(nodes))
    gold_names = {str(x) for x in gold_df['leaf_label']}
    shared = pred_names & gold_names
    if len(shared) < 2:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    sys_cl  = [c & shared for c in _system_clusters(nodes)]
    sys_cl  = [c for c in sys_cl if len(c) >= 2]
    gold_cl = [c & shared for c in _gold_clusters(gold_df)]
    gold_cl = [c for c in gold_cl if len(c) >= 2]
    if not sys_cl or not gold_cl:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def jac(a: frozenset, b: frozenset) -> float:
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    prec = float(np.mean([max(jac(s, g) for g in gold_cl) for s in sys_cl]))
    rec  = float(np.mean([max(jac(s, g) for s in sys_cl) for g in gold_cl]))
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)}


def refinement_breakdown(nodes: list, gold_df) -> dict:
    """Decompose edge-F1 disagreements into harmless refinement vs real errors.

    • wrong_merge_rate — system sibling pairs that gold does NOT co-group anywhere
      (genuine mistakes: variables wrongly placed together).
    • refinement_rate  — gold sibling pairs the system keeps co-grouped but at a
      FINER level (split into sub-tiers).  These are deeper-but-consistent, the
      thing edge-F1 unfairly penalises.
    • missed_rate      — gold sibling pairs the system fails to co-group at all
      (real recall failures).
    """
    pred = _pred_leaf_lineage(nodes)
    gold = _gold_leaf_lineage(gold_df)
    shared = set(pred) & set(gold)
    pred = {k: v for k, v in pred.items() if k in shared}
    gold = {k: v for k, v in gold.items() if k in shared}

    sys_sib = _sibling_pairs(pred)
    sys_cog = _cogrouped_pairs(pred)
    gold_sib = _sibling_pairs(gold)
    gold_cog = _cogrouped_pairs(gold)

    wrong_merge = len(sys_sib - gold_cog)
    refined     = len((gold_sib & sys_cog) - sys_sib)
    missed      = len(gold_sib - sys_cog)
    return {
        'wrong_merge_rate': round(wrong_merge / len(sys_sib), 4) if sys_sib else 0.0,
        'refinement_rate':  round(refined / len(gold_sib), 4) if gold_sib else 0.0,
        'missed_rate':      round(missed / len(gold_sib), 4) if gold_sib else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# One-call bundle
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(nodes: list, corpus_texts: list | None = None, can=None,
             gold_df=None) -> dict:
    """Compute the full metric bundle for one hierarchy."""
    out: dict = {}
    out.update(traco_metrics(nodes))
    out['npmi_coherence'] = (npmi_coherence(nodes, corpus_texts)
                             if corpus_texts is not None else None)
    out.update({f'struct_{k}': v for k, v in structural_stats(nodes).items()})
    if can is not None:
        out['group_preservation'] = group_preservation(nodes, can)
    if gold_df is not None:
        out['gold'] = gold_comparison(nodes, gold_df)
    return out
