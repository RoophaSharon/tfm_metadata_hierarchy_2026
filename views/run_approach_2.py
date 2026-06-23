# approach_2.py — Approach 2: Dataset-Constrained Multi-Aspect Hierarchy
#
# ALGORITHM (no domain hardcoding):
#
#   Step 1  Build metadata text objects                                  [GON §3]
#   Step 2  Group-anchored L1/L2 from detected _group metadata          [BISE-26]
#   Step 3  Per terminal group — routing in priority order:
#           (a) Phrase-slot mining               [IE / slot induction]
#               decomposes one variable into multiple alternative-phrase
#               signals before clustering — fixes the document-level ceiling
#               that NMF/BERTopic/FASTopic share.
#           (b) FASTopic semantic aspect discovery [Wu et al. NeurIPS 2024]
#               transformer-based Dual Semantic-relation Reconstruction with
#               optimal transport — recent SOTA replacement for NMF.
#           (c) NMF lexical fallback             [ZHU §3.1 adapted]
#               retained for small groups or when FASTopic is unavailable.
#   Step 4  Per-aspect variable representations                          [ZHU §3.1]
#   Step 5  Independent per-aspect clustering: GMM+BIC small / KMeans large [ZHU §3.2]
#   Step 6  Top-down LoD tree (simplified silhouette best-aspect split)  [ZHU §3.3 adapted]
#   Step 7  Node labeling — deterministic by default:
#           description-prefix → group anchor → IDF + FIELD_NAME filter →
#           bigram-preferred discriminative TF-IDF suffix
#           OPTIONAL: constrained LLM re-phrasing                         [TopicTag, DocEng 2024]
#                     — every label word must appear in evidence (grounding check)
#                     — provenance stored per node (audit trail)
#   Step 8  Quality metrics: NMI/ARI/Purity + parent-child coherence + sibling diversity
#                                                                        [TraCo, AAAI 2024]
#
# Facet trees (Castanet, 2007) removed in this version — a single coherent LoD tree.
#
# PAPERS:
#   [ZHU]      Zhu et al. (2025). EMNLP 2025.   Main scaffold (adapted)
#   [FASTopic] Wu et al. (2024). NeurIPS 2024.  Semantic aspect discovery
#   [GON]      Gonçalves et al. (2019). ESWC.   Canonical text objects
#   [TopicTag] Eren et al. (2024). DocEng.      Constrained LLM label refinement
#   [TraCo]    Wu et al. (2024). AAAI.          Affinity + diversity metrics
#   [TICL]     Kejriwal et al. (2022). EAAI.    NMI/ARI/Purity evaluation
#   [BISE-26]  Motamedi et al. (2026). BISE.    Group-anchored entry validation
#   [IE-Slot]  IE / slot-induction literature (surveyed Xu et al., FCS 2024).

from __future__ import annotations
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAIClient
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False

# Ollama defaults (overridable via env vars OLLAMA_URL / OLLAMA_MODEL).
OLLAMA_URL_DEFAULT   = 'http://localhost:11434/v1'
OLLAMA_MODEL_DEFAULT = 'qwen2.5:3b-instruct'
GROQ_URL_DEFAULT     = 'https://api.groq.com/openai/v1'
GROQ_MODEL_DEFAULT   = 'qwen/qwen3-32b'

def _ping_ollama(base_url: str = OLLAMA_URL_DEFAULT, timeout: float = 1.5) -> bool:
    """Quick reachability check for the local Ollama server."""
    if not _LLM_CLIENT_AVAILABLE:
        return False
    try:
        import urllib.request as _urlreq
        # /v1/models is OpenAI-compat; Ollama also exposes /api/tags
        with _urlreq.urlopen(base_url.rstrip('/v1') + '/api/tags',
                              timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

def _make_llm_client(provider: str, base_url: str, api_key: str = '') -> Optional[object]:
    """
    Build an OpenAI-compatible client for either local Ollama or cloud Groq.

    Both providers expose an OpenAI-compatible REST endpoint, so the same
    openai.OpenAI client class works for both — only the base_url and
    auth differ.
    """
    if not _LLM_CLIENT_AVAILABLE:
        return None
    if provider == 'groq':
        if not api_key:
            return None
        return _OpenAIClient(base_url=base_url, api_key=api_key)
    # Ollama ignores the key but the SDK requires a non-empty string
    if not _ping_ollama(base_url):
        return None
    return _OpenAIClient(base_url=base_url, api_key='ollama-local')

def _parse_json_response(raw: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.

    Handles the response shapes seen across providers/models:
      • plain JSON:                  {"role": "measure"}
      • markdown-fenced:             ```json\n{...}\n```
      • reasoning-model preamble:    <think>...</think>\n{...}
        (Qwen3 / DeepSeek-R1 style models emit thinking traces before the
        answer when JSON mode is unavailable — e.g. qwen/qwen3-32b on Groq)
      • prose then JSON:             Here is the result: {...}

    Strategy: strip <think> blocks, then take the substring from the first
    '{' to the last '}' and parse it.  Raises ValueError if no JSON found.
    """
    s = (raw or '').strip()
    if not s:
        raise ValueError('empty response')
    # Strip reasoning blocks (Qwen3 / R1 style)
    while '<think>' in s:
        start = s.find('<think>')
        end   = s.find('</think>', start)
        if end == -1:
            # Unclosed think block — drop everything from <think> onward,
            # the JSON (if any) would be before it
            s = s[:start]
            break
        s = s[:start] + s[end + len('</think>'):]
    s = s.strip()
    # Take first '{' .. last '}' — covers fences and prose prefixes/suffixes
    i, j = s.find('{'), s.rfind('}')
    if i == -1 or j == -1 or j <= i:
        raise ValueError('no JSON object in response')
    return json.loads(s[i:j + 1])

def _safe_chat_completion(client, model: str, prompt: str,
                            max_tokens: int = 200, temperature: float = 0.1):
    """
    Call an OpenAI-compatible chat completion endpoint with automatic
    fallback for providers that don't support JSON mode on a given model.

    Some models on Groq (notably some Qwen 3 variants) reject
    `response_format={"type":"json_object"}` with HTTP 400 BadRequestError.
    This wrapper first tries WITH JSON mode (better reliability when
    supported), and if the provider rejects it with a bad-request error,
    retries WITHOUT.  Prompts in this codebase already say 'Output JSON only'
    and we strip ```json fences after parsing, so the retry path still
    works deterministically.
    """
    base_args = {
        'model':       model,
        'messages':    [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens':  max_tokens,
    }
    try:
        return client.chat.completions.create(
            **base_args, response_format={'type': 'json_object'})
    except Exception as e:
        # Retry without JSON mode on bad-request / unsupported-feature errors
        err_name = type(e).__name__
        err_text = str(e)
        if ('BadRequest' in err_name or '400' in err_text
                or 'response_format' in err_text):
            return client.chat.completions.create(**base_args)
        raise

try:
    from fastopic import FASTopic                # type: ignore[import-not-found]
    _FASTOPIC_AVAILABLE = True
except ImportError:
    _FASTOPIC_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# ROLE KEYS  (domain-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
LEAF_KEYS  = 'variable var field column attribute name code id item indicator question measure concept'.split()
GROUP_KEYS = 'task category domain module section table dataset assessment test variant group topic instrument form subscale construct'.split()
TEXT_KEYS  = 'description definition desc label title question meaning note notes text display full details explanation'.split()
META_KEYS  = 'type dtype data_type datatype unit units format decimal precision values value coding range min max scale'.split()

# ──────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ──────────────────────────────────────────────────────────────────────────────
def safe_name(n: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in n)

def try_read_csv(path: Path) -> pd.DataFrame:
    best, best_score = None, -1
    # Try explicit comma first (most common), then let Python sniff, then other separators.
    # Reject results with only 1 column — likely a parsing failure.
    for enc in ['utf-8-sig', 'utf-8', 'latin1']:
        for sep in [',', '\t', ';', '|', None]:
            try:
                df = pd.read_csv(path, sep=sep, engine='python', encoding=enc,
                                 on_bad_lines='skip')
                if df.shape[1] < 2:
                    continue
                s = df.shape[1] * 10 - float(df.isna().mean().mean())
                if s > best_score:
                    best, best_score = df, s
            except Exception:
                pass
    if best is None:
        raise ValueError(f'Could not read {path.name}')
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
    raise ValueError(f'Unsupported: {s}')

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN ROLE DETECTION  (domain-agnostic heuristic scoring)
# ──────────────────────────────────────────────────────────────────────────────
def _norm(c: str) -> str:
    return ''.join(ch if ch.isalnum() else ' ' for ch in str(c).lower())

def _ks(c: str, keys: list) -> int:
    return sum(1 for k in keys if k in _norm(c))

def detect_roles(df: pd.DataFrame) -> dict:
    n = max(len(df), 1)
    out = []
    for col in df.columns:
        s = df[col]
        nun = int(s.nunique(dropna=True))
        ur  = nun / n
        avg = float(s.dropna().astype(str).map(len).mean()) if s.notna().any() else 0
        # Raised group uniqueness ceiling from 80 → 300 so large datasets (e.g. HCP
        # with 100+ assessment categories) are not excluded.
        out.append({
            'column':      str(col),
            'leaf_score':  4 * _ks(col, LEAF_KEYS)  + (3 if 0.5 <= ur <= 1 else 0),
            'group_score': 4 * _ks(col, GROUP_KEYS) + (3 if 1 < nun < min(n * 0.5, 300) else 0),
            'text_score':  5 * _ks(col, TEXT_KEYS)  + (4 if avg > 50 else 0),
            'meta_score':  4 * _ks(col, META_KEYS)  + (2 if 1 < nun < min(n * 0.8, 100) else 0),
        })
    prof  = pd.DataFrame(out)
    leaf  = prof.sort_values('leaf_score', ascending=False).head(1)['column'].tolist()
    text  = (prof[prof.text_score >= 4]
             .sort_values('text_score', ascending=False)['column'].tolist()) or leaf[:]
    group = (prof[(prof.group_score >= 4) & (~prof.column.isin(leaf))]
             .sort_values('group_score', ascending=False).head(3)['column'].tolist())
    meta  = (prof[(prof.meta_score >= 4) & (~prof.column.isin(text + leaf + group))]
             .sort_values('meta_score', ascending=False).head(4)['column'].tolist())
    return {'leaf_cols': leaf, 'group_cols': group, 'text_cols': text, 'meta_cols': meta}

def sv(x) -> str:
    return '' if pd.isna(x) else str(x).strip()

def build_canonical(df: pd.DataFrame, cfg: dict, source: str) -> pd.DataFrame:
    """Build normalised per-variable rows with a unified _text field [GON §3]."""
    leaf_cols  = cfg.get('leaf_cols', [])
    group_cols = cfg.get('group_cols', [])
    text_cols  = cfg.get('text_cols', [])
    meta_cols  = cfg.get('meta_cols', [])
    rows = []
    for i, row in df.iterrows():
        label = (' / '.join(p for p in [sv(row.get(c, '')) for c in leaf_cols] if p)
                 or f'var_{i}')
        group = (' > '.join(p for p in [sv(row.get(c, '')) for c in group_cols]
                             if p and p.lower() not in ['nan', 'none'])
                 or 'Ungrouped')
        all_cols = list(dict.fromkeys(group_cols + leaf_cols + text_cols + meta_cols))
        text = ' | '.join(f'{c}: {sv(row.get(c, ""))}' for c in all_cols
                          if sv(row.get(c, '')))
        rows.append({'_source': source, '_row': int(i), '_label': label,
                     '_id': f'{group}.{label}', '_group': group, '_text': text})
    can = pd.DataFrame(rows)
    cnt: dict = defaultdict(int)
    ids = []
    for lid in can['_id']:
        cnt[lid] += 1
        ids.append(lid if cnt[lid] == 1 else f'{lid}__{cnt[lid]}')
    can['_id'] = ids
    return can

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  — ASPECT DISCOVERY VIA NMF  [ZHU §3.1 adapted]
# ──────────────────────────────────────────────────────────────────────────────
def _elbow_k(errors: list, k_range: range) -> int:
    if len(errors) <= 2:
        return list(k_range)[0]
    diffs = np.diff(errors)
    drops = np.diff(diffs)
    if drops.max() - drops.min() < 1e-8:
        return max(2, int(np.sqrt(len(errors))))
    elbow_idx = int(np.argmax(drops)) + 1
    return list(k_range)[min(elbow_idx, len(k_range) - 1)]

def discover_aspects(texts: list, max_aspects: int = 10):
    """
    Discover K latent semantic aspects via NMF on TF-IDF  [ZHU §3.1 adapted].

    Replaces Zhu et al.'s LLM aspect generation with NMF (deterministic, no
    hallucination).  K is selected by reconstruction-error elbow.

    Returns tfidf, X, nmf, W, H, K, labels.
    """
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                            max_features=3000, min_df=1)
    X     = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    n_vars = X.shape[0]

    k_range = range(2, min(max_aspects + 1, n_vars // 2 + 1, 16))
    if len(k_range) < 2:
        k_range = range(2, 3)

    errors = []
    for k in k_range:
        m = NMF(n_components=k, random_state=42, max_iter=400, init='nndsvda')
        m.fit_transform(X)
        errors.append(m.reconstruction_err_)

    K   = _elbow_k(errors, k_range)
    nmf = NMF(n_components=K, random_state=42, max_iter=400, init='nndsvda')
    W   = nmf.fit_transform(X)
    H   = nmf.components_

    # Aspect labels: top-4 terms per NMF component
    labels = []
    for k in range(K):
        top_idx = np.argsort(H[k])[-4:][::-1]
        labels.append(' / '.join(terms[top_idx]))

    return tfidf, X, nmf, W, H, K, labels

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 (FASTopic variant)  — semantic aspect discovery  [Wu et al. NeurIPS 2024]
# ──────────────────────────────────────────────────────────────────────────────
def discover_aspects_fastopic(texts: list,
                               max_aspects: int = 10,
                               fallback_tfidf: bool = True):
    """
    Recent SOTA semantic aspect discovery via FASTopic [Wu et al. NeurIPS 2024,
    arXiv:2405.17978].

    FASTopic uses a pretrained Transformer (SBERT) to embed documents, then
    learns topic/word embeddings via Dual Semantic-relation Reconstruction
    (DSR) with optimal transport.  Beats NMF, BERTopic, and CombinedTM on
    standard topic benchmarks — semantic, not lexical, and reproducible.

    Adapter shape matches `discover_aspects` so it is a drop-in replacement:
        returns (tfidf, X, model, W, H, K, labels)
    `tfidf` and `X` are still produced (used downstream by label_cluster +
    masked TF-IDF representation fallbacks); FASTopic provides W (doc-topic),
    H_proxy (topic-term scores derived from top words), K, and labels.

    Falls back to NMF if FASTopic is not installed or the group is too small
    for transformer training (< 6 documents).
    """
    n_vars = len(texts)
    # Keep a TF-IDF matrix available for downstream code paths
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                             max_features=3000, min_df=1)
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())

    if not _FASTOPIC_AVAILABLE or n_vars < 6:
        if fallback_tfidf:
            return discover_aspects(texts, max_aspects)
        raise RuntimeError('FASTopic unavailable and TF-IDF fallback disabled')

    # FASTopic K selection: bounded similar to NMF elbow but simpler — pick a
    # reasonable K from corpus size (avoids overfitting tiny groups).
    K = max(2, min(max_aspects, int(np.ceil(np.sqrt(n_vars))) + 1, n_vars - 1))

    try:
        model = FASTopic(num_topics=K,
                          doc_embed_model='all-MiniLM-L6-v2',
                          verbose=False)
        # fit_transform returns (top_words_per_topic, doc_topic_dist)
        result = model.fit_transform(texts)
        if isinstance(result, tuple) and len(result) == 2:
            top_words, doc_topic = result
        else:
            # Some FASTopic versions return only doc_topic; pull top words via API
            doc_topic = result
            top_words = [model.get_top_words(topic_id=k, num_top_words=10)
                         for k in range(K)]
    except Exception:
        # Robust fallback if FASTopic fails (small corpus, OOM, etc.)
        if fallback_tfidf:
            return discover_aspects(texts, max_aspects)
        raise

    W = np.asarray(doc_topic, dtype=np.float32)
    if W.ndim != 2 or W.shape[0] != n_vars:
        if fallback_tfidf:
            return discover_aspects(texts, max_aspects)

    # Build H_proxy: K × n_terms with weight = position-decay of each top word
    n_terms = len(terms)
    term_to_idx = {t: i for i, t in enumerate(terms)}
    H_proxy = np.zeros((K, n_terms), dtype=np.float32)
    labels  = []
    for k in range(K):
        words_k = top_words[k] if k < len(top_words) else []
        # Each entry may be 'word', or 'word score', or (word, score)
        clean: list = []
        for w in words_k:
            if isinstance(w, (list, tuple)):
                w = w[0]
            w = str(w).split(' ')[0].strip().lower()
            if w:
                clean.append(w)
        for rank, w in enumerate(clean):
            if w in term_to_idx:
                H_proxy[k, term_to_idx[w]] += 1.0 / (rank + 1)
        labels.append(' / '.join(clean[:4]) if clean else f'aspect {k+1}')

    return tfidf, X, model, W, H_proxy, K, labels

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  — PER-ASPECT VARIABLE REPRESENTATIONS  [ZHU §3.1]
# ──────────────────────────────────────────────────────────────────────────────
def per_aspect_representations(texts: list, H: np.ndarray,
                                tfidf: TfidfVectorizer,
                                sbert_model=None) -> list:
    """
    Build K independent representation matrices — one per aspect  [ZHU §3.1].

    For each aspect k:
      • identify top-T terms from H[k]
      • filter variable texts to those terms → encode with SBERT (or masked TF-IDF)

    Returns list of K arrays, each shape (n_vars, embed_dim).
    """
    terms  = np.array(tfidf.get_feature_names_out())
    X_arr  = tfidf.transform(texts).toarray()
    K      = H.shape[0]
    T      = min(30, len(terms))
    reprs  = []

    for k in range(K):
        top_idx   = np.argsort(H[k])[-T:]
        top_terms = set(terms[top_idx])

        if sbert_model is not None:
            filtered = []
            for txt in texts:
                tokens = txt.lower().split()
                kept   = ' '.join(t for t in tokens if t in top_terms)
                filtered.append(kept if kept.strip() else txt)
            emb = sbert_model.encode(filtered, show_progress_bar=False,
                                     batch_size=64, normalize_embeddings=True)
        else:
            mask  = H[k]
            emb   = X_arr * mask[np.newaxis, :]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

        reprs.append(emb.astype(np.float32))

    return reprs

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5  — INDEPENDENT PER-ASPECT CLUSTERING  [ZHU §3.2]
# ──────────────────────────────────────────────────────────────────────────────
def cluster_aspect_gmm(emb: np.ndarray, max_k: int = 8, fast_threshold: int = 30):
    """
    Cluster variables within one aspect space  [ZHU §3.2].

    Hybrid strategy:
      • n ≤ fast_threshold → GMM + BIC (accurate, recommended for small clusters)
      • n  > fast_threshold → MiniBatchKMeans + silhouette selection
        (engineering adaptation for large groups, e.g. HCP Cognition / FreeSurfer)

    Both paths use diagonal covariance / SVD pre-reduction for numerical
    stability on high-dimensional sparse embeddings.
    """
    n = emb.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), 1, 0.0

    d_target = min(20, emb.shape[1], n - 1)
    if emb.shape[1] > d_target:
        svd   = TruncatedSVD(n_components=d_target, random_state=42)
        emb_r = svd.fit_transform(emb)
    else:
        emb_r = emb.copy()

    best_score, best_labels, best_k = -np.inf, None, 2

    if n > fast_threshold:
        # Fast path: MiniBatchKMeans + silhouette  (large groups)
        for k in range(2, min(max_k + 1, n)):
            try:
                km     = MiniBatchKMeans(n_clusters=k, random_state=42,
                                          n_init=3, batch_size=min(256, n),
                                          max_iter=100)
                labels = km.fit_predict(emb_r)
                if len(set(labels)) < 2:
                    continue
                sil = float(silhouette_score(emb_r, labels))
                if sil > best_score:
                    best_score, best_labels, best_k = sil, labels, k
            except Exception:
                continue
    else:
        # Accurate path: GMM + BIC  (small groups)
        best_bic = np.inf
        for k in range(2, min(max_k + 1, n)):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42,
                                      covariance_type='diag', reg_covar=1e-3,
                                      max_iter=80, n_init=1,
                                      init_params='random_from_data')
                gmm.fit(emb_r)
                bic    = gmm.bic(emb_r)
                labels = gmm.predict(emb_r)
                if bic < best_bic:
                    best_bic, best_labels, best_k = bic, labels, k
            except Exception:
                continue

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)

    sil = 0.0
    if len(set(best_labels)) > 1:
        try:
            sil = float(silhouette_score(emb_r, best_labels))
        except Exception:
            pass

    return best_labels.astype(int), best_k, sil

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8a  — OPTIONAL CONSTRAINED LLM LABEL REFINEMENT  [TopicTag, DocEng 2024]
# ──────────────────────────────────────────────────────────────────────────────
def _light_stem(w: str) -> str:
    """
    Minimal English morphological normalisation — no NLTK dependency.

    Used by the LLM grounding check so that 'latencies' matches 'latency',
    'errors' matches 'error', 'completion' matches 'completed', etc.
    Avoids rejecting plurals and common tense variants while still requiring
    every label word to derive from evidence vocabulary.

    Based on Porter-stemmer-style suffix stripping (Porter 1980, adapted).
    """
    w = w.lower().strip()
    for suffix in ('ization', 'isation', 'ousness', 'iveness',
                   'ization', 'ities', 'iness',
                   'ation', 'ments', 'ness',
                   'ies', 'ied', 'ing', 'ers',
                   'ed', 'es', 'er', 'ly', 's'):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    return w

def make_llm_label_fn(base_url: str = OLLAMA_URL_DEFAULT,
                       model: str = OLLAMA_MODEL_DEFAULT,
                       provider: str = 'ollama',
                       api_key: str = '') -> Optional[Callable]:
    """
    Build a TopicTag-style constrained LLM label refinement function backed
    by a local Ollama server (OpenAI-compatible API at /v1).

    [TopicTag] Eren et al. (2024) run NMF to discover topics, then use an LLM
    to generate human-readable concept labels from the NMF topic terms.
    The LLM receives ONLY the extracted evidence from the CSV — it cannot
    alter the tree, cannot introduce new vocabulary, and must pass a strict
    grounding check (every word in the proposed label must appear in evidence).

    Local-LLM choice (Qwen 2.5 3B Instruct via Ollama) is deliberate:
      • zero cost, zero API dependency, fully reproducible
      • no external data transmission (privacy + thesis defensibility)
      • TopicTag itself benchmarks open models (Llama, Mistral) — using an
        open local model matches the paper's evaluation setup more closely
        than a closed hosted model.

    Returns a callable (candidate, top_terms, parent_path, sample_texts)
    → (label, metadata_dict).  Returns None if Ollama is unreachable or the
    openai client package is missing.
    """
    client = _make_llm_client(provider, base_url, api_key)
    if client is None:
        return None

    def _refine(candidate: str, top_terms: list, parent_path: str,
                sample_texts: list):
        meta = {'confidence': 0.0, 'evidence_terms': [],
                'reason': '', 'raw_label': ''}
        prompt = (
            'You are labeling a cluster in a metadata variable hierarchy.\n'
            'The label MUST be derived strictly from the evidence terms and '
            'sample variable descriptions provided. Do not introduce concepts '
            'or vocabulary that are not visible in the evidence.\n\n'
            f'Parent path: {parent_path}\n'
            f'Evidence terms (from NMF/TF-IDF over the cluster): {", ".join(top_terms[:10])}\n'
            f'Sample variable descriptions:\n'
            + '\n'.join(f'  - {str(t)[:160]}' for t in sample_texts[:4]) + '\n'
            f'\nCurrent candidate label: {candidate}\n\n'
            'Task: Return a concise 2–5 word concept label that PARAPHRASES '
            'the evidence into a cleaner concept name.\n'
            'Rules:\n'
            '1. Every word in the label must appear in (or be an obvious '
            'morphological variant of) the evidence terms or sample descriptions.\n'
            '2. Do not invent domain concepts that are not in the evidence.\n'
            '3. Prefer multi-word noun phrases over single keywords.\n'
            '4. Avoid generic words: data, score, variable, assessment, total, '
            'description, value, decimal.\n'
            '5. Use base forms — singular nouns (Latency not Latencies), and '
            'avoid -ing / -ed verb suffixes unless required.\n'
            '6. Output strict JSON only — no prose, no markdown.\n\n'
            'Output: {"label": "...", "evidence_terms": ["...", "..."], "confidence": 0.0}'
        )
        try:
            # max_tokens generous: reasoning models (Qwen3) emit <think> traces
            # that consume budget before the JSON appears.
            resp = _safe_chat_completion(client, model, prompt,
                                           max_tokens=1200, temperature=0.2)
            raw = (resp.choices[0].message.content or '').strip()
            result = _parse_json_response(raw)
            label  = str(result.get('label', '')).strip()
            conf   = float(result.get('confidence', 0))
            evid   = result.get('evidence_terms', []) or []
            meta['confidence'], meta['evidence_terms'] = conf, evid
            meta['raw_label'] = label
            if not label:
                meta['reason'] = 'empty'
                return candidate, meta
            # NOTE: we IGNORE the LLM's self-reported confidence.  Qwen 3B
            # routinely returns conf ≈ 0.5 on perfectly good labels — using it
            # as a gate rejected useful refinements.  Grounding (below) is the
            # real anti-hallucination check; if every label word stems back to
            # the corpus, the label is accepted regardless of self-confidence.
            #
            # Stem-aware grounding accepts morphological variants
            # (latencies ↔ latency, errors ↔ error, completion ↔ completed).
            haystack_text = ' '.join(
                str(s) for s in (
                    top_terms[:10] + evid + list(sample_texts[:4]) + [parent_path]
                )
            ).lower()
            haystack_stems = {_light_stem(w) for w in haystack_text.split()
                               if len(w) >= 3}
            label_words = [w for w in label.lower().split() if len(w) >= 3]
            label_stems = {_light_stem(w) for w in label_words}
            ungrounded  = label_stems - haystack_stems
            if label_words and ungrounded:
                meta['reason'] = f'ungrounded_words: {sorted(ungrounded)}'
                return candidate, meta
            meta['reason'] = 'accepted'
            return label, meta
        except Exception as e:
            meta['reason'] = f'exception: {type(e).__name__}'
            return candidate, meta

    return _refine

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5a — UPSTREAM LLM PHRASE-ROLE CLASSIFIER  [TopicGPT, NAACL 2024 adapted]
#
# Used to discover semantic roles (Measure / Statistic / Condition / Subtype
# style dimensions) from the corpus and ASSIGN each mined phrase to a role.
# This is fundamentally different from TopicTag-style label refinement:
#   • TopicTag (and the make_llm_label_fn above) uses LLMs to RENAME clusters
#     after the tree structure is already decided.  LLM is downstream.
#   • This function uses an LLM to CREATE the role schema before the tree is
#     built.  Slot ordering and parent-child placement are driven by the
#     LLM-derived roles.  LLM is upstream — closer to TopicGPT (Pham et al.,
#     NAACL 2024), where LLMs perform topic discovery, not topic labeling.
#
# Anti-hallucination contract (preserved):
#   • Every phrase in the role assignment MUST come from the corpus verbatim.
#     Outputs are validated against the input phrase list — anything new is
#     dropped.
#   • Role NAMES are proposed by the LLM but kept to 1–2 generic English
#     words; long or fanciful role names are rejected.
#   • If validation fails, the caller falls back to deterministic mutual-
#     exclusion clustering (current slot mining behaviour).  No silent
#     failure modes.
# ──────────────────────────────────────────────────────────────────────────────
def make_llm_role_classifier_fn(base_url: str = OLLAMA_URL_DEFAULT,
                                  model: str = OLLAMA_MODEL_DEFAULT,
                                  provider: str = 'ollama',
                                  api_key: str = '') -> Optional[Callable]:
    """
    Build a callable that classifies repeated corpus phrases into semantic
    roles using an LLM (Ollama or Groq).
    """
    client = _make_llm_client(provider, base_url, api_key)
    if client is None:
        return None

    def _classify(phrases: list, sample_descs: list, group_name: str = ''):
        meta = {'reason': '', 'raw': '', 'role_count': 0}
        if not phrases:
            meta['reason'] = 'no_phrases'
            return None, meta
        # Truncate to keep the prompt small and the model focused
        phrases_list = list(dict.fromkeys(str(p) for p in phrases))[:60]
        prompt = (
            'You are analysing a list of repeated phrases mined from a '
            'data-dictionary corpus.\n\n'
            f'Group context: {group_name or "(unknown)"}\n\n'
            'Repeated phrases (verbatim from the corpus):\n'
            + '\n'.join(f'  - {p}' for p in phrases_list) + '\n\n'
            'Sample variable descriptions for context:\n'
            + '\n'.join(f'  - {str(s)[:180]}' for s in sample_descs[:4]) + '\n\n'
            'TASK: Group these phrases into 2–5 SEMANTIC ROLES. Each role '
            'represents one ORTHOGONAL DIMENSION of what the variable measures '
            '(for example: what is measured, what statistic is used, under '
            'what condition).\n\n'
            'STRICT RULES:\n'
            '1. Use 2 to 5 roles. Fewer is better when phrases do not really '
            'belong to different dimensions.\n'
            '2. Role names: ONE OR TWO generic English words, lowercase. '
            'Examples of good role names: "measure", "statistic", "condition", '
            '"outcome", "subtype", "modifier". Do not invent fancy or domain-'
            'specific role names.\n'
            '3. Every phrase from the input list MUST appear EXACTLY ONCE in '
            'exactly one role. Copy phrases VERBATIM. Do not rephrase, '
            'normalise, plural-strip, or invent new phrases.\n'
            '4. Phrases that are alternatives (rarely co-occur in the same '
            'variable) should go in the SAME role.\n'
            '5. Phrases that describe DIFFERENT dimensions of the same '
            'variable should go in DIFFERENT roles.\n'
            '6. Output strict JSON only — no prose, no markdown fences.\n\n'
            'Output schema:\n'
            '{\n'
            '  "roles": {\n'
            '    "role_name_1": ["phrase a", "phrase b", ...],\n'
            '    "role_name_2": ["phrase c", "phrase d", ...]\n'
            '  }\n'
            '}'
        )
        try:
            resp = _safe_chat_completion(client, model, prompt,
                                           max_tokens=2000, temperature=0.1)
            raw = (resp.choices[0].message.content or '').strip()
            meta['raw'] = raw[:400]
            result = _parse_json_response(raw)
            roles_raw = result.get('roles', {}) or {}
            if not isinstance(roles_raw, dict):
                meta['reason'] = 'roles_not_dict'
                return None, meta

            # ── Validation ────────────────────────────────────────────────────
            input_set = {p.lower(): p for p in phrases_list}
            roles_clean: dict = {}
            seen_phrases: set = set()
            for role_name, items in roles_raw.items():
                # Role name must be 1–2 generic words
                rn = str(role_name).strip().lower()
                if not rn or len(rn.split()) > 2 or len(rn) > 24:
                    continue
                kept: list = []
                if not isinstance(items, list):
                    continue
                for it in items:
                    s = str(it).strip().lower()
                    if not s or s in seen_phrases:
                        continue
                    if s not in input_set:
                        # Phrase invented by LLM — drop (anti-hallucination)
                        continue
                    kept.append(input_set[s])   # original casing
                    seen_phrases.add(s)
                if len(kept) >= 2:
                    roles_clean[rn] = kept
            if len(roles_clean) < 2:
                meta['reason'] = 'too_few_valid_roles'
                return None, meta
            meta['reason']     = 'accepted'
            meta['role_count'] = len(roles_clean)
            return roles_clean, meta
        except Exception as e:
            meta['reason'] = f'exception: {type(e).__name__}'
            return None, meta

    return _classify

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8b–d  — NODE LABELING  [ZHU §4.3 / TopicTag]
# ──────────────────────────────────────────────────────────────────────────────
# Generic metadata field-name boilerplate that appears in essentially any
# data-dictionary header (description, value, name, ...).  Dataset-specific
# column tokens (e.g. 'fullDisplayName' for HCP, 'Decimal Places' for AI-MIND)
# are added at runtime by build_field_noise() — derived from the user-confirmed
# column names, never enumerated by hand.
_GENERIC_FIELD_NOISE = frozenset({
    'description', 'definition', 'value', 'metadata', 'desc',
    'name', 'item', 'variable', 'field', 'attribute',
    'code', 'type', 'dtype',
})

def build_field_noise(configs: dict) -> frozenset:
    """
    Build the field-name boilerplate set entirely from the user-confirmed
    column roles.  For each detected column name we extract alphanumeric tokens
    and add them to the noise set.

    Examples (derived, NOT hardcoded):
      HCP   'fullDisplayName' → {fulldisplayname}
      HCP   'columnHeader'    → {columnheader}
      AI-MIND 'Decimal Places' → {decimal, places}

    Result: the same effective filter as a hand-crafted list, but produced
    deterministically from whatever columns the current CSV has.  Datasets we
    have never seen get an automatically-tailored noise set.
    """
    noise = set(_GENERIC_FIELD_NOISE)
    for cfg in (configs or {}).values():
        for col_list in cfg.values():
            for col in col_list:
                tokens = ''.join(c if c.isalnum() else ' '
                                 for c in str(col)).split()
                noise.update(t.lower() for t in tokens if len(t) >= 2)
    return frozenset(noise)

# Module-level fallback used when label_cluster is called without a configs-
# derived noise set.  Replaced at build time by the Streamlit pipeline below.
FIELD_NAME_NOISE: frozenset = _GENERIC_FIELD_NOISE

def _extract_common_prefix_phrase(cluster_texts: list,
                                    min_coverage: float = 0.6) -> str:
    """
    Many data dictionaries write 'Concept Name: definition...' in the
    description.  If most cluster members share a concept-name prefix, that
    prefix IS the concept label.  Fully data-driven — works on any dictionary
    using the 'name: definition' convention.

    Returns a Title-cased phrase, or '' if no shared prefix is strong enough.
    """
    # _text is "col1: val1 | col2: val2 | ...".  Find description-like field
    # and take its prefix before the inner colon.
    prefixes = []
    for t in cluster_texts:
        for chunk in str(t).split(' | '):
            if ':' not in chunk:
                continue
            key, val = chunk.split(':', 1)
            key_l = key.lower()
            if 'descrip' in key_l or 'def' in key_l or 'full' in key_l:
                phrase = val.split(':')[0].strip()
                tokens = phrase.split()
                if 2 <= len(tokens) <= 6:
                    prefixes.append(tokens)
                break

    if not prefixes:
        return ''

    n_thresh = max(1, int(min_coverage * len(prefixes)))
    max_len  = max(len(p) for p in prefixes)
    for length in range(min(6, max_len), 1, -1):
        starts = Counter(tuple(p[:length]) for p in prefixes if len(p) >= length)
        if not starts:
            continue
        top, cnt = starts.most_common(1)[0]
        if cnt >= n_thresh:
            return ' '.join(top).title()
    return ''

def _bigram_preferred_terms(diff: np.ndarray, terms: np.ndarray,
                             boilerplate: set, prefix_lower: str,
                             n_terms: int) -> list:
    """
    Pick top-n discriminative terms, preferring bigrams and removing redundancy.

    Rules:
    1. Skip short tokens (<3 chars), boilerplate, field-name noise, and any
       token already in the prefix.
    2. When a bigram is selected, drop any previously-picked unigram that is
       a substring of it.
    3. Skip unigrams that are substrings of any already-picked bigram.
    """
    order  = np.argsort(diff)[::-1]
    picked, picked_lower = [], []
    for i in order:
        t  = terms[i]; tl = t.lower()
        if (len(t) < 3 or tl in boilerplate or tl in FIELD_NAME_NOISE
                or (prefix_lower and tl in prefix_lower)):
            continue
        is_unigram = ' ' not in t
        # Rule 3: unigram already covered by a picked bigram?
        if is_unigram and any(tl in pl for pl in picked_lower if ' ' in pl):
            continue
        # Rule 2: replace picked unigrams subsumed by this new bigram
        if not is_unigram:
            keep = [(p, pl) for p, pl in zip(picked, picked_lower)
                    if not (' ' not in pl and pl in tl)]
            picked, picked_lower = [k[0] for k in keep], [k[1] for k in keep]
        picked.append(t); picked_lower.append(tl)
        if len(picked) >= n_terms:
            break
    return picked

def label_cluster(cluster_texts: list, all_texts: list,
                  tfidf: TfidfVectorizer, n_terms: int = 3,
                  cluster_groups: Optional[list] = None,
                  parent_path: str = '',
                  llm_label_fn: Optional[Callable] = None,
                  return_provenance: bool = False):
    """
    Label a cluster node using a five-stage deterministic pipeline.

    A) Description-prefix candidate: extract concept phrase from 'Name: def...'
       pattern shared by ≥60% of cluster members.
    B) Group-purity prefix: if ≥70% share a top-level _group value, prepend it.
    C) Boilerplate filter: drop terms with TF-IDF IDF ≤ 1.7 plus FIELD_NAME_NOISE.
    D) Bigram-preferred discriminative suffix [ZHU §4.3].
    E) Optional constrained LLM refinement [TopicTag, DocEng 2024].

    When return_provenance=True returns a (label, provenance_dict) tuple where
    provenance records which stage produced the label (description_prefix /
    tfidf_bigram / group_anchor / llm) plus evidence terms and (for LLM) the
    confidence score and a grounding check result.
    """
    prov: dict = {'label_source': 'fallback',
                  'evidence_terms': [],
                  'confidence': 1.0,
                  'llm_used': False,
                  'llm_rejected': False}

    if not cluster_texts:
        return ('Group', prov) if return_provenance else 'Group'

    terms    = np.array(tfidf.get_feature_names_out())
    idf_vals = tfidf.idf_
    boilerplate = set(terms[idf_vals <= 1.7].tolist())

    # ── B) group-purity prefix ───────────────────────────────────────────────
    prefix = ''
    if cluster_groups:
        top_lvl = [str(g).split(' > ')[0].strip() for g in cluster_groups]
        top_grp, cnt = Counter(top_lvl).most_common(1)[0]
        if cnt / len(top_lvl) >= 0.70 and top_grp not in ('', 'Ungrouped'):
            prefix = top_grp

    # ── A) description-prefix candidate phrase ───────────────────────────────
    phrase = _extract_common_prefix_phrase(cluster_texts)
    if phrase and prefix and phrase.lower().startswith(prefix.lower()):
        phrase = phrase[len(prefix):].strip()

    # ── D) bigram-preferred discriminative terms ─────────────────────────────
    top_terms_raw, words = [], []
    try:
        X_all  = tfidf.transform(all_texts).toarray()
        X_clus = tfidf.transform(cluster_texts).toarray()
        diff   = X_clus.mean(axis=0) - X_all.mean(axis=0)
        top_terms_raw = [terms[i] for i in np.argsort(diff)[::-1][:20]]
        words = _bigram_preferred_terms(diff, terms, boilerplate,
                                          prefix.lower(), n_terms)
    except Exception:
        pass

    # ── compose deterministic candidate (records which stage produced it) ────
    if phrase:
        candidate = f'{prefix} — {phrase}' if prefix else phrase
        prov['label_source']   = 'description_prefix'
        prov['evidence_terms'] = [phrase] + ([prefix] if prefix else [])
    elif words:
        suffix    = ' / '.join(w.title() for w in words)
        candidate = f'{prefix} — {suffix}' if prefix else suffix
        prov['label_source']   = 'tfidf_bigram'
        prov['evidence_terms'] = list(words) + ([prefix] if prefix else [])
    elif prefix:
        candidate = prefix
        prov['label_source']   = 'group_anchor'
        prov['evidence_terms'] = [prefix]
    else:
        candidate = 'Group'

    # ── E) optional constrained LLM refinement [TopicTag] ────────────────────
    # LLM is used ONLY as a re-phraser of evidence already present in the
    # cluster.  Refinement is rejected if the returned label is not grounded
    # in the evidence terms — keeping the user's contract that "labels should
    # come from the csv itself".
    if llm_label_fn and candidate != 'Group':
        prov['llm_used'] = True
        try:
            refined, llm_meta = llm_label_fn(
                candidate, top_terms_raw[:10],
                parent_path, cluster_texts[:4])
            # Always record the raw LLM proposal and the grounding-check reason
            # — even when rejected — so the Provenance tab can show "what did
            # the LLM suggest and why was it dropped?"
            prov['llm_raw_label'] = llm_meta.get('raw_label', '')
            prov['llm_reason']    = llm_meta.get('reason', '')
            if refined and refined != candidate:
                candidate = refined
                prov['label_source']   = 'llm'
                prov['confidence']     = llm_meta.get('confidence', 0.0)
                prov['evidence_terms'] = llm_meta.get('evidence_terms',
                                                       prov['evidence_terms'])
            else:
                prov['llm_rejected'] = True
        except Exception as _e:
            prov['llm_rejected'] = True
            prov['llm_reason']   = f'exception: {type(_e).__name__}'

    if return_provenance:
        return candidate, prov
    return candidate

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5b  — PHRASE-SLOT MINING  (data-driven IE-style slot induction)
#
# Information-extraction adaptation for structured metadata dictionaries.
# Many data dictionaries (e.g. AI-MIND cognitive tests, HCP Study Completion)
# express each variable as a regular phrase combining several semantic
# dimensions (measure type, statistic, condition, etc.).  Document-level
# methods (NMF, BERTopic, CTM) cannot separate these because all dimensions
# collapse into one vector.  Phrase-slot mining decomposes each description
# into multiple phrases, identifies which phrases are mutually-exclusive
# alternatives across the corpus, and uses each alternative-set as a slot.
#
# No domain hardcoding: slot phrases are discovered from n-gram co-occurrence
# patterns in the actual descriptions.  Slot names are derived from the
# phrases themselves; if structure is too weak, the algorithm reports
# `valid=False` and the caller falls back to NMF.
# ──────────────────────────────────────────────────────────────────────────────
# ── Generic dictionary markers stripped before slot mining ────────────────────
# These are universal data-dictionary conventions (`KEY: ...`, `Note: ...`,
# `Question: ...`) — not domain knowledge.  Without stripping, they become
# false slot phrases (e.g. AI-MIND output contained `Key` nodes promoted from
# `KEY: DMS Percent Correct ...`).
_LEADING_MARKERS = frozenset({
    'key', 'note', 'notes', 'definition', 'description', 'desc',
    'question', 'q', 'item', 'value', 'meaning', 'label',
})

def _strip_leading_markers(text: str) -> str:
    """Iteratively strip leading dictionary markers like 'KEY:' or 'Note:'."""
    out = text
    for _ in range(4):  # bounded — never more than a few stacked markers
        if ':' not in out:
            break
        head, rest = out.split(':', 1)
        if head.strip().lower() in _LEADING_MARKERS:
            out = rest.strip()
        else:
            break
    return out

def _split_concept_and_body(text: str) -> tuple:
    """
    Split a 'Concept Name: definition sentence' description into its concept
    prefix and its definition body.

    Data dictionaries overwhelmingly use the convention
        <short concept phrase> : <longer explanatory definition>
    e.g.  'DMS Correct Latency Standard Deviation (SD) (0 second delay):
           The standard deviation of response latencies for trials ...'

    The concept phrase is the clean, canonical label; the body is explanatory
    prose that introduces boilerplate ('the number of times', 'a subject
    revisits a box ...') and weaker surface phrases ('response latencies').

    Returns (concept, body).  If no clear concept/body boundary exists, concept
    is the whole text and body is ''.  Generic — no domain knowledge.
    """
    s = _strip_leading_markers(str(text).strip())
    if ':' not in s:
        return s, ''
    head, body = s.split(':', 1)
    head = head.strip()
    body = body.strip()
    # Only treat `head` as a concept prefix if it's phrase-length (not a whole
    # sentence) — a real concept name is short.
    if 1 <= len(head.split()) <= 14:
        return head, body
    return s, ''

# Generic data-dictionary definition-prose openers.  Role values starting with
# these are explanatory fragments, not concept labels — reject them.  Generic
# English, not domain vocabulary.
_BOILERPLATE_VALUE_PREFIXES = (
    'the number of', 'number of', 'the subject', 'a subject', 'the participant',
    'a participant', 'this measure', 'this variable', 'calculated across',
    'calculated as', 'the percentage of', 'the proportion of', 'the total number',
    'the mean of', 'the median of', 'the standard deviation of', 'the amount of',
    'the time', 'the length of', 'expressed as', 'defined as', 'measured as',
)

def _is_boilerplate_value(val: str) -> bool:
    """True if a role value looks like a definition fragment rather than a label."""
    v = str(val).strip().lower()
    if not v:
        return True
    if any(v.startswith(p) for p in _BOILERPLATE_VALUE_PREFIXES):
        return True
    # Sentence-length values are definitions, not concept labels
    if len(v.split()) > 6:
        return True
    return False

# ── English stopwords for phrase-quality filtering ────────────────────────────
# Reject phrases like 'and', 'them', 'to be', 'have a lot of' from becoming
# slot nodes.  This is general English filtering, not domain knowledge.
_STOPWORDS = frozenset(
    'a an the and or but of in on at by for with about into during '
    'through over under above below from to as is are was were be being '
    'been have has had do does did this that these those they them their '
    'it its he she his her you your we our i my me us not no nor '
    'how often when where which who why what '
    'lot lots much many more most some any all none '
    'very also too just so such only even still even also '
    'one two three first second '   # 'second' as standalone — careful: kept in n-grams via context
    .split()
)

def _phrase_is_meaningful(phrase: str, group_name_lower: str = '') -> bool:
    """
    Reject phrases that should never be a hierarchy node.

    Rules (all generic, no domain knowledge):
      • all-stopword phrases ('and', 'to be', 'have a lot of')
      • boundary stopwords ('the response time' → 'the' boundary)
      • phrase equals the parent group name ('DMS' inside DMS branch)
      • single dictionary-marker words ('key', 'note')
      • pure numeric tokens
    """
    p = phrase.strip().lower()
    if not p:
        return False
    if p == group_name_lower:
        return False
    if p in _LEADING_MARKERS:
        return False
    tokens = p.split()
    if not tokens:
        return False
    # Boundary stopwords
    if tokens[0] in _STOPWORDS or tokens[-1] in _STOPWORDS:
        return False
    # Need at least one content token (non-stopword, length ≥ 2)
    content = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    if not content:
        return False
    # Single-token phrases must be a meaningful word, not a bare digit
    if len(tokens) == 1 and tokens[0].isdigit():
        return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# OPTION D — SBERT PHRASE CLUSTERING + CONSTRAINED LLM ROLE NAMING
#
# Following the EDC pattern (Zhang & Soh, EMNLP 2024) and ZOES (arXiv
# 2506.04458, 2025):
#   Extract   → mine repeated phrases from each variable description
#   Define    → semantically cluster phrases via SBERT + AgglomerativeClustering
#   Canonicalize → constrained-vocabulary LLM names each cluster with a role
#                  (measure / statistic / condition / subtype / outcome / modifier)
#
# Anti-hallucination contract:
#   • Phrases are extracted verbatim from the corpus, never invented
#   • LLM only NAMES discovered clusters — cannot move phrases
#   • Role names are constrained to a fixed generic English vocabulary
#     (configurable via constrained=True/False)
#   • Anonymous fallback when LLM is unavailable or invalid (cluster_0, ...)
#
# Different from Approach 1: Approach 1 embeds the WHOLE description as one
# vector; this embeds each EXTRACTED PHRASE as its own vector.  Approach 1
# clusters variables; this clusters phrases.  Approach 1 produces a single
# similarity tree; this produces a multi-role decomposition + nested tree.
# ──────────────────────────────────────────────────────────────────────────────
_ALLOWED_ROLES_CONSTRAINED = frozenset({
    'measure', 'statistic', 'condition', 'subtype', 'outcome', 'modifier',
})

# Generic role priority for hierarchy nesting (no domain knowledge).
# Higher priority = outermost (closest to root) level inside the group.
_ROLE_PRIORITY = {
    'measure':   6,
    'outcome':   5,
    'statistic': 4,
    'condition': 3,
    'modifier':  2,
    'subtype':   1,
}

def make_llm_role_namer_fn(base_url: str = OLLAMA_URL_DEFAULT,
                            model: str = OLLAMA_MODEL_DEFAULT,
                            constrained: bool = True,
                            provider: str = 'ollama',
                            api_key: str = '') -> Optional[Callable]:
    """
    Returns a callable (phrases, sample_descs) → (role_name, meta).
    Works with Ollama (local) or Groq (cloud).
    """
    client = _make_llm_client(provider, base_url, api_key)
    if client is None:
        return None

    def _name_cluster(phrases: list, sample_descs: list = None):
        meta = {'reason': '', 'raw': '', 'constrained': constrained}
        if not phrases:
            meta['reason'] = 'empty_phrases'
            return None, meta
        sample_descs = sample_descs or []

        head = (
            'Below is a cluster of semantically related phrases mined from a '
            'data-dictionary corpus. The phrases were grouped by sentence-'
            'transformer embedding similarity.\n\n'
            'Phrases in cluster:\n'
            + '\n'.join(f'  - {p}' for p in phrases[:20]) + '\n\n'
            'Sample variable descriptions for context:\n'
            + '\n'.join(f'  - {str(s)[:160]}' for s in sample_descs[:3]) + '\n\n'
        )
        if constrained:
            prompt = head + (
                'Choose EXACTLY ONE semantic role from this fixed list that '
                'best names what these phrases share. Pick the SINGLE best fit:\n'
                '  - measure   (the base concept being measured)\n'
                '  - statistic (mean, median, standard deviation, total, ...)\n'
                '  - condition (when/where/under what circumstances)\n'
                '  - outcome   (result/output type)\n'
                '  - subtype   (a specific kind of measure)\n'
                '  - modifier  (a qualifier)\n\n'
                'Output strict JSON only — no markdown:\n'
                '{"role": "measure"}'
            )
        else:
            prompt = head + (
                'Propose ONE generic 1–2 word lowercase English noun that names '
                'the semantic role these phrases share. Examples of good roles: '
                'measure, statistic, condition, subtype, location, time, '
                'identifier, frequency, severity, quality.\n\n'
                'Output strict JSON only:\n'
                '{"role": "..."}'
            )

        try:
            # max_tokens must cover reasoning-model <think> traces (Qwen3)
            # before the tiny JSON answer appears.
            resp = _safe_chat_completion(client, model, prompt,
                                           max_tokens=800, temperature=0.1)
            raw = (resp.choices[0].message.content or '').strip()
            meta['raw'] = raw[:200]
            result = _parse_json_response(raw)
            role = str(result.get('role', '')).strip().lower()
            if not role:
                meta['reason'] = 'empty_role'
                return None, meta
            if constrained:
                if role not in _ALLOWED_ROLES_CONSTRAINED:
                    meta['reason'] = f'role_not_allowed: {role}'
                    return None, meta
            else:
                toks = role.split()
                if len(toks) > 2 or len(role) > 24:
                    meta['reason'] = 'role_too_long'
                    return None, meta
                if not all(t.isalpha() and t.islower() for t in toks):
                    meta['reason'] = 'role_not_alpha_lower'
                    return None, meta
            meta['reason'] = 'accepted'
            return role, meta
        except Exception as e:
            meta['reason'] = f'exception: {type(e).__name__}'
            return None, meta

    return _name_cluster

def _extract_phrases_for_role_clustering(texts: list,
                                           text_col_names: Optional[list],
                                           group_name: str,
                                           min_phrase_count: int = 2
                                           ) -> tuple:
    """
    Shared helper: extract repeated meaningful phrases from a group of
    descriptions. Returns (phrases_list, per_row_phrase_sets, regularity).
    Regularity = fraction of rows that contain ≥2 mined phrases.
    """
    text_keys = ({c.strip().lower() for c in (text_col_names or [])}
                 if text_col_names else None)
    n = len(texts)

    # 1. extract description-prefix per row
    prefixes = []
    for t in texts:
        prefix = ''
        for chunk in str(t).split(' | '):
            if ':' not in chunk:
                continue
            key, val = chunk.split(':', 1)
            key_l = key.strip().lower()
            if text_keys is not None and key_l not in text_keys:
                continue
            prefix = val.split(':')[0].strip()
            break
        if not prefix:
            prefix = str(t).strip()
        prefix = _strip_leading_markers(prefix)
        prefixes.append(prefix)

    # 2. normalise
    def _norm(p):
        s = ''.join(ch if ch.isalnum() or ch == ' ' else ' '
                    for ch in str(p).lower())
        return ' '.join(s.split())
    norm = [_norm(p) for p in prefixes]

    # 3. mine n-grams 1..4 ≥3 chars
    row_phrases: list = []
    phrase_count: Counter = Counter()
    for text in norm:
        tokens = text.split()
        row_set = set()
        for ngram_n in range(1, 5):
            for i in range(len(tokens) - ngram_n + 1):
                phrase = ' '.join(tokens[i:i + ngram_n])
                if len(phrase) >= 3:
                    row_set.add(phrase)
        row_phrases.append(row_set)
        for p in row_set:
            phrase_count[p] += 1

    # 4. keep repeated, retain maximal per row
    keep = {p for p, c in phrase_count.items() if c >= min_phrase_count}
    refined = []
    for row in row_phrases:
        sorted_by_len = sorted(row & keep, key=lambda p: -len(p.split()))
        kept: list = []
        for p in sorted_by_len:
            if not any(p != q and p in q for q in kept):
                kept.append(p)
        refined.append(set(kept))

    phrase_count = Counter()
    for r in refined:
        for p in r:
            phrase_count[p] += 1

    group_lower = (group_name or '').strip().lower()
    phrases = sorted([p for p, c in phrase_count.items()
                       if c >= min_phrase_count
                       and _phrase_is_meaningful(p, group_lower)])

    regularity = sum(1 for r in refined if len(r) >= 2) / max(1, n)
    return phrases, refined, round(regularity, 4)

# ──────────────────────────────────────────────────────────────────────────────
# PER-ROW LLM ROLE EXTRACTOR  [Zhu et al. EMNLP 2025 — proper implementation]
#
# For each variable description, ONE LLM call extracts role values directly:
#   measure   = base concept being measured (e.g. "Correct Latency")
#   statistic = statistical aggregation (e.g. "Standard Deviation")
#   condition = experimental condition (e.g. "0 second delay")
#   subtype   = specific error / outcome subtype (e.g. "Incorrect Colour")
#
# Anti-hallucination contract:
#   • Each returned value MUST be a literal substring of the input description
#     (grounding check rejects anything else).
#   • LLM cannot invent new roles — only the four canonical roles are output.
#   • Empty string is a valid output ("this row has no statistic" etc.).
#
# Cost: ONE call per variable.  AI-MIND ≈ 108 calls, HCP ≈ 813.
# ──────────────────────────────────────────────────────────────────────────────
_PER_ROW_ROLES = ('measure', 'statistic', 'condition', 'subtype')

def make_per_row_role_extractor_fn(base_url: str = OLLAMA_URL_DEFAULT,
                                     model: str = OLLAMA_MODEL_DEFAULT,
                                     provider: str = 'ollama',
                                     api_key: str = '') -> Optional[Callable]:
    """
    Returns a callable (description, sample_descs_in_group) → (roles_dict, meta).

    Implements the Zhu et al. (2025, EMNLP) multi-aspect encoding pattern:
    instead of clustering phrases and naming clusters, ask the LLM to extract
    each role value directly from each row's description.  Each extracted
    value must be a verbatim substring (anti-hallucination).
    """
    client = _make_llm_client(provider, base_url, api_key)
    if client is None:
        return None

    def _extract(concept: str, sample_concepts: list = None,
                 full_text: str = None):
        meta = {'reason': '', 'raw': ''}
        if not concept or len(str(concept).strip()) < 3:
            return {}, {'reason': 'empty_description'}

        # `concept` = the clean concept-name phrase (preferred extraction source)
        # `full_text` = concept + definition body (used for subtype + grounding)
        concept = str(concept)[:300]
        full    = str(full_text or concept)[:600]
        sample_concepts = sample_concepts or []
        prompt = (
            'You are extracting semantic role values from one variable in a '
            'data dictionary.\n\n'
            f'CONCEPT NAME (use this for measure / statistic / condition):\n'
            f'  {concept}\n\n'
            f'FULL DEFINITION (use ONLY for subtype, and only if needed):\n'
            f'  {full}\n\n'
            'Similar concept names in the same group (context):\n'
            + '\n'.join(f'  - {str(s)[:120]}' for s in sample_concepts[:3]) + '\n\n'
            'TASK: Extract values for these four semantic roles. Take '
            'measure, statistic and condition from the CONCEPT NAME wherever '
            'possible. Every value MUST be a verbatim substring of the CONCEPT '
            'NAME or FULL DEFINITION. Empty string if a role does not apply.\n\n'
            '  measure   = the base quantity being measured\n'
            '              (e.g. "reaction time", "accuracy", "score")\n'
            '  statistic = a statistical aggregation operator\n'
            '              (e.g. "mean", "median", "standard deviation", "total")\n'
            '  condition = an experimental condition or scope\n'
            '              (e.g. "baseline", "follow-up", "task condition")\n'
            '  subtype   = a specific subtype / kind / error type\n'
            '              (e.g. "error type", "response type", "trial type")\n\n'
            'STRICT RULES:\n'
            '1. Each value COPIED VERBATIM — do not invent, summarise, paraphrase.\n'
            '2. Prefer short concept phrases over long definition fragments.\n'
            '3. Do NOT return a value that is a sentence or starts with "the '
            'number of", "the subject", "calculated across" — those are '
            'definition prose, not labels.\n'
            '4. Empty string "" for roles that do not apply.\n'
            '5. Output strict JSON only:\n\n'
            '{"measure": "...", "statistic": "...", "condition": "...", "subtype": ""}'
        )

        try:
            resp = _safe_chat_completion(client, model, prompt,
                                           max_tokens=1500, temperature=0.1)
            raw = (resp.choices[0].message.content or '').strip()
            meta['raw'] = raw[:300]
            result = _parse_json_response(raw)
        except Exception as e:
            meta['reason'] = f'exception: {type(e).__name__}: {str(e)[:80]}'
            return {}, meta

        # Grounding is checked against the FULL text (concept + body) so that
        # subtype values living in the definition body still pass.
        ground_lower  = full.lower()
        ground_tokens = [w.strip(',.()[]{}"\'') for w in ground_lower.split()]
        ground_stems  = {_light_stem(w) for w in ground_tokens
                          if len(w) >= 3 and w not in _STOPWORDS}

        roles: dict = {}
        rejected: list = []
        for role in _PER_ROW_ROLES:
            val = result.get(role, '')
            if not isinstance(val, str):
                continue
            val_clean = val.strip().strip('"').strip("'")
            if not val_clean:
                continue
            # P2: reject definition-prose fragments before grounding
            if _is_boilerplate_value(val_clean):
                rejected.append((role, val_clean, ['boilerplate']))
                continue
            # Strict substring (cheapest, most common)
            if val_clean.lower() in ground_lower:
                roles[role] = val_clean
                continue
            # Token-stem grounding fallback (morphological variants)
            val_tokens = [w.strip(',.()[]{}"\'') for w in val_clean.lower().split()]
            val_stems  = {_light_stem(w) for w in val_tokens
                           if len(w) >= 3 and w not in _STOPWORDS}
            if val_stems and val_stems.issubset(ground_stems):
                roles[role] = val_clean
            else:
                missing = sorted(val_stems - ground_stems) if val_stems else ['(no content tokens)']
                rejected.append((role, val_clean, missing))

        meta['reason']   = 'accepted' if roles else 'all_rejected'
        meta['rejected'] = rejected
        meta['n_extracted'] = len(roles)
        meta['n_rejected']  = len(rejected)
        return roles, meta

    return _extract

def discover_roles_via_per_row_extraction(
        texts: list,
        text_col_names: Optional[list],
        per_row_extractor_fn: Callable,
        group_name: str = '',
        regularity_threshold: float = 0.40) -> dict:
    """
    Zhu et al. (EMNLP 2025) style: one LLM call per row extracts measure /
    statistic / condition / subtype values directly from each description.

    Returns the same dict shape as discover_roles_via_sbert_phrase_clustering()
    so it can flow into build_role_hierarchy().
    """
    n = len(texts)
    if n < 4:
        return {'valid': False, 'reason': 'too_few_rows',
                'regularity': 0.0, 'roles': {}, 'row_assignments': []}

    # Extract a "concept-prefix" snippet per row for context to the extractor
    text_keys = ({c.strip().lower() for c in (text_col_names or [])}
                 if text_col_names else None)

    def _row_desc_value(t: str) -> str:
        """Return the raw description-column value (everything after 'desc:')."""
        for chunk in str(t).split(' | '):
            if ':' not in chunk:
                continue
            key, val = chunk.split(':', 1)
            key_l = key.strip().lower()
            if text_keys is not None and key_l not in text_keys:
                continue
            return val.strip()
        return str(t).strip()

    # For each row split into (concept-name prefix, definition body).
    # The concept feeds measure/statistic/condition; the full text grounds
    # subtype and the grounding check.
    row_concepts: list = []
    row_fulls:    list = []
    for t in texts:
        dv = _row_desc_value(t)
        concept, body = _split_concept_and_body(dv)
        row_concepts.append(concept)
        row_fulls.append((concept + ' ' + body).strip() if body else concept)

    sample_descs = row_concepts[:3]

    # Call extractor per row.  Use Streamlit progress bar so the user sees
    # what's happening (especially important for HCP).
    try:
        import streamlit as _st_p
        pbar = _st_p.progress(0.0, text=f'Extracting roles for "{group_name}"…')
        show_progress = True
    except Exception:
        pbar = None
        show_progress = False

    per_row_roles: list = []
    per_row_audit: list = []   # full audit trail (proposed + rejected per row)
    all_phrases_by_role: dict = defaultdict(list)
    extractor_meta_summary: dict = defaultdict(int)

    for i, concept in enumerate(row_concepts):
        if show_progress and pbar is not None:
            try:
                pbar.progress((i + 1) / max(1, n),
                               text=f'[{group_name}] row {i+1}/{n}')
            except Exception:
                pass
        try:
            roles, meta = per_row_extractor_fn(
                concept, sample_descs, full_text=row_fulls[i])
        except Exception as e:
            roles, meta = {}, {'reason': f'exception: {type(e).__name__}'}
        per_row_roles.append(roles)
        # Audit trail: store per-row details for the Role Decomposition tab
        per_row_audit.append({
            'row_idx':    i,
            'description_snippet': str(concept)[:120],
            'accepted_roles':      dict(roles),
            'rejected':            meta.get('rejected', []),
            'reason':              meta.get('reason', ''),
            'raw':                 meta.get('raw', ''),
        })
        extractor_meta_summary[meta.get('reason', 'unknown')] += 1
        for role, val in roles.items():
            if val and val not in all_phrases_by_role[role]:
                all_phrases_by_role[role].append(val)

    try:
        if show_progress and pbar is not None:
            pbar.empty()
    except Exception:
        pass

    # Keep only roles that have ≥ 2 distinct values across the corpus
    roles_final: dict = {}
    for role in _PER_ROW_ROLES:
        vals = all_phrases_by_role.get(role, [])
        if len(vals) >= 2:
            roles_final[role] = vals

    # Coverage = fraction of rows with ≥1 non-empty role assignment
    covered = sum(1 for r in per_row_roles if any(r.get(rl) for rl in roles_final))
    coverage = covered / max(1, n)
    regularity = coverage   # for per-row extractor, coverage is regularity

    valid = (len(roles_final) >= 2 and coverage >= regularity_threshold)

    return {
        'roles':             roles_final,
        'row_assignments':   per_row_roles,
        'coverage':          round(coverage, 4),
        'regularity':        round(regularity, 4),
        'valid':             valid,
        'role_source':       'per_row_llm_extraction',
        'extractor_summary': dict(extractor_meta_summary),
        'per_row_audit':     per_row_audit,
        'group_name':        group_name,
    }

def discover_roles_via_sbert_phrase_clustering(
        texts: list,
        text_col_names: Optional[list],
        sbert_model,
        llm_role_namer_fn: Optional[Callable] = None,
        min_phrase_count: int = 2,
        min_role_size: int = 2,
        n_clusters_range: tuple = (2, 6),
        group_name: str = '',
        regularity_threshold: float = 0.40) -> dict:
    """
    Option D core: discover semantic-role schema for a group via
        SBERT phrase clustering  +  constrained-vocab LLM cluster naming.

    Pipeline (EDC / ZOES style):
      1. Extract repeated phrases per row (shared helper).
      2. Compute group regularity = fraction of rows with ≥2 mined phrases.
         If < regularity_threshold → return invalid (caller falls back to
         existing slot mining / FASTopic / NMF path).
      3. SBERT-embed each unique phrase.
      4. Agglomerative-cluster phrases by cosine similarity; select K by
         silhouette score (range 2..6).
      5. Name each cluster via LLM (constrained vocab).  Anonymous fallback
         when LLM is off or rejects.
      6. Per-row: assign one phrase per role (longest mined phrase wins ties).
      7. Return roles + row_assignments + diagnostics.

    Returns a dict in the same shape as mine_phrase_slots() so it can flow
    straight into build_slot_hierarchy / build_role_hierarchy.
    """
    n = len(texts)
    if n < 4 or sbert_model is None:
        return {'valid': False, 'reason': 'too_few_rows_or_no_sbert',
                'regularity': 0.0, 'roles': {}, 'row_assignments': []}

    # 1. extract phrases
    phrases, refined, regularity = _extract_phrases_for_role_clustering(
        texts, text_col_names, group_name, min_phrase_count)

    if regularity < regularity_threshold:
        return {'valid': False, 'reason': f'low_regularity ({regularity:.2f})',
                'regularity': regularity, 'roles': {}, 'row_assignments': []}
    if len(phrases) < 4:
        return {'valid': False, 'reason': 'too_few_phrases',
                'regularity': regularity, 'roles': {}, 'row_assignments': []}

    # 2. SBERT embed
    try:
        embs = sbert_model.encode(phrases, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=64)
    except Exception as e:
        return {'valid': False, 'reason': f'sbert_failed: {type(e).__name__}',
                'regularity': regularity, 'roles': {}, 'row_assignments': []}

    # 3. agglomerative + silhouette K selection
    best_score, best_labels, best_k = -1.0, None, 2
    for k in range(n_clusters_range[0],
                    min(n_clusters_range[1] + 1, len(phrases))):
        try:
            ac = AgglomerativeClustering(n_clusters=k, metric='cosine',
                                          linkage='average')
            labels = ac.fit_predict(embs)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(embs, labels, metric='cosine'))
            if sil > best_score:
                best_score, best_labels, best_k = sil, labels, k
        except Exception:
            continue

    if best_labels is None:
        return {'valid': False, 'reason': 'no_clusters',
                'regularity': regularity, 'roles': {}, 'row_assignments': []}

    # 4. group phrases by cluster id
    clusters_by_id: dict = defaultdict(list)
    for p, lbl in zip(phrases, best_labels):
        clusters_by_id[int(lbl)].append(p)
    valid_clusters = {cid: ps for cid, ps in clusters_by_id.items()
                      if len(ps) >= min_role_size}
    if len(valid_clusters) < 2:
        return {'valid': False, 'reason': 'too_few_valid_clusters',
                'regularity': regularity, 'roles': {}, 'row_assignments': []}

    # 5. name each cluster
    sample_descs = [str(t)[:200] for t in texts[:3]]
    roles: dict = {}
    naming_meta: dict = {}
    used_names: set = set()
    for cid, cluster_phrases in valid_clusters.items():
        role_name = None
        if llm_role_namer_fn:
            role_name, name_meta = llm_role_namer_fn(cluster_phrases, sample_descs)
            naming_meta[cid] = name_meta
        if not role_name:
            role_name = f'cluster_{cid}'
        # Disambiguate if LLM gave the same name to two clusters
        original = role_name
        suffix = 2
        while role_name in used_names:
            role_name = f'{original}_{suffix}'
            suffix += 1
        used_names.add(role_name)
        roles[role_name] = cluster_phrases

    # 6. per-row role assignment (longest phrase per role per row)
    phrase_to_role = {p: r for r, ps in roles.items() for p in ps}
    row_assignments: list = []
    covered = 0
    for r in refined:
        assignment: dict = {}
        for p in r:
            role = phrase_to_role.get(p)
            if not role:
                continue
            if role not in assignment or len(p) > len(assignment[role]):
                assignment[role] = p
        if assignment:
            covered += 1
        row_assignments.append(assignment)

    coverage = covered / n
    valid = (coverage >= 0.50 and len(roles) >= 2)

    return {
        'roles':             roles,
        'row_assignments':   row_assignments,
        'coverage':          round(coverage, 4),
        'regularity':        regularity,
        'valid':             valid,
        'role_source':       'sbert_phrase_clustering',
        'phrase_silhouette': round(float(best_score), 4),
        'n_clusters':        best_k,
        'naming_meta':       naming_meta,
    }

def build_role_hierarchy(vi_list: list,
                          role_result: dict,
                          can: pd.DataFrame,
                          parent_id: int,
                          parent_path: str,
                          nodes: list,
                          node_map: dict,
                          var_nodes: dict,
                          max_depth_remaining: int = 4,
                          post_split_fn: Optional[Callable] = None,
                          min_post_split_size: int = 4) -> bool:
    """
    Build a role-nested hierarchy from the Option D role decomposition.
    Roles ordered by generic priority:
        measure > outcome > statistic > condition > modifier > subtype
    Variables without a value for a given role skip that level.

    Returns True on success (≥ 2 aggregation nodes added), else False.
    """
    if not role_result.get('valid'):
        return False
    roles = role_result['roles']
    row_assigns = role_result['row_assignments']
    if len(vi_list) != len(row_assigns):
        return False

    role_names = sorted(roles.keys(),
                         key=lambda r: _ROLE_PRIORITY.get(r, 0),
                         reverse=True)
    if not role_names:
        return False

    aggregations_made = [0]
    vi_to_local = {vi: idx for idx, vi in enumerate(vi_list)}

    def _terminal_attach(vi_grp: list, parent: int, depth_left: int):
        if (post_split_fn is not None
                and len(vi_grp) >= min_post_split_size
                and depth_left > 0):
            try:
                added = post_split_fn(vi_grp, parent, depth_left)
                if added > 0:
                    aggregations_made[0] += added
                    return
            except Exception:
                pass
        for vi in vi_grp:
            _add_child(node_map, parent, var_nodes[vi])

    def _split(vi_subset: list, role_idx: int, current_parent: int,
                depth_remaining: int):
        if (depth_remaining <= 0 or role_idx >= len(role_names)
                or len(vi_subset) <= 1):
            _terminal_attach(vi_subset, current_parent, depth_remaining)
            return

        role = role_names[role_idx]
        groups: dict = defaultdict(list)
        unassigned: list = []
        for vi in vi_subset:
            local = vi_to_local.get(vi)
            val = row_assigns[local].get(role) if local is not None else None
            if val:
                groups[val].append(vi)
            else:
                unassigned.append(vi)

        if len(groups) <= 1:
            _split(vi_subset, role_idx + 1, current_parent, depth_remaining)
            return

        # BUGFIX: read role_source from the result so the same builder
        # correctly labels per-row LLM extraction nodes vs SBERT clustering
        # nodes.  Previously this was hardcoded to 'sbert_phrase_clustering',
        # hiding which route actually ran in the exported provenance.
        _route_src = role_result.get('role_source', 'sbert_phrase_clustering')
        _label_src = ('per_row_llm_role'
                       if _route_src == 'per_row_llm_extraction'
                       else 'sbert_phrase_role')
        _node_source_str = ('per-row LLM extraction (Zhu et al. 2025)'
                              if _route_src == 'per_row_llm_extraction'
                              else 'SBERT phrase cluster + LLM role naming')
        for val, vi_grp in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(vi_grp) == 1:
                _add_child(node_map, current_parent, var_nodes[vi_grp[0]])
                continue
            nid = _next_id(nodes)
            nd  = _make_node(nid, val.title(),
                              desc=(f'Role: {role} | Value: "{val}" | '
                                    f'Variables: {len(vi_grp)} | '
                                    f'Source: {_node_source_str}'))
            nd['label_provenance'] = {
                'label_source':  _label_src,
                'evidence_terms': [val],
                'confidence':    1.0,
                'llm_used':      True,
                'llm_rejected':  False,
                'role':          role,
            }
            nd['structure_provenance'] = {
                'route':            _route_src,
                'aspect_method':    _route_src,
                'slot_role':        role,
                'phrase_silhouette': role_result.get('phrase_silhouette'),
                'regularity':       role_result.get('regularity'),
                'n_clusters':       role_result.get('n_clusters'),
            }
            nodes.append(nd)
            node_map[nid] = nd
            _add_child(node_map, current_parent, nid)
            aggregations_made[0] += 1
            _split(vi_grp, role_idx + 1, nid, depth_remaining - 1)

        if unassigned:
            _terminal_attach(unassigned, current_parent, depth_remaining)

    _split(vi_list, 0, parent_id, max_depth_remaining)
    return aggregations_made[0] >= 2

def strip_group_prefix_from_labels(nodes: list) -> int:
    """
    Post-build pass: for every aggregation node, if its name starts with the
    parent group's name (case-insensitive), strip the prefix.

    Effect: 'DMS — Total Errors' under DMS becomes 'Total Errors'.
            'Pal Total Errors' under PAL becomes 'Total Errors'.

    Returns the number of labels modified.
    """
    node_map = {int(n['id']): n for n in nodes}
    modified = 0

    def _walk(nid: int, parent_name: str):
        nonlocal modified
        n = node_map.get(int(nid))
        if not n:
            return
        if n.get('type') == 'aggregation' and parent_name:
            current = str(n.get('name', '')).strip()
            cn_lower = current.lower()
            pn_lower = parent_name.strip().lower()
            if pn_lower and (cn_lower.startswith(pn_lower + ' ')
                              or cn_lower.startswith(pn_lower + '—')
                              or cn_lower.startswith(pn_lower + '-')
                              or cn_lower.startswith(pn_lower + ':')
                              or cn_lower.startswith(pn_lower + '/')):
                stripped = current[len(parent_name):].lstrip(' —-—:/').strip()
                if stripped and len(stripped) >= 2:
                    n['name'] = stripped
                    modified += 1
        new_parent = (n.get('name', '') if n.get('type') in ('aggregation', 'root')
                       else parent_name)
        for cid in n.get('related', []):
            _walk(int(cid), new_parent)

    _walk(0, '')
    return modified

def enforce_single_parent(nodes: list) -> int:
    """
    POST-BUILD PASS 4 — guarantee the result is a tree (each node has exactly
    one parent).

    The role builder can attach a variable both directly to a group and again
    under a sub-aggregation of that same group — e.g. a leaf under '3 Targets'
    *and* under '3 Targets > False Alarm Sequences'.  That makes the branch a
    DAG, which (a) diverges from the single-parent gold standard, (b) fragments
    the branch, and (c) breaks proportional ('total') sunburst/treemap sizing.

    For every node with more than one parent, keep the MOST SPECIFIC (deepest)
    parent and detach it from the shallower ones.  Keeping the deepest parent
    removes the redundant direct attachment while preserving the finer
    sub-grouping the role extractor discovered.  Fully generic — no domain
    knowledge, no hardcoded labels.

    Returns the number of redundant parent links removed.
    """
    from collections import deque
    node_map = {int(n['id']): n for n in nodes}
    # depth = shortest distance from root (id 0) along child edges
    depth = {0: 0}
    dq = deque([0])
    while dq:
        cur = dq.popleft()
        for c in node_map.get(cur, {}).get('related', []):
            c = int(c)
            if c not in depth:
                depth[c] = depth[cur] + 1
                dq.append(c)
    parents = defaultdict(list)
    for n in nodes:
        for c in n.get('related', []):
            parents[int(c)].append(int(n['id']))
    removed = 0
    for child, ps in parents.items():
        if len(ps) <= 1:
            continue
        keep = max(ps, key=lambda p: depth.get(p, 0))  # deepest = most specific
        for p in ps:
            if p == keep:
                continue
            par = node_map.get(p)
            if par and int(child) in par['related']:
                par['related'] = [x for x in par['related'] if int(x) != int(child)]
                removed += 1
    return removed

def mine_phrase_slots(texts: list,
                       text_col_names: Optional[list] = None,
                       min_phrase_count: int = 2,
                       min_slot_size: int = 2,
                       coverage_threshold: float = 0.55,
                       excl_threshold: float = 0.75,
                       group_name: str = '',
                       llm_role_classifier_fn: Optional[Callable] = None) -> dict:
    """
    Discover phrase slots in a group of variable descriptions.

    Algorithm:
      1. Extract concept-prefix from each description (text before ':' in a
         description-like column; full text if no such column).
      2. Tokenise + lowercase; generate n-grams (1–4 tokens, ≥3 chars).
      3. Keep n-grams that appear in ≥ min_phrase_count rows.
      4. For each row, retain only maximal phrases (drop sub-phrases of
         longer phrases present in the same row).
      5. Compute mutual-exclusion score per phrase pair:
            M[a,b] = 1 − cooc[a,b] / min(count[a], count[b])
         Phrases with M[a,b] ≥ excl_threshold are 'alternatives' (rarely
         appear together → likely fill the same slot in different rows).
      6. Cluster phrases into slots via greedy mutual-exclusion BFS, starting
         from the most-frequent phrase.
      7. A slot is valid if it has ≥ min_slot_size distinct phrases.
      8. Coverage: fraction of rows that contain ≥1 phrase from ≥1 slot.
         If coverage < threshold or < 2 slots survive → valid=False.

    Returns:
      {
        'slots':            list[ {phrases: set[str], best: str} ],
        'row_assignments':  list[ dict[slot_id → phrase] ] for each row,
        'coverage':         float in [0,1],
        'valid':            bool — True if slot structure is strong enough.
      }
    """
    n = len(texts)
    if n < 4:
        return {'slots': [], 'row_assignments': [], 'coverage': 0.0, 'valid': False}

    text_keys = ({c.strip().lower() for c in (text_col_names or [])}
                 if text_col_names else None)

    # ── 1. extract concept prefixes ──────────────────────────────────────────
    prefixes = []
    for t in texts:
        prefix = ''
        for chunk in str(t).split(' | '):
            if ':' not in chunk:
                continue
            key, val = chunk.split(':', 1)
            key_l = key.strip().lower()
            if text_keys is not None and key_l not in text_keys:
                continue
            prefix = val.split(':')[0].strip()
            break
        if not prefix:
            prefix = str(t).strip()
        # Strip 'KEY:', 'Note:', etc. before tokenising
        prefix = _strip_leading_markers(prefix)
        prefixes.append(prefix)

    # ── 2. normalise: alphanumerics + parens preserved, others → spaces ──────
    def _normalize(p: str) -> str:
        s = ''.join(ch if ch.isalnum() or ch == ' ' else ' '
                    for ch in str(p).lower())
        return ' '.join(s.split())

    norm = [_normalize(p) for p in prefixes]

    # ── 3. extract n-grams (1..4) per row, keep ≥3 chars ─────────────────────
    row_phrases: list = []
    phrase_count: Counter = Counter()
    for text in norm:
        tokens = text.split()
        row_set = set()
        for ngram_n in range(1, 5):
            for i in range(len(tokens) - ngram_n + 1):
                phrase = ' '.join(tokens[i:i + ngram_n])
                if len(phrase) >= 3:
                    row_set.add(phrase)
        row_phrases.append(row_set)
        for p in row_set:
            phrase_count[p] += 1

    # ── 4. keep repeated phrases, retain only maximal phrases per row ────────
    keep = {p for p, c in phrase_count.items() if c >= min_phrase_count}
    refined = []
    for row in row_phrases:
        sorted_by_len = sorted(row & keep, key=lambda p: -len(p.split()))
        kept: list = []
        for p in sorted_by_len:
            if not any(p != q and p in q for q in kept):
                kept.append(p)
        refined.append(set(kept))

    # Recount after refinement, re-filter
    phrase_count = Counter()
    for r in refined:
        for p in r:
            phrase_count[p] += 1
    # NEW: filter out non-meaningful phrases (stopwords, group-name echoes,
    # dictionary markers) before they enter mutual-exclusion clustering.
    group_lower = (group_name or '').strip().lower()
    phrases = sorted([p for p, c in phrase_count.items()
                       if c >= min_phrase_count
                       and _phrase_is_meaningful(p, group_lower)])
    if len(phrases) < 2 * min_slot_size:
        return {'slots': [], 'row_assignments': [], 'coverage': 0.0, 'valid': False}

    p_idx  = {p: i for i, p in enumerate(phrases)}
    n_p    = len(phrases)
    counts = np.array([phrase_count[p] for p in phrases])

    # ── 5. co-occurrence + mutual-exclusion matrix ───────────────────────────
    cooc = np.zeros((n_p, n_p), dtype=int)
    for r in refined:
        idxs = [p_idx[p] for p in r if p in p_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    cooc[i, j] += 1
    min_counts = np.minimum.outer(counts, counts).astype(float)
    min_counts[min_counts == 0] = 1.0
    mut_excl   = 1.0 - cooc / min_counts
    np.fill_diagonal(mut_excl, 0)

    # ── 6a. UPSTREAM ROUTE: ask the LLM to classify phrases into roles ───────
    # The LLM proposes a role schema (e.g. {measure: [...], statistic: [...],
    # condition: [...]}) — phrases are assigned to roles, role names provide
    # semantic ordering for the hierarchy.  Anti-hallucination: every phrase
    # must come back verbatim, otherwise rejected by the validator inside
    # make_llm_role_classifier_fn.
    slot_source = 'mutual_exclusion'
    role_names: list = []
    slots: list = []  # list[set[int]]  — phrase indices per slot

    if llm_role_classifier_fn is not None:
        try:
            classified, classifier_meta = llm_role_classifier_fn(
                phrases, texts, group_name)
        except Exception:
            classified, classifier_meta = None, {'reason': 'exception'}
        if classified:
            # Build slots in the order the LLM proposed them.  Each role is
            # one slot containing the phrase-index set.
            for role_name, role_phrases in classified.items():
                idx_set = {p_idx[p] for p in role_phrases if p in p_idx}
                if len(idx_set) >= min_slot_size:
                    slots.append(idx_set)
                    role_names.append(role_name)
            if len(slots) >= 2:
                slot_source = 'llm_role_classification'

    # ── 6b. FALLBACK: greedy mutual-exclusion BFS ────────────────────────────
    if slot_source == 'mutual_exclusion':
        visited: set = set()
        slots = []
        order   = np.argsort(-counts)
        for seed in order:
            if seed in visited:
                continue
            slot = {int(seed)}
            queue = [int(seed)]
            while queue:
                cur = queue.pop()
                for j in np.where(mut_excl[cur] >= excl_threshold)[0]:
                    j = int(j)
                    if j in slot:
                        continue
                    if all(mut_excl[j, k] >= excl_threshold - 0.15 for k in slot):
                        slot.add(j)
                        queue.append(j)
            if len(slot) >= min_slot_size:
                slots.append(slot)
                visited |= slot
        # Synthesise anonymous role names from the most-frequent phrase in each
        # slot — these become the visible slot tags in provenance.
        role_names = [phrases[max(s, key=lambda i: counts[i])] for s in slots]

    if len(slots) < 2:
        return {'slots': [], 'row_assignments': [], 'coverage': 0.0, 'valid': False}

    # ── 7. assign per-row phrase per slot ────────────────────────────────────
    row_assignments: list = []
    covered = 0
    for r in refined:
        row_idx = {p_idx[p] for p in r if p in p_idx}
        assignment: dict = {}
        any_match = False
        for slot_id, slot in enumerate(slots):
            matched = row_idx & slot
            if matched:
                best = max(matched, key=lambda i: counts[i])
                assignment[slot_id] = phrases[best]
                any_match = True
        if any_match:
            covered += 1
        row_assignments.append(assignment)

    coverage = covered / n
    valid    = (coverage >= coverage_threshold and len(slots) >= 2)

    return {
        'slots': [{'phrases':  {phrases[i] for i in s},
                    'best':     phrases[max(s, key=lambda i: counts[i])],
                    'role_name': role_names[idx] if idx < len(role_names) else ''}
                   for idx, s in enumerate(slots)],
        'row_assignments': row_assignments,
        'coverage':        round(coverage, 4),
        'valid':           valid,
        'slot_source':     slot_source,    # 'llm_role_classification' or 'mutual_exclusion'
    }


def build_slot_hierarchy(vi_list: list,
                          slot_result: dict,
                          can: pd.DataFrame,
                          parent_id: int,
                          parent_path: str,
                          nodes: list,
                          node_map: dict,
                          var_nodes: dict,
                          max_depth_remaining: int = 4,
                          post_slot_split_fn: Optional[Callable] = None,
                          min_post_slot_size: int = 4) -> bool:
    """
    Build a hierarchy for `vi_list` using inferred phrase slots.

    Slots are ordered by partition quality (more distinct values + fewer
    singletons = higher priority).  Each slot becomes one tree level.
    Variables that lack a phrase at a given slot level skip that level.
    Returns True on success; False if the result is too shallow to be useful
    (caller should then fall back to NMF/GMM).
    """
    if not slot_result.get('valid'):
        return False

    slots       = slot_result['slots']
    row_assigns = slot_result['row_assignments']
    if len(vi_list) != len(row_assigns):
        return False

    # Index map: global vi → local position in row_assigns
    vi_to_local = {vi: idx for idx, vi in enumerate(vi_list)}

    # Order slots by partition-quality + semantic-shape heuristics.
    # Higher score → used at a shallower level in the hierarchy.
    #
    # Heuristics (all generic, no domain knowledge):
    #   + many distinct values, low singleton fraction (existing)
    #   + average phrase token-length (multi-word noun phrases preferred)
    #   + total row coverage of the slot
    #   − slots whose top phrases look like pure conditions
    #     (numeric token + temporal/quantity word)
    #   − slots where every phrase is just a number or 'all' / 'none' modifier
    _CONDITION_HINTS = {'second', 'seconds', 'minute', 'minutes', 'hour',
                        'hours', 'day', 'days', 'month', 'months', 'year',
                        'years', 'week', 'weeks', 'box', 'boxes', 'token',
                        'tokens', 'pattern', 'patterns', 'trial', 'trials'}

    def _looks_like_condition(phrase: str) -> bool:
        toks = phrase.split()
        if not toks:
            return False
        has_num = any(t.isdigit() or t in {'all','none','simultaneous'} for t in toks)
        has_hint = any(t in _CONDITION_HINTS for t in toks)
        return has_num and has_hint

    def _slot_score(slot_id: int) -> float:
        vals = Counter()
        for a in row_assigns:
            v = a.get(slot_id)
            if v:
                vals[v] += 1
        if not vals:
            return -1.0
        n_distinct   = len(vals)
        n_singletons = sum(1 for c in vals.values() if c == 1)
        coverage     = sum(vals.values()) / max(1, len(row_assigns))
        avg_tokens   = float(np.mean([len(p.split()) for p in vals]))
        condition_frac = sum(1 for p in vals if _looks_like_condition(p)) / n_distinct

        base = n_distinct - 0.6 * n_singletons
        base += 0.4 * coverage
        base += 0.3 * (avg_tokens - 1)
        base -= 0.8 * condition_frac
        return base

    # When the LLM produced the role schema, trust its role ordering for slots
    # whose role name is broadly "measure-like" (base concept) over
    # "statistic/condition/subtype" (modifiers).  This is generic English
    # vocabulary, not domain knowledge — same heuristic used by IE slot-
    # induction work (cf. Xu et al., FCS 2024 IE survey).
    slot_source = slot_result.get('slot_source', 'mutual_exclusion')
    _MEASURE_LIKE = {'measure', 'outcome', 'metric', 'variable', 'quantity'}
    _STATISTIC_LIKE = {'statistic', 'stat', 'aggregate', 'summary'}
    _CONDITION_LIKE = {'condition', 'modifier', 'context', 'setting'}
    _SUBTYPE_LIKE   = {'subtype', 'type', 'kind', 'category'}

    def _role_priority(role: str) -> int:
        r = (role or '').strip().lower()
        if any(k in r for k in _MEASURE_LIKE):   return 4
        if any(k in r for k in _STATISTIC_LIKE): return 3
        if any(k in r for k in _CONDITION_LIKE): return 2
        if any(k in r for k in _SUBTYPE_LIKE):   return 1
        return 0   # unknown role — fall back to data-driven score

    if slot_source == 'llm_role_classification':
        ordered_slots = sorted(
            range(len(slots)),
            key=lambda i: (_role_priority(slots[i].get('role_name', '')),
                            _slot_score(i)),
            reverse=True)
    else:
        ordered_slots = sorted(range(len(slots)), key=_slot_score, reverse=True)
    ordered_slots = [s for s in ordered_slots if _slot_score(s) > 0]
    if not ordered_slots:
        return False

    aggregations_made = [0]   # mutable counter for fallback decision

    def _attach_or_sub_recurse(vi_grp: list, parent: int, depth_left: int):
        """
        Terminal-leaf attach point inside slot mining.  When slots are
        exhausted but the cluster still has enough variables AND we have a
        post-slot callback (NMF/FASTopic + GMM splitter), recurse further to
        deepen the tree.  Otherwise attach leaves directly.
        """
        if (post_slot_split_fn is not None
                and len(vi_grp) >= min_post_slot_size
                and depth_left > 0):
            try:
                added = post_slot_split_fn(vi_grp, parent, depth_left)
                if added > 0:
                    aggregations_made[0] += added
                    return
            except Exception:
                pass
        for vi in vi_grp:
            _add_child(node_map, parent, var_nodes[vi])

    def _split(vi_subset: list, slot_ord_pos: int, current_parent: int,
               depth_remaining: int):
        if (depth_remaining <= 0 or slot_ord_pos >= len(ordered_slots)
                or len(vi_subset) <= 1):
            _attach_or_sub_recurse(vi_subset, current_parent, depth_remaining)
            return

        slot_id = ordered_slots[slot_ord_pos]
        groups: dict = defaultdict(list)
        unassigned: list = []
        for vi in vi_subset:
            local = vi_to_local.get(vi)
            val = row_assigns[local].get(slot_id) if local is not None else None
            if val:
                groups[val].append(vi)
            else:
                unassigned.append(vi)

        # If this slot doesn't partition the subset, move to next slot
        if len(groups) <= 1:
            _split(vi_subset, slot_ord_pos + 1, current_parent, depth_remaining)
            return

        # The slot's role name (from LLM classification, when applicable)
        slot_role = slots[slot_id].get('role_name', '') if slot_id < len(slots) else ''
        for val, vi_grp in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(vi_grp) == 1:
                _add_child(node_map, current_parent, var_nodes[vi_grp[0]])
                continue
            nid = _next_id(nodes)
            nd  = _make_node(nid, val.title(),
                             desc=(f'Role: {slot_role or "—"} | '
                                   f'Slot phrase: "{val}" | '
                                   f'Variables: {len(vi_grp)} | '
                                   f'Source: phrase-slot mining ({slot_source})'))
            nd['label_provenance'] = {
                'label_source': 'phrase_slot',
                'evidence_terms': [val],
                'confidence': 1.0,
                'llm_used': slot_source == 'llm_role_classification',
                'llm_rejected': False,
                'role':        slot_role,
            }
            nd['structure_provenance'] = {
                'route':           'slot_mining',
                'aspect_method':   slot_source,
                'silhouette':      None,
                'slot_coverage':   round(float(slot_result.get('coverage', 0)), 3),
                'slot_role':       slot_role,
            }
            nodes.append(nd); node_map[nid] = nd
            _add_child(node_map, current_parent, nid)
            aggregations_made[0] += 1
            _split(vi_grp, slot_ord_pos + 1, nid, depth_remaining - 1)

        if unassigned:
            _attach_or_sub_recurse(unassigned, current_parent, depth_remaining)

    _split(vi_list, 0, parent_id, max_depth_remaining)

    # Reject the slot-built tree if it added almost no structure (likely the
    # slots were not actually useful for this group).
    return aggregations_made[0] >= 2

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6  — DYNAMIC TOP-DOWN LOD TREE  [ZHU §3.3 adapted]
# ──────────────────────────────────────────────────────────────────────────────
def _next_id(nodes: list) -> int:
    return max((int(n['id']) for n in nodes), default=0) + 1

def _add_child(node_map: dict, parent_id: int, child_id: int):
    p = node_map.get(int(parent_id))
    if p and int(child_id) not in p['related']:
        p['related'].append(int(child_id))

def _make_node(nid, name, ntype='aggregation', desc='', dtype='determine') -> dict:
    return {'id': int(nid), 'name': str(name), 'related': [],
            'type': ntype, 'desc': str(desc), 'dtype': dtype, 'isShown': True}

# ──────────────────────────────────────────────────────────────────────────────
# POST-BUILD PASS 1  — SIBLING COMMON-PREFIX FACTORING
# ──────────────────────────────────────────────────────────────────────────────
def factor_sibling_common_prefixes(nodes: list,
                                     min_siblings: int = 3,
                                     min_prefix_tokens: int = 2) -> int:
    """
    For each parent whose ≥`min_siblings` aggregation children share a
    common multi-token title prefix, insert a new intermediate parent named
    by that prefix and re-attach the matching siblings under it (with the
    prefix stripped from each name).

    Generic, no domain knowledge.  Inspired by sibling-label factoring
    common in faceted-classification systems (Stoica & Hearst, NAACL 2007,
    'Castanet') — collapsing redundant repeated tokens in sibling names.

    Returns the number of factor-parents inserted.
    """
    node_map = {int(n['id']): n for n in nodes}
    inserted = 0
    # We iterate over a snapshot of current aggregation nodes
    queue = [int(n['id']) for n in nodes
             if n.get('type') in ('aggregation', 'root')]
    while queue:
        parent_id = queue.pop(0)
        parent = node_map.get(parent_id)
        if not parent:
            continue
        # Gather aggregation children with their tokenised names
        agg_children = []
        for cid in parent.get('related', []):
            child = node_map.get(int(cid))
            if not child or child.get('type') != 'aggregation':
                continue
            toks = str(child.get('name', '')).split()
            if len(toks) >= min_prefix_tokens:
                agg_children.append((int(cid), toks))
        if len(agg_children) < min_siblings:
            continue
        # Greedy: find the longest prefix shared by ≥ min_siblings children
        best_prefix: list = []
        best_group: list = []
        # Sort children by name tokens for stable grouping
        agg_children.sort(key=lambda x: x[1])
        # Try each possible prefix length from longest down
        max_len = max(len(t) for _, t in agg_children)
        for length in range(max_len, min_prefix_tokens - 1, -1):
            prefix_counts: Counter = Counter()
            for cid, toks in agg_children:
                if len(toks) > length:   # must have something AFTER the prefix
                    prefix_counts[tuple(t.lower() for t in toks[:length])] += 1
            for pfx, cnt in prefix_counts.most_common():
                if cnt >= min_siblings:
                    # Reject low-quality prefixes: all-stopword, or starting/
                    # ending with a stopword (e.g. "the number of", "the").
                    # A good factored parent is a real concept phrase.
                    pfx_l = [t.lower() for t in pfx]
                    if all(t in _STOPWORDS for t in pfx_l):
                        continue
                    if pfx_l[0] in _STOPWORDS or pfx_l[-1] in _STOPWORDS:
                        continue
                    group = [(cid, toks) for cid, toks in agg_children
                              if len(toks) > length
                              and tuple(t.lower() for t in toks[:length]) == pfx]
                    if len(group) >= min_siblings:
                        best_prefix = list(pfx)
                        best_group  = group
                        break
            if best_prefix:
                break
        if not best_prefix:
            continue

        # Build the new intermediate parent
        new_id = max(node_map) + 1
        # Title-case the prefix using the original child capitalisation
        # (take it from the first matched child's tokens)
        orig_tokens = best_group[0][1][:len(best_prefix)]
        prefix_name = ' '.join(orig_tokens)
        new_node = _make_node(new_id, prefix_name, ntype='aggregation',
                               desc=(f'Factored common prefix: "{prefix_name}" | '
                                     f'Siblings: {len(best_group)} | '
                                     f'Source: sibling factoring [Castanet 2007]'))
        new_node['label_provenance'] = {
            'label_source':     'factored_common_prefix',
            'evidence_terms':   [c[0] for c in best_group],
            'confidence':       1.0,
            'llm_used':         False,
            'llm_rejected':     False,
        }
        new_node['structure_provenance'] = {
            'route':            'sibling_factoring',
            'aspect_method':    None,
            'silhouette':       None,
            'slot_coverage':    None,
            'factored_from':    [c[0] for c in best_group],
            'common_prefix_tokens': len(best_prefix),
        }
        # Rename the factored siblings (strip the prefix from their names)
        for cid, toks in best_group:
            child = node_map[cid]
            new_name = ' '.join(toks[len(best_prefix):]).strip()
            if new_name:
                child['name'] = new_name
        # Rewire parent → new_node → factored siblings
        moved_ids = {c[0] for c in best_group}
        parent['related'] = [c for c in parent['related']
                             if int(c) not in moved_ids]
        parent['related'].append(new_id)
        new_node['related'] = [c[0] for c in best_group]
        nodes.append(new_node)
        node_map[new_id] = new_node
        inserted += 1
        # Re-examine this parent in case multiple prefix groups exist
        queue.append(parent_id)
        # Also examine the new parent for further nesting
        queue.append(new_id)
    return inserted

# ──────────────────────────────────────────────────────────────────────────────
# POST-BUILD PASS 2  — TRACO-INSPIRED LOW-QUALITY NODE PRUNING
# ──────────────────────────────────────────────────────────────────────────────
def prune_low_quality_aggregations(nodes: list,
                                     tfidf: TfidfVectorizer = None,
                                     min_coherence: float = 0.0,
                                     max_child_ratio: float = 1.0) -> int:
    """
    Conservative noise pruning.

    ONLY rule applied by default: dissolve aggregation nodes whose name is
    pure noise — single-word stopword titles ('And', 'Them'), dictionary
    markers ('Key', 'Note'), or all-stopword titles ('To Be', 'Have A Lot Of').

    Group anchors (route='group_anchor') and sibling-factoring nodes
    (route='sibling_factoring') are NEVER pruned — they are structural and
    legitimately have short names that may not score well on TF-IDF metrics.

    Rules B (parent-child coherence) and C (lopsided split) are intentionally
    DISABLED by default — both metrics punish good hierarchies where children
    are legitimately more specific than parents (low TF-IDF cosine) or where
    a slot mining branch happens to be dominated by one large subgroup.
    They can be opted into by passing tfidf and tightening the thresholds.

    Inspired by TraCo (Wu et al., AAAI 2024): affinity / rationality /
    diversity failures are common in hierarchical topic models.  Here we
    enforce the most conservative form of that — only obviously-noise names.

    Returns the number of nodes dissolved.
    """
    node_map = {int(n['id']): n for n in nodes}
    parent_of: dict = {}
    for n in nodes:
        for c in n.get('related', []):
            parent_of.setdefault(int(c), int(n['id']))

    FORBIDDEN = frozenset(_STOPWORDS) | frozenset({
        'key', 'note', 'item', 'label', 'group', 'cluster',
    })
    # Routes that are ALWAYS structural — never prune these even if the name
    # is short / stopword-like (e.g. 'MOT' is a group anchor, not noise).
    PROTECTED_ROUTES = frozenset({'group_anchor', 'sibling_factoring'})

    to_dissolve: list = []
    for n in nodes:
        if n.get('type') != 'aggregation':
            continue
        nid = int(n['id'])
        if nid not in parent_of:
            continue
        sp = n.get('structure_provenance', {})
        if sp.get('route') in PROTECTED_ROUTES:
            continue

        name_lower = str(n.get('name', '')).strip().lower()
        if not name_lower:
            to_dissolve.append(nid)
            continue

        # Only Rule A is enforced by default.  A token is "noise" if it is a
        # stopword/marker OR shorter than 2 chars AND not a digit (numeric
        # labels like '2' / '4' / '6' / '8' are kept — they may be meaningful,
        # e.g. PAL pattern counts).
        toks = name_lower.split()
        def _is_noise_tok(t: str) -> bool:
            if t in FORBIDDEN:
                return True
            if len(t) < 2 and not t.isdigit():
                return True
            return False
        all_noise = bool(toks) and all(_is_noise_tok(t) for t in toks)
        if name_lower in FORBIDDEN or all_noise:
            to_dissolve.append(nid)

    # Dissolve: promote children to grandparent
    dissolved = 0
    for nid in to_dissolve:
        n = node_map.get(nid)
        if not n:
            continue
        gp_id = parent_of.get(nid)
        if gp_id is None:
            continue
        gp = node_map.get(gp_id)
        if not gp:
            continue
        gp['related'] = [c for c in gp['related'] if int(c) != nid]
        for cid in n.get('related', []):
            if int(cid) not in gp['related']:
                gp['related'].append(int(cid))
            parent_of[int(cid)] = gp_id
        n['isShown'] = False
        n['type']    = 'dissolved'
        n.setdefault('structure_provenance', {})['dissolved_reason'] = \
            'noise_label_only'
        dissolved += 1
    return dissolved

def build_dynamic_lod_tree(can: pd.DataFrame,
                            aspect_reprs: list,
                            aspect_labels: list,
                            tfidf: TfidfVectorizer,
                            max_depth: int = 6,
                            min_cluster_size: int = 2,
                            sil_threshold: float = 0.04,
                            max_clusters_per_split: int = 8,
                            project: str = 'project',
                            local_nmf: bool = True,
                            min_local_nmf_size: int = 8,
                            max_aspects: int = 10,
                            sbert_model=None,
                            llm_label_fn: Optional[Callable] = None,
                            use_slot_mining: bool = True,
                            text_col_names: Optional[list] = None,
                            use_fastopic: bool = True,
                            fastopic_min_size: int = 8,
                            llm_role_classifier_fn: Optional[Callable] = None,
                            use_role_decomposition: bool = True,
                            llm_role_namer_fn: Optional[Callable] = None,
                            role_regularity_threshold: float = 0.40,
                            per_row_role_extractor_fn: Optional[Callable] = None,
                            use_per_row_role_extraction: bool = True) -> list:
    """
    Build a dynamic top-down LoD tree.

    Entry strategy (new — fully data-driven, no hardcoding):
      • If detected _group metadata provides L1/L2 structure, materialise those
        path segments as aggregation nodes first.
      • Then apply NMF aspect discovery and GMM clustering *locally* inside each
        terminal group (not globally across all variables).
      • Falls back to global NMF from root when no group structure is found.

    Within each recursive split [ZHU §3.3 adapted]:
      • Evaluate K aspects by silhouette — select the highest  (simplified
        best-aspect split; not the full probabilistic search of Zhu Eq.6/7).
      • Single-variable clusters are attached directly without an aggregation
        wrapper (singleton prevention).
    """
    texts  = can['_text'].fillna('').astype(str).tolist()
    n_vars = len(can)

    # ── build leaf attribute nodes (ids 1..n_vars) ───────────────────────────
    nodes: list    = [_make_node(0, project, ntype='root', desc='Root node')]
    var_nodes: dict = {}   # can positional index → node id
    for i, (_, row) in enumerate(can.iterrows(), start=1):
        nd = _make_node(i, row['_label'], ntype='attribute',
                        desc=row['_text'], dtype='determine')
        nd['metadata'] = {'row_index': int(row['_row']), 'group': row['_group']}
        nodes.append(nd)
        var_nodes[int(row.name)] = i
    node_map: dict = {int(n['id']): n for n in nodes}

    # ── recursive splitter ────────────────────────────────────────────────────
    def _recurse(vi_global: list,
                 cur_reprs: list,
                 cur_labels: list,
                 cur_tfidf: TfidfVectorizer,
                 parent_id: int,
                 depth: int,
                 parent_path: str,
                 aspect_method_tag: str = 'nmf'):
        """
        vi_global  : global positional indices into `can` for this node's variables.
        cur_reprs  : list of K arrays, each shape (len(vi_global), d).
                     Rows correspond positionally to vi_global — no global indexing.
        cur_labels : NMF aspect labels for cur_reprs.
        cur_tfidf  : TF-IDF vectorizer fitted on this scope's texts.
        """
        if depth >= max_depth or len(vi_global) < min_cluster_size:
            for vi in vi_global:
                _add_child(node_map, parent_id, var_nodes[vi])
            return

        # NEW: shortcut for tiny homogeneous clusters — same _group, ≤3 vars.
        # Avoids spending GMM/silhouette evaluation on already-meaningful leaves.
        if len(vi_global) <= 3:
            groups_here = {str(can.iloc[vi]['_group']) for vi in vi_global}
            if len(groups_here) == 1:
                for vi in vi_global:
                    _add_child(node_map, parent_id, var_nodes[vi])
                return

        sub_texts = [texts[vi] for vi in vi_global]

        # Evaluate every aspect — pick the one with highest silhouette [ZHU §3.3]
        best_sil, best_k_idx, best_lbls = -1.0, -1, None
        for k_idx, rep in enumerate(cur_reprs):
            if rep.shape[0] < 3:
                continue
            lbls, _, sil = cluster_aspect_gmm(
                rep, max_k=min(max_clusters_per_split, len(vi_global) // 2))
            if sil > best_sil:
                best_sil, best_k_idx, best_lbls = sil, k_idx, lbls

        if best_k_idx == -1 or best_sil < sil_threshold or best_lbls is None:
            for vi in vi_global:
                _add_child(node_map, parent_id, var_nodes[vi])
            return

        # NEW: split-quality rejection (TraCo-inspired structural check).
        # Reject splits that are extremely imbalanced or mostly singletons —
        # silhouette can be high even when one cluster swallows most variables.
        cluster_sizes = list(Counter(best_lbls).values())
        max_ratio     = max(cluster_sizes) / sum(cluster_sizes)
        n_singletons  = sum(1 for s in cluster_sizes if s == 1)
        if max_ratio > 0.85 or n_singletons > len(cluster_sizes) // 2:
            for vi in vi_global:
                _add_child(node_map, parent_id, var_nodes[vi])
            return

        # Group variables by cluster assignment
        cluster_to_global: dict = defaultdict(list)
        for local_pos, (vi, cl) in enumerate(zip(vi_global, best_lbls)):
            cluster_to_global[int(cl)].append(vi)

        aspect_name     = cur_labels[best_k_idx]
        global_to_local = {vi: idx for idx, vi in enumerate(vi_global)}

        for cl_id, cl_global in cluster_to_global.items():
            if not cl_global:
                continue

            # Singleton prevention: attach single-variable clusters directly [NEW]
            if len(cl_global) == 1:
                _add_child(node_map, parent_id, var_nodes[cl_global[0]])
                continue

            cl_texts  = [texts[vi] for vi in cl_global]
            cl_groups = [str(can.iloc[vi]['_group']) for vi in cl_global]
            lbl, prov = label_cluster(cl_texts, sub_texts, cur_tfidf,
                                       cluster_groups=cl_groups,
                                       parent_path=parent_path,
                                       llm_label_fn=llm_label_fn,
                                       return_provenance=True)
            desc = (f'Aspect: {aspect_name} | '
                    f'Silhouette: {best_sil:.3f} | '
                    f'Variables: {len(cl_global)}')
            nid = _next_id(nodes)
            nd  = _make_node(nid, lbl, desc=desc)
            nd['label_provenance'] = prov
            nd['structure_provenance'] = {
                'route':           'aspect_clustering',
                'aspect_method':   aspect_method_tag,
                'silhouette':      round(float(best_sil), 4),
                'slot_coverage':   None,
            }
            nodes.append(nd)
            node_map[nid] = nd
            _add_child(node_map, parent_id, nid)

            # Slice embeddings to this cluster's local positions and recurse
            cl_pos     = [global_to_local[vi] for vi in cl_global]
            sub_reprs  = [rep[cl_pos] for rep in cur_reprs]
            _recurse(cl_global, sub_reprs, cur_labels, cur_tfidf,
                     nid, depth + 1, f'{parent_path} > {lbl}',
                     aspect_method_tag=aspect_method_tag)

    # ── group-anchored entry  ─────────────────────────────────────────────────
    # The _group column is built from user-confirmed group_cols — fully data-driven.
    # We materialise each path segment as an aggregation node, then run NMF/GMM
    # only inside each terminal group.

    group_node_ids: dict = {}   # path string → node id

    def _get_or_create_group_node(path: str, parent_id: int) -> int:
        if path in group_node_ids:
            return group_node_ids[path]
        seg_name = path.split(' > ')[-1].strip()
        nid = _next_id(nodes)
        nd  = _make_node(nid, seg_name, ntype='aggregation', desc=f'Group: {path}')
        nd['structure_provenance'] = {
            'route':           'group_anchor',
            'aspect_method':   None,
            'silhouette':      None,
            'slot_coverage':   None,
        }
        nodes.append(nd)
        node_map[nid] = nd
        _add_child(node_map, parent_id, nid)
        group_node_ids[path] = nid
        return nid

    # Collect global variable positions per terminal group path
    terminal_groups: dict = defaultdict(list)
    for vi, (_, row) in enumerate(can.iterrows()):
        terminal_groups[str(row['_group'])].append(vi)

    non_ungrouped = [g for g in terminal_groups if g.strip().lower() != 'ungrouped']

    if not non_ungrouped:
        # No group structure detected → fall back to global NMF from root
        full_reprs = [rep[list(range(n_vars))] for rep in aspect_reprs]
        _recurse(list(range(n_vars)), full_reprs, aspect_labels, tfidf,
                 0, 0, project)
    else:
        # Optional Streamlit progress bar — visible feedback for large datasets
        # like HCP where the per-group loop dominates runtime.
        try:
            import streamlit as _st_progress
            _pbar = _st_progress.progress(0.0, text='Building groups…')
            _show_progress = True
        except Exception:
            _pbar = None
            _show_progress = False

        sorted_groups = sorted(terminal_groups.items())
        n_groups_total = len(sorted_groups)

        for _g_idx, (group_path, vi_list) in enumerate(sorted_groups):
            if _show_progress and _pbar is not None:
                try:
                    short_path = group_path[:60] + ('…' if len(group_path) > 60 else '')
                    _pbar.progress((_g_idx + 1) / max(1, n_groups_total),
                                    text=f'[{_g_idx + 1}/{n_groups_total}] '
                                         f'{short_path}  ({len(vi_list)} vars)')
                except Exception:
                    pass
            # Build L1/L2 path nodes from detected group metadata
            segments = [s.strip() for s in group_path.split(' > ') if s.strip()]
            pid = 0
            for depth_seg in range(len(segments)):
                cumpath = ' > '.join(segments[:depth_seg + 1])
                pid = _get_or_create_group_node(cumpath, pid)
            existing_depth = len(segments)

            if len(vi_list) < min_cluster_size:
                for vi in vi_list:
                    _add_child(node_map, pid, var_nodes[vi])
                continue

            # ── Aspect-discovery callback for this group ──
            # Top-level: optionally uses FASTopic (slow, transformer-based,
            # semantic).  Sub-recursion: NMF only (fast, lexical).  This split
            # gives the best of both: rich top-level structure + quick deeper
            # splits.  Critical for HCP performance (~10× speedup vs running
            # FASTopic in every sub-recursion).
            def _aspect_recurse_for_group(vi_sub: list, parent: int,
                                            depth_left: int = 99,
                                            is_top_level: bool = False) -> int:
                local_texts = [texts[vi] for vi in vi_sub]
                use_reprs, use_labels, use_tfidf = None, None, None
                aspect_method_tag = 'sliced_global'

                if (is_top_level and use_fastopic and _FASTOPIC_AVAILABLE
                        and len(vi_sub) >= fastopic_min_size):
                    try:
                        f_tfidf, _, _, _, f_H, _, f_labels = discover_aspects_fastopic(
                            local_texts, max_aspects=max_aspects)
                        f_reprs = per_aspect_representations(
                            local_texts, f_H, f_tfidf, sbert_model)
                        use_reprs, use_labels, use_tfidf = f_reprs, f_labels, f_tfidf
                        aspect_method_tag = 'fastopic'
                    except Exception:
                        use_reprs = None
                if use_reprs is None and local_nmf and len(vi_sub) >= min_local_nmf_size:
                    try:
                        l_tfidf, _, _, _, l_H, _, l_labels = discover_aspects(
                            local_texts, max_aspects)
                        l_reprs = per_aspect_representations(
                            local_texts, l_H, l_tfidf, sbert_model)
                        use_reprs, use_labels, use_tfidf = l_reprs, l_labels, l_tfidf
                        aspect_method_tag = 'nmf'
                    except Exception:
                        use_reprs = None
                if use_reprs is None:
                    use_reprs  = [rep[vi_sub] for rep in aspect_reprs]
                    use_labels, use_tfidf = aspect_labels, tfidf

                before = sum(1 for n in nodes if n.get('type') == 'aggregation')
                effective_depth = max(0, min(depth_left, max_depth))
                _recurse(vi_sub, use_reprs, use_labels, use_tfidf,
                         parent, max(0, max_depth - effective_depth), group_path,
                         aspect_method_tag=aspect_method_tag)
                after = sum(1 for n in nodes if n.get('type') == 'aggregation')
                return max(0, after - before)

            local_texts = [texts[vi] for vi in vi_list]
            _top_grp = group_path.split(' > ')[0].strip()

            role_built = False
            role_regularity = None
            route_label = None

            # ── ROUTING 1A: PER-ROW LLM ROLE EXTRACTION (Zhu et al. EMNLP 2025) ──
            # The cleanest semantic route: one LLM call per row extracts
            # measure / statistic / condition / subtype values directly from
            # the description text.  Strict substring grounding prevents
            # hallucination.  Bypasses SBERT phrase clustering entirely.
            if (use_per_row_role_extraction
                    and per_row_role_extractor_fn is not None
                    and len(vi_list) >= 4):
                try:
                    role_result = discover_roles_via_per_row_extraction(
                        local_texts,
                        text_col_names=text_col_names,
                        per_row_extractor_fn=per_row_role_extractor_fn,
                        group_name=_top_grp,
                        regularity_threshold=role_regularity_threshold,
                    )
                    role_regularity = role_result.get('regularity')
                    # Always capture audit data, even when valid=False, so the
                    # Role Decomposition tab can show what the LLM proposed
                    # and why proposals were rejected.  group_path + vi_list
                    # let the display map each audit row back to its variable.
                    try:
                        st.session_state.a2_per_row_audit.append({
                            'group_name':   _top_grp,
                            'group_path':   group_path,
                            'vi_list':      [int(v) for v in vi_list],
                            'n_rows':       len(vi_list),
                            'coverage':     role_result.get('coverage'),
                            'valid':        role_result.get('valid'),
                            'roles_final':  list(role_result.get('roles', {}).keys()),
                            'summary':      role_result.get('extractor_summary', {}),
                            'per_row_audit': role_result.get('per_row_audit', []),
                        })
                    except Exception:
                        pass
                    if role_result['valid']:
                        role_built = build_role_hierarchy(
                            vi_list, role_result, can,
                            pid, group_path, nodes, node_map, var_nodes,
                            max_depth_remaining=max(2, max_depth - existing_depth),
                            post_split_fn=lambda vi_grp, p, d:
                                _aspect_recurse_for_group(vi_grp, p, d,
                                                            is_top_level=False),
                            min_post_split_size=max(6, min_cluster_size + 4))
                        if role_built:
                            route_label = 'per_row_llm_extraction'
                except Exception:
                    role_built = False

            # ── ROUTING 1B: SBERT phrase clustering fallback (Option D original) ──
            if not role_built and (use_role_decomposition and sbert_model is not None
                    and len(vi_list) >= 6):
                try:
                    role_result = discover_roles_via_sbert_phrase_clustering(
                        local_texts,
                        text_col_names=text_col_names,
                        sbert_model=sbert_model,
                        llm_role_namer_fn=llm_role_namer_fn,
                        group_name=_top_grp,
                        regularity_threshold=role_regularity_threshold,
                    )
                    role_regularity = role_result.get('regularity')
                    if role_result['valid']:
                        role_built = build_role_hierarchy(
                            vi_list, role_result, can,
                            pid, group_path, nodes, node_map, var_nodes,
                            max_depth_remaining=max(2, max_depth - existing_depth),
                            post_split_fn=lambda vi_grp, p, d:
                                _aspect_recurse_for_group(vi_grp, p, d,
                                                            is_top_level=False),
                            min_post_split_size=max(6, min_cluster_size + 4))
                        if role_built:
                            route_label = 'sbert_phrase_clustering'
                except Exception:
                    role_built = False
            if role_built:
                anchor = node_map.get(pid)
                if anchor is not None:
                    anchor.setdefault('structure_provenance', {})
                    anchor['structure_provenance']['phrase_regularity'] = role_regularity
                    anchor['structure_provenance']['route_used'] = route_label
                continue

            # ── ROUTING 2: phrase-slot mining (IE / mutual-exclusion) ──
            slot_built = False
            if use_slot_mining and len(vi_list) >= 6:
                try:
                    slot_result = mine_phrase_slots(
                        local_texts,
                        text_col_names=text_col_names,
                        group_name=_top_grp,
                        llm_role_classifier_fn=llm_role_classifier_fn,
                    )
                    if slot_result['valid']:
                        slot_built = build_slot_hierarchy(
                            vi_list, slot_result, can,
                            pid, group_path, nodes, node_map, var_nodes,
                            max_depth_remaining=max(2, max_depth - existing_depth),
                            post_slot_split_fn=lambda vi_grp, p, d:
                                _aspect_recurse_for_group(vi_grp, p, d,
                                                            is_top_level=False),
                            min_post_slot_size=max(6, min_cluster_size + 4))
                except Exception:
                    slot_built = False
            if slot_built:
                anchor = node_map.get(pid)
                if anchor is not None:
                    anchor.setdefault('structure_provenance', {})
                    anchor['structure_provenance']['phrase_regularity'] = role_regularity
                    anchor['structure_provenance']['route_used'] = 'slot_mining_fallback'
                continue

            # ── ROUTING 3: aspect discovery (FASTopic / NMF) ──
            anchor = node_map.get(pid)
            if anchor is not None:
                anchor.setdefault('structure_provenance', {})
                anchor['structure_provenance']['phrase_regularity'] = role_regularity
                anchor['structure_provenance']['route_used'] = 'aspect_clustering_fallback'
            _aspect_recurse_for_group(vi_list, pid, is_top_level=True)

    # Clear the progress bar
    try:
        if _show_progress and _pbar is not None:
            _pbar.empty()
    except Exception:
        pass

    # ── POST-BUILD PASS 1 — sibling common-prefix factoring [Castanet 2007] ──
    try:
        n_factored = factor_sibling_common_prefixes(
            nodes, min_siblings=3, min_prefix_tokens=2)
    except Exception:
        n_factored = 0

    # ── POST-BUILD PASS 2 — conservative noise-label pruning ──────────────────
    try:
        n_dissolved = prune_low_quality_aggregations(nodes)
    except Exception:
        n_dissolved = 0

    # ── POST-BUILD PASS 3 — strip group prefix from child labels ──────────────
    # 'DMS — Total Errors' under DMS → 'Total Errors'.  Reduces visual
    # redundancy without altering tree structure.
    try:
        n_stripped = strip_group_prefix_from_labels(nodes)
    except Exception:
        n_stripped = 0

    # ── POST-BUILD PASS 4 — enforce single parent (collapse DAG → tree) ───────
    # Keeps each variable under its most specific parent so the hierarchy is a
    # true tree, matching the gold standard and rendering proportionally.
    try:
        n_reparented = enforce_single_parent(nodes)
    except Exception:
        n_reparented = 0

    # Annotate the root with post-build statistics
    if nodes and nodes[0].get('type') == 'root':
        nodes[0]['post_build_stats'] = {
            'sibling_factor_nodes_inserted': int(n_factored),
            'low_quality_nodes_dissolved':   int(n_dissolved),
            'group_prefix_labels_stripped':  int(n_stripped),
            'dag_links_removed':             int(n_reparented),
        }

    # Deduplicate children
    for nd in nodes:
        nd['related'] = list(dict.fromkeys(int(x) for x in nd['related']))

    return nodes

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9  — EVALUATION  [ZHU Table 2 / TraCo / TICL §3.4]
# ──────────────────────────────────────────────────────────────────────────────
def purity_score(true_labels, pred_labels) -> float:
    true = np.array(true_labels); pred = np.array(pred_labels)
    total = len(true)
    if total == 0:
        return 0.0
    score = 0
    for cl in np.unique(pred):
        mask = pred == cl
        if not mask.any():
            continue
        counts = np.bincount(true[mask].astype(int))
        score += counts.max()
    return score / total

def evaluate(true_labels, pred_labels) -> dict:
    le = LabelEncoder()
    tl = le.fit_transform([str(x) for x in true_labels])
    pl = np.array(pred_labels, dtype=int)
    return {
        'NMI':    round(float(normalized_mutual_info_score(tl, pl)), 4),
        'ARI':    round(float(adjusted_rand_score(tl, pl)),          4),
        'Purity': round(purity_score(tl, pl),                        4),
    }

def hierarchy_quality_metrics(nodes: list, tfidf: TfidfVectorizer) -> dict:
    """
    Compute lightweight structural quality metrics inspired by TraCo (AAAI 2024).

    [TraCo] Wu et al. identify three failure modes in hierarchical topic models:
    low affinity (children unrelated to parent), low rationality (children not
    more specific than parent), and low diversity (sibling topics too similar).

    Here we measure:
    • parent-child coherence: mean cosine(parent_label, child_label) — proxy for affinity.
    • sibling diversity: mean (1 - pairwise cosine) among siblings — proxy for diversity.
    """
    node_map = {n['id']: n for n in nodes}
    coherence_scores, diversity_scores = [], []

    for n in nodes:
        if n['type'] != 'aggregation' or not n['related']:
            continue
        children     = [node_map[c] for c in n['related'] if c in node_map]
        child_labels = [c['name'] for c in children
                        if c.get('type') == 'aggregation' and c['name'] != 'Group']
        if len(child_labels) < 2:
            continue

        try:
            parent_vec  = tfidf.transform([n['name']])
            child_vecs  = tfidf.transform(child_labels)
            # Parent-child coherence
            coh = float(cosine_similarity(parent_vec, child_vecs).mean())
            coherence_scores.append(coh)
            # Sibling diversity
            sib_sims = cosine_similarity(child_vecs)
            np.fill_diagonal(sib_sims, 0)
            diversity_scores.append(float(1 - sib_sims.mean()))
        except Exception:
            pass

    return {
        'mean_parent_child_coherence': round(np.mean(coherence_scores), 4) if coherence_scores else 0.0,
        'mean_sibling_diversity':      round(np.mean(diversity_scores),  4) if diversity_scores else 0.0,
        'n_aggregation_nodes':         len([n for n in nodes if n['type'] == 'aggregation']),
        'n_singleton_splits':          len([n for n in nodes
                                           if n['type'] == 'aggregation'
                                           and len(n['related']) == 1]),
    }

# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY-TIME ONE-CHILD CHAIN COMPRESSION  (visualization only — structure preserved in JSON)
# ──────────────────────────────────────────────────────────────────────────────
def compress_one_child_chains(nodes: list) -> list:
    """
    Display-only transformation: collapse chains where an aggregation node has
    exactly one aggregation child (e.g. 'DMS → DMS Recommended Standard').
    The collapsed node displays as 'DMS / DMS Recommended Standard'.
    Structural data in the exported JSON is NOT modified — this returns a new
    node list used only for visualisation.
    """
    nodes = _filter_dissolved(nodes)
    nm = {int(n['id']): dict(n) for n in nodes}   # deep-ish copy
    parent_of: dict = {}
    for n in nodes:
        for c in n.get('related', []):
            parent_of.setdefault(int(c), int(n['id']))

    def _is_chain_link(n):
        if n.get('type') != 'aggregation':
            return False
        children = n.get('related', [])
        return (len(children) == 1
                and nm.get(int(children[0]), {}).get('type') == 'aggregation')

    changed = True
    while changed:
        changed = False
        for nid, n in list(nm.items()):
            if _is_chain_link(n):
                child_id = int(n['related'][0])
                child    = nm[child_id]
                # Merge: child takes parent's id, parent's slot, but combined name
                merged_name = f"{n['name']} / {child['name']}"
                new_node = dict(child)
                new_node['id']   = nid
                new_node['name'] = merged_name
                new_node['desc'] = f"{n.get('desc','')} | {child.get('desc','')}"
                # Rewire child's children to nid (already nid)
                nm[nid] = new_node
                # Remove the original child node
                if child_id in nm:
                    del nm[child_id]
                # Re-point any references to child_id → nid
                for other in nm.values():
                    other['related'] = [nid if int(c) == child_id else int(c)
                                        for c in other.get('related', [])]
                changed = True
                break

    return list(nm.values())

# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _filter_dissolved(nodes: list) -> list:
    """
    Drop dissolved/hidden nodes for visualisation.  Children of dissolved
    nodes were already promoted to the grandparent by the pruning pass, so
    dropping the dissolved wrapper here makes the tree render cleanly.
    """
    drop_ids = {int(n['id']) for n in nodes
                 if n.get('type') == 'dissolved' or n.get('isShown') is False}
    if not drop_ids:
        return nodes
    out = []
    for n in nodes:
        if int(n['id']) in drop_ids:
            continue
        m = dict(n)
        m['related'] = [int(c) for c in n.get('related', [])
                         if int(c) not in drop_ids]
        out.append(m)
    return out

def _leaf_ids(nodes: list, nid: int) -> list:
    m = {int(n['id']): n for n in nodes}
    out = []
    def rec(x):
        n = m.get(int(x))
        if not n: return
        if n.get('type') == 'attribute': out.append(int(x)); return
        for c in n.get('related', []): rec(int(c))
    rec(nid)
    return list(dict.fromkeys(out))

def _parent_map(nodes: list) -> dict:
    pm = {}
    for n in nodes:
        for c in n.get('related', []):
            if int(c) not in pm:
                pm[int(c)] = int(n['id'])
    return pm

def _tree_value_map(nodes: list, pm: dict) -> dict:
    """
    Leaf count per node measured along the *rendered* tree (each node has
    exactly one parent, per `pm`).  Plotly draws sectors using that same
    single-parent structure, so values built this way always satisfy
    parent == sum(children) — which is what branchvalues='total' requires.

    The full hierarchy can be a DAG (a variable promoted under more than one
    role branch), in which case `_leaf_ids` double-counts a shared leaf and a
    parent's unique-leaf count comes out *less* than the sum of its children's
    counts.  Feeding those numbers to a 'total' chart blanks it.  Counting on
    the rendered tree instead avoids that without changing the hierarchy.
    """
    kids = {}
    for child, par in pm.items():
        kids.setdefault(int(par), []).append(int(child))
    nodemap = {int(n['id']): n for n in nodes}
    memo = {}
    def count(nid: int) -> int:
        if nid in memo:
            return memo[nid]
        memo[nid] = 1  # guard against cycles while recursing
        n = nodemap.get(nid)
        if n is not None and n.get('type') == 'attribute':
            memo[nid] = 1
            return 1
        ch = kids.get(nid, [])
        v = sum(count(c) for c in ch) if ch else 1
        memo[nid] = max(1, v)
        return memo[nid]
    return {nid: count(nid) for nid in nodemap}

def _wrap_hover(text: str, width: int = 80) -> str:
    """Soft-wrap long descriptions onto multiple <br>-separated lines so the
    Plotly hover tooltip shows the full text instead of being cut off."""
    import textwrap as _tw
    s = str(text or '')
    if not s:
        return ''
    lines = []
    for raw_line in s.split('\n'):
        lines.extend(_tw.wrap(raw_line, width=width) or [''])
    return '<br>'.join(lines)

def plot_sunburst(nodes: list, max_depth: int = 4):
    nodes = _filter_dissolved(nodes)
    pm = _parent_map(nodes)
    vm = _tree_value_map(nodes, pm)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id'])
        lc  = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get('name', ''))[:40])
        parents.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(vm.get(nid, 1))
        hover.append(f"<b>{n.get('name','')}</b><br>Type: {n.get('type','')}<br>"
                     f"Variables: {lc}<br><br>{_wrap_hover(n.get('desc',''))}")
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues='total', hovertext=hover, hoverinfo='text',
        maxdepth=max_depth, insidetextorientation='radial',
        marker=dict(colorscale='Viridis', line=dict(width=1, color='white'))))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text='Click sector to drill down — click centre to go back',
                                 font=dict(size=13), x=0.5))
    return fig

def plot_treemap(nodes: list):
    nodes = _filter_dissolved(nodes)
    pm = _parent_map(nodes)
    vm = _tree_value_map(nodes, pm)
    ids, labels, parents, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id'])
        lc  = len(_leaf_ids(nodes, nid))
        ids.append(str(nid))
        labels.append(str(n.get('name', ''))[:40])
        parents.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(vm.get(nid, 1))
        hover.append(f"<b>{n.get('name','')}</b><br>Variables: {lc}<br>"
                     f"{_wrap_hover(n.get('desc',''))}")
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues='total', hovertext=hover, hoverinfo='text',
        textinfo='label+value',
        marker=dict(colorscale='Viridis', line=dict(width=1, color='white'))))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# NODE-LINK TREE  — Reingold-Tilford layout (matches Approach 1.1 interface)
# ──────────────────────────────────────────────────────────────────────────────
def _a2_node_color(n: dict) -> str:
    t = n.get('type', '')
    if t == 'root':      return '#c44e52'
    if t == 'attribute': return '#4C72B0'
    if t == 'collapsed': return '#bbbbbb'
    return '#8C8C8C'

def _display_graph(nodes: list, max_depth: int = 4, show_hidden: bool = False):
    """Walk tree to chosen depth, inserting 'collapsed' placeholders for cut-off branches."""
    m = {int(n['id']): n for n in nodes}
    dnodes: dict = {}
    edges: list  = []
    counter = 10 ** 9

    def rec(nid, depth):
        nonlocal counter
        n = m.get(int(nid))
        if not n:
            return
        if not show_hidden and n.get('isShown') is False and depth > 0:
            return
        dnodes[int(nid)] = n
        if depth >= max_depth and n.get('related'):
            counter += 1
            cid = counter
            n_leaves = len(_leaf_ids(nodes, nid))
            dnodes[cid] = {'id': cid,
                           'name': f'… {n_leaves} variables',
                           'type': 'collapsed', 'dtype': 'determine',
                           'related': [], 'desc': f"Collapsed: {n.get('name')}",
                           'isShown': True}
            edges.append((int(nid), cid))
            return
        for c in n.get('related', []):
            ch = m.get(int(c))
            if not ch:
                continue
            if not show_hidden and ch.get('isShown') is False:
                continue
            edges.append((int(nid), int(c)))
            rec(int(c), depth + 1)

    rec(0, 0)
    return list(dnodes.values()), edges

def _positions(dnodes: list, edges: list):
    """Reingold-Tilford style positions: x=depth, y=subtree-aware vertical."""
    H_SCALE = 3.0
    V_SPACE = 1.8
    children: dict = defaultdict(list)
    for p, c in edges:
        children[p].append(c)
    pos: dict = {}
    counter = {'v': 0}

    def rec(nid, depth):
        ch = children.get(nid, [])
        if not ch:
            y_pos = counter['v'] * V_SPACE
            counter['v'] += 1
            pos[nid] = (depth * H_SCALE, y_pos)
            return y_pos
        child_ys = [rec(c, depth + 1) for c in ch]
        y_pos = float(np.mean(child_ys))
        pos[nid] = (depth * H_SCALE, y_pos)
        return y_pos

    rec(0, 0)
    return pos

def plot_node_link(nodes: list, max_depth: int = 4,
                    show_hidden: bool = False, show_leaf_labels: bool = False):
    """
    Node-link tree with elbow edges (matches Approach 1.1 layout).
    Best for exploring structure at moderate depth — Sunburst remains
    recommended for large hierarchies per Taxonomizer (Bian et al. 2020).
    """
    nodes = _filter_dissolved(nodes)
    dnodes, edges = _display_graph(nodes, max_depth, show_hidden)
    pos = _positions(dnodes, edges)

    # Elbow edges
    ex, ey = [], []
    for p, c in edges:
        if p not in pos or c not in pos:
            continue
        x0, y0 = pos[p]
        x1, y1 = pos[c]
        xm = (x0 + x1) / 2
        ex += [x0, xm, xm, x1, None]
        ey += [y0, y0, y1, y1, None]
    traces = [go.Scatter(x=ex, y=ey, mode='lines',
                          line=dict(width=1, color='#c8c8c8'),
                          hoverinfo='skip', showlegend=False)]

    agg_xs, agg_ys, agg_labels, agg_colors, agg_hover = [], [], [], [], []
    lf_xs,  lf_ys,  lf_labels,  lf_colors,  lf_hover  = [], [], [], [], []

    for n in dnodes:
        nid = int(n['id'])
        if nid not in pos:
            continue
        x, y = pos[nid]
        lc   = len(_leaf_ids(nodes, nid))
        lab  = n.get('name', str(nid))
        htxt = (f"<b>{n.get('name','')}</b><br>"
                f"Type: {n.get('type','')}<br>"
                f"Variables: {lc}<br><br>{_wrap_hover(n.get('desc',''))}")
        col  = _a2_node_color(n)

        if n.get('type') in ('root', 'aggregation', 'collapsed'):
            display_lab = (lab + (f' ({lc})' if lc else ''))[:50]
            agg_xs.append(x); agg_ys.append(y)
            agg_labels.append(display_lab); agg_colors.append(col); agg_hover.append(htxt)
        else:
            display_lab = lab[:40] if show_leaf_labels else ''
            lf_xs.append(x); lf_ys.append(y)
            lf_labels.append(display_lab); lf_colors.append(col); lf_hover.append(htxt)

    if agg_xs:
        traces.append(go.Scatter(
            x=agg_xs, y=agg_ys, mode='markers+text',
            text=agg_labels, textposition='middle right',
            hovertext=agg_hover, hoverinfo='text',
            marker=dict(size=16, color=agg_colors,
                        line=dict(color='white', width=2)),
            showlegend=False))
    if lf_xs:
        traces.append(go.Scatter(
            x=lf_xs, y=lf_ys, mode='markers+text',
            text=lf_labels, textposition='middle right',
            hovertext=lf_hover, hoverinfo='text',
            marker=dict(size=7, color=lf_colors, symbol='circle',
                        opacity=0.75, line=dict(color='white', width=1)),
            showlegend=False))

    n_leaves = max(12, len(lf_xs))
    fig = go.Figure(traces)
    fig.update_layout(
        height=max(700, min(4000, int(n_leaves * 32))),
        margin=dict(l=20, r=220, t=40, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(visible=False, fixedrange=False),
        yaxis=dict(visible=False, autorange='reversed', fixedrange=False),
        dragmode='pan',
        annotations=[dict(
            text='Tip: Sunburst is better for large hierarchies [Taxonomizer 2020]',
            xref='paper', yref='paper', x=0.0, y=1.01,
            showarrow=False, font=dict(size=11, color='grey'), align='left')]
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────
# set_page_config handled by the navigation router (demo.py)
st.title('Approach 2 — Role-Decomposed Metadata Hierarchy')
st.caption('Group anchoring → LLM role extraction → role-nested LoD tree. '
           'Full method details and citations in the Method tab.')

# Session-state init — defensive: always ensure these keys exist with safe
# defaults.  Streamlit can occasionally drop attribute-style access if the key
# was set via [setter] in a previous run; using setdefault here works on both
# new and existing sessions.
st.session_state.setdefault('a2_nodes', None)
st.session_state.setdefault('a2_can',   None)
st.session_state.setdefault('a2_meta',  {})
st.session_state.setdefault('a2_per_row_audit', [])   # list of group audits

# Local-LLM auto-detection: probe Ollama server.  LLM refinement defaults to
# ON when Ollama is reachable; user can switch it off any time.  Env vars
# OLLAMA_URL and OLLAMA_MODEL override the localhost / qwen2.5:3b defaults.
_ollama_url     = os.environ.get('OLLAMA_URL', OLLAMA_URL_DEFAULT).strip() or OLLAMA_URL_DEFAULT
_ollama_model   = os.environ.get('OLLAMA_MODEL', OLLAMA_MODEL_DEFAULT).strip() or OLLAMA_MODEL_DEFAULT
_ollama_reachable = _ping_ollama(_ollama_url)

# Groq detection: env-var GROQ_API_KEY makes the cloud option available.
_groq_url       = os.environ.get('GROQ_URL', GROQ_URL_DEFAULT).strip() or GROQ_URL_DEFAULT
_groq_model     = os.environ.get('GROQ_MODEL', GROQ_MODEL_DEFAULT).strip() or GROQ_MODEL_DEFAULT
_groq_key_env   = os.environ.get('GROQ_API_KEY', '').strip()

_default_provider = 'groq' if (_groq_key_env and _LLM_CLIENT_AVAILABLE) else 'ollama'
_default_llm_on   = (
    (_ollama_reachable or bool(_groq_key_env)) and _LLM_CLIENT_AVAILABLE)

with st.sidebar:
    st.header('1 · Input')
    uploads = st.file_uploader('Metadata / data-dictionary file(s)',
                                type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'json'],
                                accept_multiple_files=True)

    st.header('2 · Algorithm')
    max_aspects     = st.slider('Max aspects (K upper bound)', 3, 15, 6)
    max_depth       = st.slider('Max tree depth', 2, 10, 6)
    min_cluster_sz  = st.slider('Min variables per cluster', 1, 10, 2)
    sil_thresh      = st.slider('Silhouette threshold',
                                0.01, 0.30,
                                value=0.04 if not _SBERT_AVAILABLE else 0.05,
                                step=0.01)
    max_k_split     = st.slider('Max child clusters per split', 2, 12, 5)
    use_sbert       = st.checkbox('SBERT embeddings',
                                  value=_SBERT_AVAILABLE,
                                  disabled=not _SBERT_AVAILABLE)
    local_nmf       = st.checkbox('Local NMF fallback', value=True)
    min_local_nmf   = st.slider('Min group size for NMF', 5, 30, 6)
    use_slot_mining = st.checkbox('Phrase-slot mining', value=True)
    use_fastopic   = st.checkbox(
        'FASTopic aspects',
        value=_FASTOPIC_AVAILABLE,
        disabled=not _FASTOPIC_AVAILABLE,
    )
    fastopic_min_sz = st.slider('Min group size for FASTopic', 6, 100, 40)
    if not _FASTOPIC_AVAILABLE:
        st.warning('FASTopic not installed — using NMF.')

    st.header('3 · LLM provider')
    provider_options = ['Ollama (local)', 'Groq (cloud)']
    provider_default_idx = 1 if _default_provider == 'groq' else 0
    provider_label = st.radio('Provider', provider_options,
                              index=provider_default_idx, horizontal=True)
    llm_provider = 'groq' if provider_label.startswith('Groq') else 'ollama'

    if llm_provider == 'groq':
        if not _LLM_CLIENT_AVAILABLE:
            st.warning('`openai` package not installed.')
        elif _groq_key_env:
            st.success('GROQ_API_KEY detected → ready')
        groq_key_in = st.text_input(
            'Groq API key', value=_groq_key_env, type='password',
            help='Free key at console.groq.com/keys, or set GROQ_API_KEY env var.')
        groq_model_in = st.text_input('Groq model', value=_groq_model)
        llm_base_url = _groq_url
        llm_model    = groq_model_in
        llm_api_key  = groq_key_in
        llm_ready    = bool(groq_key_in and _LLM_CLIENT_AVAILABLE)
    else:
        if not _LLM_CLIENT_AVAILABLE:
            st.warning('`openai` package not installed.')
        elif _ollama_reachable:
            st.success(f'Ollama reachable at `{_ollama_url}`')
        else:
            st.warning(f'Ollama not reachable at `{_ollama_url}`.')
        ollama_url_in   = st.text_input('Ollama URL', value=_ollama_url)
        ollama_model_in = st.text_input('Ollama model', value=_ollama_model)
        llm_base_url = ollama_url_in
        llm_model    = ollama_model_in
        llm_api_key  = ''
        llm_ready    = bool(_LLM_CLIENT_AVAILABLE and _ollama_reachable)

    st.header('4 · LLM features')
    use_per_row_role_extraction = st.checkbox(
        'Per-row role extraction (primary route)',
        value=llm_ready,
        disabled=not llm_ready,
        help='One LLM call per variable extracts measure / statistic / '
             'condition / subtype, grounded to the description text.'
    )
    use_llm = st.checkbox(
        'LLM label refinement',
        value=llm_ready,
        disabled=not llm_ready,
    )
    use_llm_roles = st.checkbox(
        'LLM phrase-role classifier (fallback)',
        value=llm_ready,
        disabled=not llm_ready,
    )
    use_role_decomposition = st.checkbox(
        'SBERT phrase clustering (fallback)',
        value=_SBERT_AVAILABLE and llm_ready,
        disabled=not (_SBERT_AVAILABLE and llm_ready),
    )
    role_namer_constrained = st.checkbox(
        'Constrained role vocabulary',
        value=True,
    )
    role_regularity_threshold = st.slider(
        'Min phrase regularity', 0.05, 0.80, 0.20, 0.05,
    )

    st.header('5 · Project')
    project_name = st.text_input('Project name', value='project')

# ── load and configure files ──────────────────────────────────────────────────
if uploads:
    import tempfile
    tmp    = Path(tempfile.mkdtemp())
    raw_by = {}
    cfg_by = {}

    st.subheader('Step 1 — Inspect metadata')
    for f in uploads:
        p = tmp / safe_name(f.name)
        p.write_bytes(f.getbuffer())
        try:
            df = load_any(p)
            raw_by[f.name] = df
            cfg_by[f.name] = detect_roles(df)
            with st.expander(f'{f.name}', expanded=False):
                st.write(f'Rows: **{len(df):,}**  Columns: **{len(df.columns)}**')
                st.dataframe(df.head(8), width='stretch')
        except Exception as e:
            st.error(f'Could not load {f.name}: {e}')

    st.subheader('Step 2 — Confirm column roles')
    configs = {}
    for name, df in raw_by.items():
        cols = list(df.columns)
        auto = cfg_by[name]
        with st.expander(f'{name}', expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                leaf  = st.multiselect('Leaf variable column(s)', cols,
                                       default=[c for c in auto['leaf_cols']  if c in cols],
                                       key=f'lf_{name}')
                group = st.multiselect('Group / task column(s)', cols,
                                       default=[c for c in auto['group_cols'] if c in cols],
                                       key=f'gr_{name}')
            with c2:
                text  = st.multiselect('Description column(s)', cols,
                                       default=[c for c in auto['text_cols']  if c in cols],
                                       key=f'tx_{name}')
                meta  = st.multiselect('Type / unit column(s)', cols,
                                       default=[c for c in auto['meta_cols']  if c in cols],
                                       key=f'mt_{name}')
            configs[name] = {'leaf_cols': leaf, 'group_cols': group,
                             'text_cols': text,  'meta_cols':  meta}

    if st.button('Build Approach 2 Hierarchy', type='primary'):
        try:
            # Clear stale audit data from any previous build
            st.session_state.a2_per_row_audit = []

            cans  = [build_canonical(df, configs[name], name)
                     for name, df in raw_by.items()]
            can   = pd.concat(cans, ignore_index=True)
            st.session_state.a2_can = can
            texts = can['_text'].fillna('').astype(str).tolist()

            # Global NMF (used as fallback and for facet trees)
            with st.spinner('Step 3 — Discovering global aspects via NMF …'):
                tfidf, X, nmf, W, H, K, alabels = discover_aspects(texts, max_aspects)
                st.session_state.a2_meta = {
                    'K': K, 'aspect_labels': alabels, 'tfidf': tfidf, 'W': W, 'H': H
                }
                st.info(f'Global aspects discovered: **{K}** — {" | ".join(alabels)}')

            sbert_model = None
            if use_sbert and _SBERT_AVAILABLE:
                with st.spinner('Loading SBERT model …'):
                    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

            with st.spinner('Step 4 — Building global per-aspect representations …'):
                reprs = per_aspect_representations(texts, H, tfidf, sbert_model)

            # Build LLM function trio — same provider + endpoint for all.
            llm_fn = None
            if use_llm:
                llm_fn = make_llm_label_fn(llm_base_url, llm_model,
                                             provider=llm_provider,
                                             api_key=llm_api_key)
                if llm_fn is None:
                    st.warning('LLM unreachable — deterministic labels only.')

            llm_role_namer = None
            if use_role_decomposition:
                llm_role_namer = make_llm_role_namer_fn(
                    llm_base_url, llm_model,
                    constrained=role_namer_constrained,
                    provider=llm_provider, api_key=llm_api_key)

            llm_role_fn = None
            if use_llm_roles:
                llm_role_fn = make_llm_role_classifier_fn(
                    llm_base_url, llm_model,
                    provider=llm_provider, api_key=llm_api_key)

            # NEW: per-row role extractor (Zhu et al. EMNLP 2025) — primary route
            per_row_extractor = None
            if use_per_row_role_extraction:
                per_row_extractor = make_per_row_role_extractor_fn(
                    llm_base_url, llm_model,
                    provider=llm_provider, api_key=llm_api_key)
                if per_row_extractor:
                    st.info(f'Per-row role extraction: **{llm_provider}** · `{llm_model}`')
                else:
                    st.warning('LLM unreachable for per-row role extraction.')

            # Collect detected text-column names across all uploaded configs —
            # used by phrase-slot mining to identify description-like fields.
            all_text_cols: list = []
            for cfg in configs.values():
                for c in cfg.get('text_cols', []):
                    if c not in all_text_cols:
                        all_text_cols.append(c)

            # Replace the module-level FIELD_NAME_NOISE with a noise set DERIVED
            # from the actual detected column names — zero hardcoding.  All
            # downstream calls (label_cluster, _bigram_preferred_terms) read
            # the module-level name so this swap propagates everywhere.
            import sys as _sys
            _sys.modules[__name__].FIELD_NAME_NOISE = build_field_noise(configs)

            with st.spinner('Step 6 — Building group-anchored LoD tree '
                            '(slot mining → FASTopic → NMF) …'):
                nodes = build_dynamic_lod_tree(
                    can, reprs, alabels, tfidf,
                    max_depth=max_depth,
                    min_cluster_size=min_cluster_sz,
                    sil_threshold=sil_thresh,
                    max_clusters_per_split=max_k_split,
                    project=project_name,
                    local_nmf=local_nmf,
                    min_local_nmf_size=min_local_nmf,
                    max_aspects=max_aspects,
                    sbert_model=sbert_model,
                    llm_label_fn=llm_fn,
                    use_slot_mining=use_slot_mining,
                    text_col_names=all_text_cols,
                    use_fastopic=use_fastopic,
                    fastopic_min_size=fastopic_min_sz,
                    llm_role_classifier_fn=llm_role_fn,
                    use_role_decomposition=use_role_decomposition,
                    llm_role_namer_fn=llm_role_namer,
                    role_regularity_threshold=role_regularity_threshold,
                    per_row_role_extractor_fn=per_row_extractor,
                    use_per_row_role_extraction=use_per_row_role_extraction,
                )
                st.session_state.a2_nodes = nodes


            # Concise build summary — per-route node counts
            route_counts: dict = Counter()
            for n in nodes:
                if n.get('type') == 'aggregation':
                    route_counts[
                        n.get('structure_provenance', {}).get('route', '—')] += 1
            n_leaves   = len([n for n in nodes if n.get('type') == 'attribute'])
            n_internal = len([n for n in nodes if n.get('type') == 'aggregation'])
            route_str  = ' · '.join(f'{r}: {c}' for r, c in route_counts.most_common())
            st.success(f'Done — {n_leaves} variables · {n_internal} internal nodes '
                       f'({route_str})')

            # If any LLM call hit a rate-limit (429), the model ran out of
            # tokens — tell the user to switch model in the sidebar and rebuild.
            ran_out = any(
                ('RateLimit' in str(r) or '429' in str(r))
                for a in (st.session_state.get('a2_per_row_audit') or [])
                for r in (a.get('summary') or {})
            )
            if ran_out:
                st.error(f'Ran out of tokens on `{llm_model}`. '
                         f'Switch to another Groq model in the sidebar '
                         f'(e.g. llama-3.1-8b-instant) and rebuild.')
        except Exception as e:
            st.error(f'Build failed: {e}')
            import traceback; st.code(traceback.format_exc())

# ── display ───────────────────────────────────────────────────────────────────
# Robust session-state reads — use .get() so a partial/incomplete build that
# wrote some keys but not others doesn't crash the display layer.
if st.session_state.get('a2_nodes') is None:
    st.info('Upload a metadata file and click **Build Approach 2 Hierarchy** to start.')
    st.stop()

nodes  = st.session_state.get('a2_nodes')
can    = st.session_state.get('a2_can')
meta   = st.session_state.get('a2_meta') or {}

tabs = st.tabs(['LoD Tree', 'Evaluation', 'Role Decomposition',
                'Label Provenance', 'Metadata', 'Export', 'Method'])

with tabs[0]:
    # ── Visualization controls (above chart — easy to find, matches Approach 1.1) ─
    vc1, vc2, vc3, vc4, vc5 = st.columns([2, 2, 1, 1, 1])
    with vc1:
        viz_mode = st.radio(
            'View mode',
            ['Sunburst (drill-down)', 'Treemap', 'Node-link tree'],
            horizontal=True, index=0,
            help='Sunburst best for large hierarchies [Taxonomizer]. '
                 'Node-link best for moderate depth structure inspection.'
        )
    with vc2:
        depth_display = st.slider('Depth (Level of Detail)', 1, 8, 4, 1)
    with vc3:
        show_leaf_labels = st.checkbox('Leaf labels', value=False)
    with vc4:
        show_hidden = st.checkbox('Hidden nodes', value=False)
    with vc5:
        compress_chains = st.checkbox('Compress chains', value=True,
                                       help='Merge one-child aggregation chains '
                                            '(e.g. "DMS → DMS Recommended Standard") '
                                            'for display. Export JSON keeps original structure.')
    st.divider()

    display_nodes = compress_one_child_chains(nodes) if compress_chains else nodes

    if viz_mode == 'Sunburst (drill-down)':
        st.plotly_chart(plot_sunburst(display_nodes, max_depth=depth_display),
                        width='stretch')
    elif viz_mode == 'Treemap':
        st.plotly_chart(plot_treemap(display_nodes), width='stretch')
    else:
        st.plotly_chart(plot_node_link(display_nodes, depth_display,
                                        show_hidden, show_leaf_labels),
                        width='stretch')

    n_l = len([n for n in nodes if n.get('type') == 'attribute'])
    n_i = len([n for n in nodes if n.get('type') == 'aggregation'])
    # max depth
    pm  = _parent_map(nodes)
    def _node_depth(nid):
        d = 0; cur = nid
        while cur in pm:
            cur = pm[cur]; d += 1
        return d
    max_d = max((_node_depth(n['id']) for n in nodes), default=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Variables', n_l)
    c2.metric('Internal nodes', n_i)
    c3.metric('Global aspects', meta.get('K', '?'))
    c4.metric('Max depth', max_d)

with tabs[1]:
    import hierarchy_eval as he

    st.markdown('### Evaluation')
    if can is None or meta.get('W') is None:
        st.info('Run the builder first.')
    else:
        st.caption(
            'The group column is a *construction input* (group-anchored L1/L2, BISE 2026), '
            'so it cannot be ground truth. The primary metrics below are **reference-free** '
            '— they assess the hierarchy itself, no gold standard. Identical definitions to '
            'the Baseline and Approach 1 apps, so the three are directly comparable.'
        )

        # ── PRIMARY: reference-free hierarchy quality (compute on demand) ──────
        # These use SBERT, which is slow to load. Computing them only on a button
        # click keeps the tree, sliders and Save button instant.
        st.markdown('#### Primary — reference-free hierarchy quality')
        if st.button('▶Compute reference-free metrics', key='a2_eval_btn'):
            with st.spinner('Computing reference-free metrics (loads SBERT once)…'):
                tm   = he.traco_metrics(nodes)
                npmi = he.npmi_coherence(nodes, can['_text'].tolist())
            st.session_state['a2_eval_cache'] = {'tm': tm, 'npmi': npmi}

        _ev = st.session_state.get('a2_eval_cache')
        if _ev:
            tm, npmi = _ev['tm'], _ev['npmi']
            p1, p2, p3 = st.columns(3)
            p1.metric('Parent–child coherence', tm['pc_coherence'],
                      help='TraCo (Wu et al., AAAI 2024). Children nest under parent theme.')
            p2.metric('Sibling diversity', tm['sibling_diversity'],
                      help='TraCo (Wu et al., AAAI 2024). Higher = distinct siblings; LOW = redundant.')
            p3.metric('NPMI label coherence', npmi,
                      help='Lau et al., EACL 2014. Label terms genuinely co-occur in the data.')
            st.caption(f'Embedding backend: **{tm["encoder"]}**.')
        else:
            st.info('Click the button above to compute coherence / diversity / NPMI '
                    '(takes a few seconds the first time while SBERT loads).')

        # ── Label-quality proxies (interpretability) ──────────────────────────
        st.markdown('#### Label quality *(interpretability — reference-free)*')
        lq = he.label_quality(nodes)
        l1, l2, l3 = st.columns(3)
        l1.metric('Concept-valid labels', f"{lq['concept_label_pct']}%",
                  help='% of internal labels that read as a real concept (short noun '
                       'phrase, WordNet head) rather than a "/"-joined term fragment.')
        l2.metric('Sibling label redundancy', f"{lq['redundancy_pct']}%",
                  help='% of internal labels duplicating a sibling label (lower is better).')
        l3.metric('Avg label words', lq['avg_label_words'],
                  help='Mean label length in words.')

        # ── Structural statistics ─────────────────────────────────────────────
        st.markdown('#### Structural statistics')
        sm = he.structural_stats(nodes)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric('Aggregation nodes', sm['n_aggregation_nodes'])
        s2.metric('Max leaf depth',    sm['max_depth'])
        s3.metric('Avg leaf depth',    sm['avg_leaf_depth'])
        s4.metric('Avg branching',     sm['avg_branching_factor'])
        s5.metric('Singleton nodes',   f"{sm['singleton_nodes_%']}%")

        # ── Group-structure self-consistency (descriptive, NOT accuracy) ───────
        st.markdown('#### Group-structure self-consistency *(descriptive — not accuracy)*')
        st.caption(
            'The group column is a **construction input** (group-anchored L1/L2), so this '
            'only confirms the NMF aspect partition reflects its own input — expected high, '
            "NOT a quality signal and NOT comparable to the Baseline's held-out recovery."
        )
        true_labels = can['_group'].apply(
            lambda x: str(x).split(' > ')[0].strip()).tolist()
        W        = meta['W']
        pred_nmf = np.argmax(W, axis=1).tolist()
        metrics  = evaluate(true_labels, pred_nmf)
        st.metric('ARI (self-consistency)', metrics['ARI'])

        # ── legacy global aspect table (diagnostic only) ─────────────────────
        with st.expander('Legacy global NMF aspect table (diagnostic — not the '
                          'primary result)'):
            st.caption('Global NMF aspects are a coarse lexical fallback. The '
                       'hierarchy is built from per-variable role extraction — '
                       'see the Role Decomposition tab for the actual result.')
            alabs = meta.get('aspect_labels', [])
            W_df  = pd.DataFrame(
                W, columns=[f'Aspect {k+1}: {alabs[k][:30]}' for k in range(W.shape[1])])
            W_df.insert(0, 'Variable', can['_label'].tolist())
            st.dataframe(W_df.round(4), width='stretch')

with tabs[2]:
    st.markdown('### Role decomposition')
    st.caption('Each variable decomposed into measure / statistic / condition / subtype.')

    if can is None:
        st.info('Build a hierarchy first.')
    else:
        # ── Per-group summary ─────────────────────────────────────────────────
        st.markdown('#### Per-group routing')
        reg_rows = []
        for n in nodes:
            sp = n.get('structure_provenance', {})
            if sp.get('route') == 'group_anchor' and 'phrase_regularity' in sp:
                reg_rows.append({
                    'Group':       n.get('name', ''),
                    'Regularity':  sp.get('phrase_regularity'),
                    'Route used':  sp.get('route_used', '—'),
                })
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows).sort_values(
                'Regularity', ascending=False, na_position='last')
            st.dataframe(reg_df, width='stretch', hide_index=True)

        # ── Per-variable role table ───────────────────────────────────────────
        st.markdown('#### Per-variable role table')

        # Primary source: raw per-row LLM extractions captured in the audit
        # (complete — includes roles that didn't become tree levels, e.g.
        # condition values skipped by singleton prevention).
        # Fallback per variable: roles collected from tree-ancestor nodes.
        audit_roles_by_vi: dict = {}
        for a in (st.session_state.get('a2_per_row_audit') or []):
            vi_list_a = a.get('vi_list') or []
            for r in a.get('per_row_audit', []):
                ridx = r.get('row_idx')
                accepted = r.get('accepted_roles') or {}
                if ridx is not None and ridx < len(vi_list_a) and accepted:
                    audit_roles_by_vi[int(vi_list_a[ridx])] = accepted

        # Tree-walk fallback (roles that became hierarchy levels)
        node_map_disp = {int(n['id']): n for n in nodes}
        parent_lookup: dict = {}
        for n in nodes:
            for c in n.get('related', []):
                parent_lookup.setdefault(int(c), int(n['id']))

        def _tree_roles_for_attr(node_id: int) -> dict:
            roles_here: dict = {}
            cur = node_id
            while cur in parent_lookup:
                cur = parent_lookup[cur]
                cur_node = node_map_disp.get(cur)
                if not cur_node:
                    break
                lp = cur_node.get('label_provenance', {})
                role = (lp.get('role')
                         or cur_node.get('structure_provenance', {}).get('slot_role'))
                if role and role not in roles_here:
                    roles_here[role] = cur_node.get('name', '')
            return roles_here

        role_rows = []
        for vi, (_, crow) in enumerate(can.iterrows()):
            # Audit roles win; tree roles fill any gaps
            roles_here = dict(_tree_roles_for_attr(vi + 1))   # attribute ids = 1..n
            for r, v in (audit_roles_by_vi.get(vi) or {}).items():
                if v:
                    roles_here[r] = v
            row = {
                'Group':    str(crow.get('_group', '')).split(' > ')[0].strip(),
                'Variable': str(crow.get('_label', '')),
            }
            for std_role in ('measure', 'statistic', 'condition',
                              'subtype', 'outcome', 'modifier'):
                row[std_role.title()] = roles_here.pop(std_role, '')
            if roles_here:
                row['Other roles'] = '; '.join(
                    f'{r}: {v}' for r, v in roles_here.items())
            role_rows.append(row)

        if role_rows:
            role_df = pd.DataFrame(role_rows)
            st.dataframe(role_df, width='stretch', hide_index=True)
            st.download_button(
                'Download per-variable role CSV',
                data=role_df.to_csv(index=False).encode('utf-8'),
                file_name=f'{safe_name(project_name)}_approach2_role_decomposition.csv',
                mime='text/csv',
            )

        # ── Per-row LLM extractor audit ───────────────────────────────────────
        audits = st.session_state.get('a2_per_row_audit', []) or []
        if audits:
            st.markdown('#### Extraction audit')

            # Summary table per group
            sum_rows = []
            for a in audits:
                sum_rows.append({
                    'Group':         a.get('group_name', ''),
                    'Rows':          a.get('n_rows', 0),
                    'Coverage':      a.get('coverage'),
                    'Valid':         a.get('valid'),
                    'Roles found':   ', '.join(a.get('roles_final', []))[:60],
                    'Reasons':       ', '.join(f'{k}:{v}' for k, v in
                                                (a.get('summary', {}) or {}).items()),
                })
            st.dataframe(pd.DataFrame(sum_rows), width='stretch',
                          hide_index=True)

            # Drill-down per group
            grp_names = [a.get('group_name', '?') for a in audits]
            if grp_names:
                sel_grp = st.selectbox(
                    'Drill into a group to see per-row proposals + rejections:',
                    grp_names)
                sel_audit = next((a for a in audits
                                    if a.get('group_name') == sel_grp), None)
                if sel_audit:
                    row_rows = []
                    for r in sel_audit.get('per_row_audit', [])[:60]:
                        accepted = r.get('accepted_roles', {}) or {}
                        rejected = r.get('rejected', []) or []
                        row_rows.append({
                            'Row #':      r.get('row_idx', ''),
                            'Description': r.get('description_snippet', ''),
                            'Accepted':   '; '.join(f'{k}={v}'
                                                     for k, v in accepted.items())[:140],
                            'Rejected':   '; '.join(
                                f'{x[0]}={x[1]!r} (missing stems: {x[2]})'
                                if isinstance(x, (list, tuple)) and len(x) >= 3
                                else str(x) for x in rejected)[:200],
                            'Reason':     r.get('reason', ''),
                        })
                    if row_rows:
                        st.dataframe(pd.DataFrame(row_rows),
                                      width='stretch', hide_index=True)
                        # Download as CSV for offline analysis
                        csv_bytes = pd.DataFrame(row_rows).to_csv(index=False).encode('utf-8')
                        st.download_button(
                            'Download per-row audit for this group',
                            data=csv_bytes,
                            file_name=f'{safe_name(project_name)}_audit_{safe_name(sel_grp)}.csv',
                            mime='text/csv',
                        )
        else:
            st.info('No role assignments recorded yet — Option D may have '
                    'fallen back to slot mining or aspect clustering for all '
                    'groups in this dataset.')

with tabs[3]:
    st.markdown('### Label provenance')
    st.caption('Audit trail: which stage produced each node label.')
    rows = []
    for n in nodes:
        if n.get('type') != 'aggregation':
            continue
        p = n.get('label_provenance', {})
        s = n.get('structure_provenance', {})
        rows.append({
            'Node':         n.get('name', ''),
            'Source':       p.get('label_source', '—'),
            'Route':        s.get('route', '—'),
            'Aspect method': s.get('aspect_method') or '—',
            'Silhouette':   s.get('silhouette') if s.get('silhouette') is not None else '—',
            'LLM used':     p.get('llm_used', False),
            'LLM rejected': p.get('llm_rejected', False),
            'LLM proposed': p.get('llm_raw_label', ''),
            'LLM reason':   p.get('llm_reason', '')[:60],
            'Confidence':   round(float(p.get('confidence', 1.0)), 3),
            'Evidence':     ', '.join(str(t) for t in p.get('evidence_terms', []))[:120],
        })
    if not rows:
        st.info('No internal nodes yet — build a hierarchy first.')
    else:
        prov_df = pd.DataFrame(rows)

        # ── Labels by source ──────────────────────────────────────────────────
        source_counts = prov_df['Source'].value_counts()
        st.write('**Labels by source**')
        cols_src = st.columns(min(5, max(2, len(source_counts))))
        for i, (src, cnt) in enumerate(source_counts.items()):
            cols_src[i % len(cols_src)].metric(str(src), int(cnt))

        # ── Structure routes ──────────────────────────────────────────────────
        am_counts = prov_df['Aspect method'].value_counts()
        st.write('**Structure routes used**')
        cols_am = st.columns(min(5, max(2, len(am_counts))))
        for i, (am, cnt) in enumerate(am_counts.items()):
            cols_am[i % len(cols_am)].metric(str(am), int(cnt))

        # ── LLM usage — split per-row extraction from the downstream refiner ──
        # Per-row nodes are LLM-BUILT (source 'per_row_llm_role'); the refiner
        # only renames deterministically-labeled nodes (source 'llm' when its
        # proposal is accepted).  Counting them together made the panel read
        # "N calls, 0 accepted" even on a fully successful build.
        n_per_row        = int((prov_df['Source'] == 'per_row_llm_role').sum())
        refiner_accepted = int((prov_df['Source'] == 'llm').sum())
        refiner_rejected = int(((prov_df['LLM rejected'] == True)  # noqa: E712
                                 & (prov_df['Source'] != 'per_row_llm_role')).sum())
        st.write('**LLM usage**')
        cL1, cL2, cL3 = st.columns(3)
        cL1.metric('Per-row extraction nodes', n_per_row)
        cL2.metric('Refiner accepted', refiner_accepted)
        cL3.metric('Refiner rejected', refiner_rejected)
        if refiner_accepted == 0 and refiner_rejected == 0 and n_per_row > 0:
            st.caption('Label refiner did not run — the tree was built entirely '
                        'by per-row extraction, leaving no deterministic labels '
                        'to refine.')
        if refiner_rejected > 0:
            with st.expander('Rejected refiner proposals'):
                rej = prov_df[(prov_df['LLM rejected'] == True)  # noqa: E712
                              & (prov_df['LLM proposed'].astype(str).str.len() > 0)]
                if len(rej):
                    st.dataframe(rej[['Node', 'LLM proposed', 'LLM reason']],
                                  width='stretch', hide_index=True)

        # ── Full provenance table ─────────────────────────────────────────────
        st.write('**Full per-node provenance**')
        st.dataframe(prov_df, width='stretch', hide_index=True)

with tabs[4]:
    if can is not None:
        st.dataframe(can.drop(columns=['_row'], errors='ignore'),
                     width='stretch')

with tabs[5]:
    # ── derive a per-CSV base name from the uploaded files ────────────────────
    # Uses the actual uploaded file names so different CSVs get different
    # output filenames (e.g. ai-mind-…json vs HCP_S1200_…json).
    csv_basis = ''
    if can is not None and '_source' in can.columns:
        sources = [str(s) for s in can['_source'].dropna().unique().tolist()]
        # Drop extensions, join with '+' if multiple files merged
        bases = []
        for s in sources:
            stem = Path(s).stem
            bases.append(safe_name(stem))
        csv_basis = '+'.join(bases) if bases else safe_name(project_name)
    if not csv_basis:
        csv_basis = safe_name(project_name)

    lod_fname = f'{csv_basis}_approach2_lod.json'

    st.caption(f'Filename basis: **{csv_basis}**  '
                f'(taken from the uploaded CSV — different CSVs export under different names)')

    col1, col2 = st.columns(2)
    with col1:
        if nodes:
            st.download_button(
                'LoD tree JSON',
                data=json.dumps(nodes, indent=2, ensure_ascii=False).encode(),
                file_name=f'{csv_basis}_approach2_lod.json',
                mime='application/json',
                width='stretch',
            )
    with col2:
        if can is not None:
            st.download_button(
                'Canonical CSV',
                data=can.to_csv(index=False).encode('utf-8'),
                file_name=f'{csv_basis}_approach2_canonical.csv',
                mime='text/csv',
                width='stretch',
            )

    st.divider()
    # ── Save directly into the project's outputs/approach_2/ folder ────────────
    _out_dir = Path(__file__).resolve().parent / 'outputs' / 'approach_2'
    st.markdown('### Save to project folder')
    st.caption(
        'The download buttons above go to your browser’s Downloads folder (a browser '
        f'restriction). This button instead writes the files into `{_out_dir}` with the '
        'dataset name — convenient for `evaluate_all.py`.'
    )
    if st.button('Save all to outputs/approach_2/', type='primary',
                 width='stretch'):
        try:
            _out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            if nodes:
                (_out_dir / f'{csv_basis}_approach2_lod.json').write_text(
                    json.dumps(nodes, indent=2, ensure_ascii=False), encoding='utf-8')
                saved.append(f'{csv_basis}_approach2_lod.json')
            if can is not None:
                can.to_csv(_out_dir / f'{csv_basis}_approach2_canonical.csv', index=False)
                saved.append(f'{csv_basis}_approach2_canonical.csv')
            st.success(f'Saved to `{_out_dir}`:\n\n- ' + '\n- '.join(saved))
        except Exception as _e:
            st.error(f'Could not save: {_e}')

with tabs[6]:
    st.markdown("""
## Approach 2 — Role-Decomposed Hierarchy via SBERT Phrase Clustering
### Option D primary route + slot mining + FASTopic + constrained LLM labels

### Algorithm

```
Step 1  Build metadata text objects (variable name + description + group)
        [GON §3] — Gonçalves et al. (2019)

Step 2  Group-anchored L1/L2 structure  [NEW]
        — detected group columns → path nodes (no hardcoding)
        — e.g. category > assessment, or task > variant
        — falls back to global NMF if no groups detected

Step 3a Phrase-slot mining (slot-first routing)  [IE / slot induction]
        — For each terminal group, attempt deterministic phrase-slot
          decomposition of variable descriptions:
            • extract concept-prefix per row
            • mine repeated n-grams (1–4 tokens, ≥2 rows)
            • compute mutual-exclusion across phrase pairs:
                M[a,b] = 1 − cooc[a,b] / min(count[a], count[b])
            • cluster phrases by mutual exclusion → slots
        — A slot is a set of phrases that rarely co-occur within a row but
          each co-occur with phrases from other slots (= alternatives at
          the same semantic position).
        — Activates only when slot structure is statistically strong:
            • ≥ 2 slots discovered
            • ≥ 55% row coverage
            • each slot ≥ 2 distinct phrases
        — No domain hardcoding: phrase content is discovered from the data;
          slot names = the highest-coverage phrase in each slot.
        — When valid, the local hierarchy is built directly from slot values.
        — When invalid (free-form descriptions, e.g. parts of HCP), routing
          falls through to NMF (Step 3b).

Step 3b Local NMF aspect discovery per terminal group  [ZHU §3.1 adapted]
        — Fallback path when slot mining does not apply.
        — NMF runs inside each group, not across all variables
        — prevents globally-dominant terms from polluting local aspects
        — K selected by reconstruction-error elbow (deterministic)
        — falls back to global embeddings for small groups (< min_local_nmf_size)

Step 4  Per-aspect variable representations  [ZHU §3.1]
        — for each aspect k: filter text to top-T terms → SBERT or masked TF-IDF

Step 5  Independent per-aspect GMM clustering  [ZHU §3.2]
        — GMM with diagonal covariance + BIC for stable k selection
        — runs inside each group's aspect space

Step 6  Simplified best-aspect split  [ZHU §3.3 adapted]
        — at each node: evaluate all K aspects by silhouette score
        — highest silhouette → GMM split → child aggregation nodes
        — NOTE: this is a silhouette-based greedy split, not the full
          probabilistic search of Zhu et al. Eq. 6/7
        — singleton prevention: 1-variable clusters attach directly (no wrapper node)

Step 5a UPSTREAM LLM phrase-role classification  [TopicGPT, NAACL 2024 adapted]
        — One LLM call per terminal group:
          input  : (i) all repeated mined phrases (verbatim from the corpus)
                   (ii) 2–4 sample variable descriptions for context
                   (iii) the group name
          output : {role_name: [phrases]}  e.g. {measure: [...], statistic: [...],
                                                  condition: [...]}
        — Anti-hallucination:
          • every phrase in the returned roles MUST match an input phrase
            verbatim (validator drops anything else)
          • role names must be 1–2 generic English words (drops fancy/long names)
          • ≥ 2 valid roles required; else falls back to mutual-exclusion
        — Phrase ↔ role mapping drives the slot hierarchy in Step 6:
          measure-like roles become outer levels, condition-like roles inner.
        — STRUCTURAL use of the LLM (TopicGPT discovers topics).  Distinct
          from the downstream label-refiner step (TopicTag, see Step 7e).

Step 7  Node labeling  [ZHU §4.3 / TopicTag DocEng 2024]
        a) description-prefix phrase shared by ≥60% of cluster
        b) group-purity prefix: if ≥70% share one _group top-level value
        c) data-driven boilerplate + FIELD_NAME_NOISE filter
        d) bigram-preferred discriminative TF-IDF suffix
        e) OPTIONAL downstream LLM refinement [TopicTag]:
           — receives only evidence terms + parent path + sample descriptions
           — strict grounding check: every label word must appear in evidence
           — rejected proposals fall back to deterministic label
           — provenance stored on each node (label_source, confidence, evidence_terms)

Step 8  Evaluation  [ZHU Table 2 / TraCo AAAI 2024 / TICL §3.4]
        — NMI, ARI, Purity vs. detected group ground-truth
        — parent-child coherence (TraCo affinity proxy)
        — sibling diversity (TraCo diversity proxy)
        — label-provenance audit table
```

### Key design decisions

| Decision | Rationale |
|---|---|
| FASTopic replaces NMF as primary aspect discovery | NMF (1999) is lexical only; FASTopic (NeurIPS 2024) uses pretrained Transformer + Dual Semantic-relation Reconstruction → semantic, not lexical. |
| NMF kept as fallback | Required for very small groups or when FASTopic / SBERT model is unavailable. |
| Slot mining tried first | Decomposes variables along multiple semantic dimensions before any topic model. No document-level method (NMF, BERTopic, FASTopic) can do this — they all collapse one variable into one vector. |
| No facet trees | Removed: a single coherent LoD tree is easier to defend than parallel views of one clustering. |
| Deterministic labels = default thesis result | Reproducible without API access. LLM is opt-in re-phrasing only. |
| LLM via local Ollama | Localhost OpenAI-compatible endpoint (`http://localhost:11434/v1`) → LLM ON by default whenever Ollama is reachable; easy to disable. Override `OLLAMA_URL` / `OLLAMA_MODEL` env vars for non-default deployments. No external API, no key management, fully reproducible from a known model checkpoint. |
| Strict LLM grounding | Every label word must appear in evidence — labels come from the CSV, LLM only rewords. |
| Per-node provenance | Audit trail: `label_source ∈ {description_prefix, tfidf_bigram, group_anchor, phrase_slot, llm, fallback}`. |

### Thesis wording (defense-safe)

*Approach 2 is a dataset-constrained multi-aspect hierarchy with strict separation
between structural decisions and label generation. The hierarchy topology is produced
deterministically: detected group metadata anchors the upper levels, IE-style phrase-slot
mining decomposes variables along multiple semantic dimensions when description structure
permits, and FASTopic (NeurIPS 2024) discovers latent semantic aspects in the remaining
groups, with NMF retained as a lexical fallback for small groups. Concept labels are
generated by a deterministic five-stage pipeline whose evidence comes exclusively from
the dataset itself. An optional TopicTag-style LLM refinement layer may re-phrase these
labels, but every LLM proposal must pass a grounding check — each word in the proposed
label must appear in the extracted evidence — and every label records its provenance
(source stage, confidence, evidence terms). The LLM can neither alter the tree structure
nor introduce vocabulary absent from the input CSV.*

### Papers used

| Ref | Citation | Role in this method |
|---|---|---|
| [ZHU] | Zhu et al. (2025). *Context-Aware Hierarchical Taxonomy Generation via LLM-Guided Multi-Aspect Clustering.* EMNLP 2025. | Main scaffold — adapted (FASTopic+NMF replace LLM aspect generation; greedy silhouette replaces Eq. 6/7 search). |
| [FASTopic] | Wu et al. (2024). *FASTopic: Pretrained Transformer is a Fast, Adaptive, Stable, and Transferable Topic Model.* NeurIPS 2024 (arXiv:2405.17978). | Recent SOTA replacement for NMF — semantic topic discovery via Dual Semantic-relation Reconstruction with optimal transport. |
| [IE-Slot] | Established IE literature on slot induction (surveyed *ACM Computing Surveys* 2022). | Phrase-slot mining adaptation — decomposes one variable into multiple alternative-phrase signals before clustering. |
| [GON] | Gonçalves et al. (2019). ESWC 2019. | Canonical metadata text-object construction. |
| [TopicGPT] | Pham et al. (2024). *TopicGPT: A Prompt-based Topic Modeling Framework.* NAACL 2024 (arXiv:2311.01449). | **STRUCTURAL** upstream LLM use — discovers semantic-role schema from mined phrases (one call per group); drives slot ordering in the hierarchy. Anti-hallucination contract: every phrase verbatim from corpus. |
| [TopicTag] | Eren et al. (2024). DocEng 2024 (arXiv:2407.19616). | Constrained LLM label-refinement pattern — LLM only names existing clusters, never modifies structure. Downstream use only. |
| [Qwen2.5] | Qwen Team (2024). *Qwen 2.5 Technical Report.* arXiv:2412.15115. | Open instruction-tuned model used as the local LLM (via Ollama) for label refinement. Replaces a hosted LLM for full offline reproducibility. |
| [TraCo] | Wu et al. (2024). AAAI 2024 (arXiv:2401.14113). | Diagnostic metrics: parent-child coherence + sibling diversity. |
| [TaxoAdapt] | Kargupta et al. (2025). ACL 2025 (arXiv:2506.10737). | Multidimensional taxonomy motivation. |
| [SC-Taxo] | (2026). arXiv:2605.00620. | Future work — bidirectional semantic consistency. |
| [BISE-26] | Motamedi, Novalija, Rei (2026). Springer BISE. | Validates group-anchored entry strategy. |
| [TICL] | Kejriwal et al. (2022). EAAI 108, 104548. | Concept-label evaluation framework. |

### Known limitations (honest)

* **FASTopic is still document-level** — better than NMF semantically, but a single variable
  is still one vector. Multi-dimension decomposition relies on phrase-slot mining.
* **Greedy silhouette split** — not the full probabilistic search of Zhu et al. Eq. 6/7.
* **TraCo metrics are diagnostic only** — measured, not enforced (no neural transport plan).
* **LLM labels are only as recent as your Anthropic model** — model choice affects reproducibility;
  the deterministic pipeline is the canonical thesis result.
""")
