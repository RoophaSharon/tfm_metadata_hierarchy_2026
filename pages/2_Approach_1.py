# approach 1.py — Automatic Metadata Hierarchy Builder — Approach 1
#
# Algorithm (no hardcoded domain-specific labels):
#   1. Read metadata → detect roles → build canonical schema
#   2. Extract candidate concepts automatically from metadata text
#   3. Detect domain → select external sources
#   4. Retrieve concept TABLE from external sources (Wikidata, Wikipedia, WordNet, BioPortal)
#   5. Embed variables + concept table (SBERT or TF-IDF fallback)
#   6. Compute N×M cosine similarity matrix [GON] — variables × concepts
#   7. Score concept assignment: embedding + string + frequency + source + hierarchy
#   8. Build task/group-first hierarchy using automatically assigned concept labels
#   9. HiExpan refinement: sibling coherence, width expansion, depth expansion, global opt
#  10. VIANNA LoD tree + Castanet parallel facets
#  11. Export with label provenance
#
# Papers:
#   [GON] Gonçalves et al. — biomedical metadata alignment via N×M concept similarity matrix
#   [TAX] Taxonomizer (Sultanum et al.) — leaf=attribute, internal node=abstract group
#   [HIE] HiExpan (Shen et al.) — width/depth expansion, sibling coherence, global opt
#   [CAS] Castanet — parallel faceted hierarchies over the same variable set

from __future__ import annotations
import csv, json, re, time, warnings
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import nltk
    for _pkg in ('wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'corpora/{_pkg}')
        except LookupError:
            nltk.download(_pkg, quiet=True)
    from nltk.corpus import wordnet as wn
    _WORDNET_AVAILABLE = True
except Exception:
    _WORDNET_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Metadata Hierarchy — Approach 1', page_icon='🌳', layout='wide')
st.title('Metadata Hierarchy Builder — Approach 1')
st.caption(
    'Automatic concept-label extraction from metadata text + HiExpan refinement + Castanet facets. '
    'External enrichment (Wikidata / Wikipedia / PubMed) activates automatically for biomedical, '
    'cognitive, and neurological domains.'
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LEAF_KEYS  = 'variable var field column attribute name code id item indicator question measure concept'.split()
GROUP_KEYS = 'task category domain module section table dataset assessment test variant group topic instrument form subscale construct'.split()
TEXT_KEYS  = 'description definition desc label title question meaning note notes text display full details explanation comment'.split()
META_KEYS  = 'type dtype data_type datatype unit units format decimal precision values value coding codebook range min max scale'.split()

RELATION_TYPES = {
    'has_measure':     'has measure type',
    'is_statistic_of': 'is statistic of',
    'has_condition':   'has condition',
    'part_of':         'part of',       # Wikidata P361
    'instance_of':     'instance of',   # Wikidata P31
    'subclass_of':     'subclass of',   # Wikidata P279
    'belongs_to':      'belongs to',
    'related_to':      'semantically related to',
}

# Source confidence weights for concept scoring [GON]
SOURCE_CONFIDENCE = {
    'group_path':        0.95,
    'description_title': 0.91,  # [FIX4][TAX][LOB] Text before first colon in description — highly discriminative
    'bioportal':         0.92,
    'cognitive_atlas':   0.94,   # [C5] Cognitive Atlas — domain-specific for CANTAB/cognitive, above Wikidata
    'wikidata':          0.88,
    'wordnet':           0.83,
    'pubmed':            0.82,
    'wikipedia':         0.78,
    'metadata_tfidf':    0.65,
    'noun_phrase':       0.55,
}

# English stop words (standard, domain-agnostic)
_STOP = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','shall','should','may','might','must','can',
    'could','of','in','on','at','to','for','with','by','from','about','as','into',
    'through','during','before','after','above','below','between','each','all',
    'both','few','more','most','other','some','such','no','nor','not','only',
    'same','so','than','too','very','just','this','that','these','those','which',
    'who','when','where','why','how','what','and','but','or','if','then','because',
    'while','although','however','therefore','thus','hence','also','well','used',
    'using','use','based','given','defined','number','value','values','score',
}

# ─── KeyBERT / labelling configuration ───────────────────────────────────────
# These tune the KeyBERT label synthesizer used in the hybrid scorer.
#
# USE_NOUN_PHRASES — True: candidate phrases are NLTK POS-tagged noun phrases
#   (needs the 'averaged_perceptron_tagger' corpus); False: plain n-gram candidates
#   from tokens. False is robust for short CANTAB/AI-MIND descriptions and avoids the
#   extra NLTK dependency.
USE_NOUN_PHRASES  = False
# USE_CTFIDF — True: multiply KeyBERT cosine relevance by corpus IDF so dataset-wide
#   boilerplate (low IDF) is down-weighted; False: plain cosine-to-centroid.
USE_CTFIDF        = False
# KEYBERT_DIVERSITY — MMR redundancy penalty weight. 0 = pure argmax cosine-to-centroid
#   (pick the single most relevant phrase); 0.5 = standard MMR diversification.
KEYBERT_DIVERSITY = 0

# ─── Title-SEEDED KeyBERT label-scorer weights ───────────────────────────────
# Concept labels are FORMED FROM THE DESCRIPTIONS (KeyBERT candidate phrases over the
# cluster's member descriptions). The pre-colon title is a ranking SEED/anchor, not the
# label itself: LABEL_W_TITLE controls how strongly it biases the choice toward the
# human-canonical phrasing (this is "Guided/Seeded KeyBERT"). Set LABEL_W_TITLE=0 for a
# pure-description ablation. Magnitudes are relative (need not sum to 1).
LABEL_W_RELEVANCE = 0.45   # cosine(candidate, cluster centroid)  — description fit (α)
LABEL_W_TITLE     = 0.35   # cosine(candidate, pre-colon title)   — title influence (β)
LABEL_W_CONTRAST  = 0.15   # discriminativeness vs sibling clusters (γ)
# NOTE: node labels are formed from DESCRIPTIONS + pre-colon TITLE only. External
# ontology sources (Cognitive Atlas / Wikidata / WordNet / PubMed) inform the embedding
# space / semantic understanding but are never used to name a node — so there is no
# external-grounding term in the label score.

# Corpus IDF over description n-grams; populated in build_concept_hierarchy() and
# consumed by _keybert_label when USE_CTFIDF=True.
_CORPUS_IDF: dict = {}

# Active dataset domain; set in build_concept_hierarchy(), read by the hybrid label
# scorer's external-grounding signal (Cognitive Atlas vs Wikidata routing).
_ACTIVE_DOMAIN: str = 'general'

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def safe_name(name):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in name)

def try_read_csv(path):
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

def load_any(path):
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
    if s in ['.md', '.markdown']:
        rows = []
        for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if '|' not in ln:
                continue
            cells = [c.strip() for c in ln.strip().strip('|').split('|')]
            if cells and not all(re.fullmatch(r':?-{2,}:?', c or '') for c in cells):
                rows.append(cells)
        if len(rows) < 2:
            raise ValueError('No Markdown table found')
        header = rows[0]
        data = [r[:len(header)] + [''] * max(0, len(header) - len(r)) for r in rows[1:]]
        return pd.DataFrame(data, columns=header)
    raise ValueError(f'Unsupported: {s}')

def probably_raw(df):
    cols = [str(c).lower() for c in df.columns]
    return df.shape[1] > 20 and not any(any(k in c for k in TEXT_KEYS) for c in cols)

def raw_to_metadata(df):
    rows = []
    for c in df.columns:
        s = df[c]
        dtype = 'number' if pd.api.types.is_numeric_dtype(s) else 'string'
        sample = ', '.join(map(str, s.dropna().astype(str).unique()[:5]))
        rows.append({'name': str(c), 'description': f'Column dtype:{dtype}. Values:{sample}', 'dtype': dtype})
    return pd.DataFrame(rows)

import tempfile
def save_uploads(files):
    tmp = Path(tempfile.mkdtemp(prefix='meta_app1_'))
    paths = []
    for f in files:
        p = tmp / safe_name(f.name)
        p.write_bytes(f.getbuffer())
        paths.append(p)
    return paths

# ─────────────────────────────────────────────────────────────────────────────
# ROLE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def norm(c):
    return re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')

def kscore(c, keys):
    nc = norm(c)
    return sum(1 for k in keys if k in nc)

def profile_columns(df):
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
            'leaf_score':     4 * kscore(col, LEAF_KEYS)  + (3 if 0.5 <= ur <= 1 else 0) + (1 if avg < 80 else 0),
            'group_score':    4 * kscore(col, GROUP_KEYS) + (3 if 1 < nun < min(n * 0.5, 80) else 0) + (1 if avg < 60 else 0),
            'text_score':     5 * kscore(col, TEXT_KEYS)  + (4 if avg > 50 else 0) + (1 if non > 0.5 else 0),
            'metadata_score': 4 * kscore(col, META_KEYS)  + (2 if 1 < nun < min(n * 0.8, 100) else 0),
        })
    return pd.DataFrame(out)

def detect_roles(df):
    prof  = profile_columns(df)
    leaf  = prof.sort_values(['leaf_score', 'unique_ratio'], ascending=False).head(1)['column'].tolist()
    text  = prof[(prof.text_score >= 4) | (prof.avg_length > 80)].sort_values('text_score', ascending=False)['column'].tolist() or leaf.copy()
    group = prof[(prof.group_score >= 4) & (~prof.column.isin(leaf)) & (prof.unique_values > 1)].sort_values('group_score', ascending=False)['column'].head(3).tolist()
    meta  = prof[(prof.metadata_score >= 4) & (~prof.column.isin(text + leaf + group))].sort_values('metadata_score', ascending=False)['column'].head(5).tolist()
    # DDI/CDISC: representation columns must never become structural hierarchy levels [GON][TAX]
    # These substrings identify physical metadata — universally, across any domain.
    _META_SUBSTR_BLOCK = {
        'decimal', 'precision', 'unit', 'dtype', 'type', 'format', 'scale',
        'values', 'range', 'min', 'max', 'coding', 'codebook', 'missing',
    }
    def _col_is_repr(col_name):
        nc = re.sub(r'[^a-z0-9]', '', str(col_name).lower())
        return any(sub in nc for sub in _META_SUBSTR_BLOCK)
    # Force representation columns out of group and into metadata
    meta_extra = [c for c in prof['column'].tolist()
                  if _col_is_repr(c) and c not in text and c not in leaf and c not in meta]
    group = [c for c in group if not _col_is_repr(c)]
    meta  = list(dict.fromkeys(meta + meta_extra))[:8]
    return {'leaf_cols': leaf, 'group_cols': group, 'text_cols': text, 'metadata_cols': meta}, prof

def sv(x):
    return '' if pd.isna(x) else str(x).strip()

def guess_dtype(row, dtype_cols, label):
    joined = ' '.join(sv(row.get(c, '')) for c in dtype_cols).lower()
    if any(t in joined for t in ['num', 'int', 'float', 'double', 'decimal', 'continuous', 'number']):
        return 'number'
    if any(t in joined for t in ['string', 'text', 'char', 'category', 'categorical', 'nominal']):
        return 'string'
    if re.search(r'(name|country|gender|sex|site|visit|status)', label.lower()):
        return 'string'
    return 'determine'

def build_canonical(df, cfg, source):
    """[GON] Build unified metadata text object from any tabular metadata file."""
    leaf_cols  = cfg.get('leaf_cols', [])
    group_cols = cfg.get('group_cols', [])
    text_cols  = cfg.get('text_cols', [])
    meta_cols  = cfg.get('metadata_cols', [])
    if not leaf_cols:
        raise ValueError('Choose at least one leaf column')
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
        # _semantic_text: description values only, no "fieldname: " prefixes [TAX][GON]
        # TAX embeds description text, not the full metadata row.
        # This is the input to SBERT and TF-IDF candidate extraction.
        sem_parts = []
        for c in text_cols:
            v = sv(row.get(c, ''))
            if v:
                sem_parts.append(v)
        if not sem_parts:
            sem_parts = list(leaf_parts) if leaf_parts else []
        semantic_text = ' '.join(sem_parts) if sem_parts else text
        rows.append({
            '_source_file':    source,
            '_row_index':      int(i),
            '_leaf_label':     label,
            '_leaf_id':        f'{gpath}.{label}' if gpath != 'Ungrouped' else label,
            '_group_path':     gpath,
            '_text':           text,
            '_semantic_text':  semantic_text,
            '_dtype':          guess_dtype(row, meta_cols, label),
            '_raw':            row.to_dict(),
            '_concept_label':  '',
            '_concept_score':  0.0,
            '_concept_source': '',
            '_code_family':    '',
        })
    can = pd.DataFrame(rows)
    if can['_leaf_id'].duplicated().any():
        cnt = defaultdict(int)
        ids = []
        for lid in can['_leaf_id']:
            cnt[lid] += 1
            ids.append(lid if cnt[lid] == 1 else f'{lid}__{cnt[lid]}')
        can['_leaf_id'] = ids
    return can

# ─────────────────────────────────────────────────────────────────────────────
# [F3] EARLY FACET PRE-COMPUTATION [CAS]
# Castanet: parallel facets (Statistic, Condition) are orthogonal split dimensions.
# These must be available BEFORE build_concept_hierarchy so _cluster_and_label
# can use them for sub-splitting. detect_facets/build_castanet_facets is called
# AFTER the hierarchy build, which is too late — so we compute them here first.
# ─────────────────────────────────────────────────────────────────────────────
def precompute_stat_cond_facets(can):
    """
    Pre-compute _facet_cond on can (numeric experimental conditions only).
    Called before build_concept_hierarchy so that _cluster_and_label can use it to
    insert Condition sub-tiers.

    NOTE: the statistic tier (Mean / Median / SD / …) is NO LONGER computed here.
    It used to come from a hardcoded statistic vocabulary regex, which (a) is domain
    hardcoding and (b) is not derived from the data's own concept titles. Statistic
    depth is now produced data-drivenly by _nest_by_measure(), which discovers the
    shared measure phrase and keeps the residual (Mean/Median/SD) as children — no
    word list. Condition detection below stays: it is structural (a digit in the
    code validated against the description text), not a hardcoded vocabulary.
    [CAS] Castanet parallel facets · [HIE] HiExpan sub-set discovery
    """
    can = can.copy()
    sem_col = '_semantic_text' if '_semantic_text' in can.columns else '_text'

    # ── Condition: digit in variable code VALIDATED by description text ──────────
    # [FIX2][GON] Gonçalves et al. (ESWC 2019): structural code alignment must be
    # validated against description text — the description is the authoritative source.
    # Previous rule: any digit in the code = condition value → caused false labels
    # like "468" (from SWMBE468) and HCP numeric suffixes that are not conditions.
    # New rule: a digit is accepted as a condition only if it ALSO appears as a
    # standalone token in the variable's description text, confirming it is a real
    # experimental parameter (delay, boxes, items, etc.).
    _num_re = re.compile(r'(\d+)')
    def _extract_cond(row):
        code = str(row['_leaf_label']).split('/')[0].strip()
        hits = _num_re.findall(code)
        if not hits:
            return ''
        desc_text = str(row.get(sem_col, row.get('_text', ''))).lower()
        for digit in hits:
            # Accept digit only if it appears as a whole word in the description
            if re.search(r'\b' + re.escape(digit) + r'\b', desc_text):
                return digit
        return ''
    cond_col = can.apply(_extract_cond, axis=1)
    can['_facet_cond'] = cond_col.where(cond_col != '', '')

    return can

# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC EMBEDDER [TAX][GON]
# ─────────────────────────────────────────────────────────────────────────────
class SemanticEmbedder:
    """SBERT with TF-IDF+SVD fallback. [TAX] Word2Vec→SBERT; [GON] GloVe→SBERT.

    Critical fix: in TF-IDF mode, a single vectorizer+SVD is fit jointly on ALL
    texts (variables + concept entries) so both live in the same vector space.
    Without this, N×M cosine similarity between separately-fit spaces is meaningless.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name  = model_name
        self._model      = None
        self._using_st   = False
        self._joint_vec  = None   # shared TF-IDF vectorizer (fallback mode only)
        self._joint_svd  = None   # shared SVD (fallback mode only)
        self._joint_dim  = 64

    def load(self):
        if _ST_AVAILABLE:
            try:
                self._model    = SentenceTransformer(self.model_name)
                self._using_st = True
                return True, f'Loaded {self.model_name} (SBERT)'
            except Exception as e:
                return False, f'sentence-transformers failed: {e}'
        return False, 'sentence-transformers not installed — using TF-IDF+SVD fallback'

    def fit_joint(self, all_texts):
        """
        Call once with variable texts + concept full_texts combined BEFORE encoding.
        Ensures TF-IDF fallback uses a single shared vector space for N×M alignment.
        No-op when SBERT is active (SBERT is already a universal space).
        """
        if self._using_st:
            return
        clean = [str(t) for t in all_texts if str(t).strip()]
        if len(clean) < 2:
            return
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                              max_features=1000, min_df=1)
        X = vec.fit_transform(clean)
        n_comp = min(self._joint_dim, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(X)
            self._joint_vec = vec
            self._joint_svd = svd

    def encode(self, texts):
        if self._using_st and self._model is not None:
            embs = self._model.encode(texts, show_progress_bar=False,
                                      batch_size=64, normalize_embeddings=True)
            return np.array(embs)
        # TF-IDF fallback — use shared space if available
        clean = [str(t) for t in texts]
        if self._joint_vec is not None and self._joint_svd is not None:
            X    = self._joint_vec.transform(clean)
            embs = self._joint_svd.transform(X)
        else:
            # Independent fit (only before fit_joint is called — e.g. early pipeline stages)
            vec  = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                   max_features=1000, min_df=1)
            X    = vec.fit_transform(clean)
            n_comp = min(self._joint_dim, X.shape[1] - 1, X.shape[0] - 1)
            embs = (TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
                    if n_comp >= 2 else X.toarray().astype(float))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (embs / norms).astype(float)

    @property
    def backend(self):
        return self.model_name if self._using_st else 'TF-IDF+SVD (joint-fit, fallback)'

# ─────────────────────────────────────────────────────────────────────────────
# CODE / ACRONYM ANALYSIS [GON]
# Detects coded variable names and groups them by shared structural prefix.
# Gonçalves et al. use string-distance clustering before semantic alignment.
# ─────────────────────────────────────────────────────────────────────────────
def detect_coded_variables(can):
    """
    Returns mask of rows whose leaf label looks like a variable code:
    all-uppercase strings with digits, short, no spaces (e.g. DMSL0SD).
    """
    pattern = re.compile(r'^[A-Z][A-Z0-9_]{2,}$')
    return can['_leaf_label'].apply(lambda x: bool(pattern.match(str(x).strip().split('/')[0].strip())))

def cluster_codes_by_prefix(can):
    """
    [F7] Groups coded variable names by their structural prefix.
    Improvement over simple ^[A-Z]+ regex: uses longest-common-prefix detection
    so that codes without digits (DMSLADSD, DMSLSSD) join the same family as
    codes with digits (DMSL0SD, DMSL4SD, DMSL12SD).

    Algorithm:
      1. For each coded variable, extract the alphabetic prefix before first digit
         (same as before for codes WITH digits, e.g. DMSL from DMSL0SD).
      2. For codes WITHOUT digits, try progressively shorter prefixes until finding
         one shared by ≥2 other codes — so DMSLADSD tries "DMSLADSD", "DMSLADS",
         "DMSLA", "DMSL" → "DMSL" matches ≥2 others → family = "DMSL".
      3. Assign the LONGEST matching prefix as the family key.

    Result: DMSL0SD, DMSL4SD, DMSL12SD, DMSLADSD, DMSLSSD, DMSLSD all share
    family "DMSL" regardless of digit presence. Works on any CSV domain.
    """
    can = can.copy()
    coded_mask = detect_coded_variables(can)
    if not coded_mask.any():
        return can

    # Step 1: collect all codes and their alpha prefix before first digit
    idx_to_code   = {}
    idx_to_alpha  = {}
    for idx, row in can[coded_mask].iterrows():
        code = str(row['_leaf_label']).strip().split('/')[0].strip()
        idx_to_code[idx] = code
        m = re.match(r'^([A-Z]+)', code)
        idx_to_alpha[idx] = m.group(1) if m else code

    # Step 2: build prefix → {indices} map for all possible prefix lengths ≥ 3
    prefix_to_idxs = defaultdict(set)
    for idx, alpha in idx_to_alpha.items():
        for length in range(3, len(alpha) + 1):
            prefix_to_idxs[alpha[:length]].add(idx)

    # Step 3: for each code find the longest prefix with ≥2 total matching codes
    prefix_counts = {p: len(idxs) for p, idxs in prefix_to_idxs.items()}

    best_prefix = {}
    for idx, alpha in idx_to_alpha.items():
        chosen = None
        for length in range(len(alpha), 2, -1):  # try longest first
            candidate = alpha[:length]
            if prefix_counts.get(candidate, 0) >= 2:
                chosen = candidate
                break
        best_prefix[idx] = chosen

    # Step 4: assign — only use a prefix if it appears in ≥2 variables
    for idx, prefix in best_prefix.items():
        if prefix:
            can.at[idx, '_code_family'] = prefix
    return can


def expand_variable_codes(can):
    """
    [GON] Automatically expand variable code segments to human-readable terms.
    Three evidence sources — all data-driven, no hardcoded domain terms:

      1. Parenthetical patterns in description text:
         'DMS (Delayed Matching to Sample)' → DMS = Delayed Matching to Sample
      2. Repeated positional suffix across a code family:
         DMSL0SD, DMSL4SD → suffix 'SD' constant → search descriptions for 'SD' expansion
      3. Group name as expansion of code prefix:
         codes in group 'Delayed Matching to Sample' → prefix DMSL ≈ group name

    Returns dict: {segment → {'expansion': str, 'evidence': [str]}}
    """
    expansions = {}

    # Source 1: parenthetical patterns  "(ABBR)" or "(Full Name)"
    paren_re = re.compile(
        r'\b([A-Z]{2,8})\b\s*[\(\[]\s*([A-Za-z][^)\]]{3,80})\s*[\)\]]'
        r'|([A-Za-z][^(\[]{3,60})\s*[\(\[]\s*([A-Z]{2,8})\s*[\)\]]'
    )
    for text in can['_text'].fillna('').astype(str):
        for m in paren_re.finditer(text):
            if m.group(1):  # ABBR (Full Name)
                seg, exp = m.group(1), m.group(2).strip()
            else:           # Full Name (ABBR)
                seg, exp = m.group(4), m.group(3).strip()
            exp = exp.split('.')[0].split(';')[0].strip()
            if len(exp) >= 4 and seg not in expansions:
                expansions[seg] = {'expansion': exp,
                                   'evidence': ['description_parenthetical']}

    # Source 2: repeated positional suffix across a code family
    coded_mask  = detect_coded_variables(can)
    family_rows = defaultdict(list)
    for _, row in can[coded_mask].iterrows():
        fam = str(row.get('_code_family', ''))
        if fam:
            family_rows[fam].append(row)

    seg_tok = re.compile(r'([A-Z]{2,})')
    for fam, rows in family_rows.items():
        if len(rows) < 2:
            continue
        codes    = [str(r['_leaf_label']).strip().split('/')[0] for r in rows]
        all_segs = [seg_tok.findall(c) for c in codes]
        min_len  = min((len(s) for s in all_segs), default=0)
        for pos in range(-1, -min_len - 1, -1):
            vals = [s[pos] for s in all_segs if len(s) >= abs(pos)]
            if not vals or vals[0].isdigit():
                continue
            seg_val = vals[0]
            if all(v == seg_val for v in vals) and seg_val not in expansions:
                look_re = re.compile(
                    rf'\b{re.escape(seg_val)}\b[\s\-–:]*([A-Za-z][a-zA-Z ]+)',
                    re.IGNORECASE
                )
                for r in rows:
                    hit = look_re.search(str(r.get('_text', '')))
                    if hit:
                        exp = hit.group(1).strip().split('.')[0].split('(')[0].strip()
                        if 4 <= len(exp) <= 60:
                            expansions[seg_val] = {
                                'expansion': exp,
                                'evidence': [f'code_family_{fam}_positional_suffix']
                            }
                            break

    # Source 3: group name as prefix expansion
    for fam, rows in family_rows.items():
        if fam in expansions:
            continue
        groups = [str(r.get('_group_path', '')).split(' > ')[0].strip()
                  for r in rows
                  if str(r.get('_group_path', '')) not in ('', 'nan', 'Ungrouped')]
        if groups and groups[0].lower() != fam.lower():
            expansions[fam] = {'expansion': groups[0],
                               'evidence': ['group_name_match']}

    return expansions


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DETECTION
# Routes to domain-specific external sources automatically.
# ─────────────────────────────────────────────────────────────────────────────
_DOMAIN_SIGNALS = {
    'cognitive': [
        'reaction time', 'response time', 'memory', 'attention', 'executive',
        'cognitive', 'correct', 'error', 'delay', 'task', 'trial', 'stimulus',
        'recall', 'recognition', 'working memory', 'inhibition', 'processing speed',
        'latency', 'accuracy', 'hit', 'false alarm', 'miss',
    ],
    'biomedical': [
        'patient', 'clinical', 'diagnosis', 'treatment', 'disease', 'symptom',
        'medication', 'hospital', 'brain', 'neural', 'mri', 'fmri', 'eeg',
        'biomarker', 'genetic', 'phenotype', 'cohort', 'longitudinal',
    ],
    'finance': [
        'price', 'return', 'portfolio', 'equity', 'bond', 'yield', 'market',
        'stock', 'currency', 'gdp', 'inflation', 'revenue', 'profit', 'index',
    ],
    'environment': [
        'temperature', 'precipitation', 'climate', 'emission', 'pollution',
        'biodiversity', 'ecosystem', 'carbon', 'species', 'habitat', 'soil',
    ],
    'survey': [
        'questionnaire', 'likert', 'respondent', 'survey', 'agree', 'disagree',
        'strongly', 'satisfaction', 'attitude', 'opinion',
    ],
}

def detect_domain(can):
    """
    Detect domain from all metadata text.
    Returns domain string used to select external sources.
    """
    all_text = ' '.join(can['_text'].fillna('').astype(str).tolist()).lower()
    scores = {domain: sum(1 for sig in signals if sig in all_text)
              for domain, signals in _DOMAIN_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'general'

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE CONCEPT EXTRACTION FROM METADATA [GON][TAX]
# Mines the metadata text itself for candidate concept labels.
# No external source needed at this stage — purely data-driven.
# Sources: group path components, TF-IDF n-grams, repeated noun phrases.
# ─────────────────────────────────────────────────────────────────────────────
def extract_candidate_concepts_from_metadata(can, max_concepts=150):
    """
    Extract candidate concept labels from the metadata itself.
    Uses _semantic_text (description values only) to avoid field-name contamination [TAX][HIE][YAKE].
    Returns list of dicts: {label, full_text, frequency, source, tfidf_score}
    """
    # No hardcoded structural/boilerplate word list. Candidate EXTRACTION now keeps
    # everything that survives standard stop-word removal; boilerplate suppression is
    # done downstream and SEMANTICALLY in score_concepts_for_cluster() via the
    # specificity (semantic-IDF) signal, which is corpus-derived and dataset-agnostic.
    # A term like "Calculated Assessed Trials" is no longer blacklisted here — it is
    # simply ranked low because it is close to every group centroid. [GON][TaxoGen]
    _STRUCT_NOISE: set = set()

    candidates = {}  # label_lower → dict

    # Boolean / value-state noise tokens — candidates made entirely of these words
    # are documentation artefacts, NOT semantic concepts [FIX5][TaxoGen KDD 2018]
    _BOOL_NOISE = {
        'true', 'false', 'yes', 'no', 'completed', 'incomplete',
        'missing', 'unknown', 'none', 'other', 'na', 'n/a',
        'not', 'done', 'pending', 'available', 'unavailable',
    }

    def _is_bool_noise(label):
        """Return True if every non-stop word in label is a boolean/value-state token."""
        words = set(re.findall(r'\b[a-z]{2,}\b', label.lower())) - _STOP
        return len(words) > 0 and words.issubset(_BOOL_NOISE)

    sem_col = '_semantic_text' if '_semantic_text' in can.columns else '_text'

    # ── Source 0: Description titles — colon-structured descriptions only ──────
    # [FIX4][TAX] Taxonomizer: "text before first colon" is the concept anchor
    # ONLY when a genuine colon separates label from explanation, e.g.
    # "Reaction Time: time from stimulus to response" → anchor = "Reaction Time".
    # [Sultanum & Mueller, IEEE TVCG 2019]; [Lobo et al., ISWC 2023]
    #
    # [FIX-R1] Cross-task leakage fix:
    # CANTAB/AI-Mind descriptions have NO colon — they are plain text like
    # "DMS Correct Latency Standard Deviation 0 second delay". The naive
    # implementation treated the FULL description as the title, so
    # "Prm Correct Latency (Sd) Delayed" entered the global pool and was
    # assigned to DMS clusters (high SBERT similarity).
    # Gate: only accept title_raw that is <80% the length of the full description.
    # This confirms a colon genuinely separates a short label from a long explanation.
    # CANTAB descriptions (no colon → title == full text) are skipped entirely.
    # Additional guards: strip task prefix (data-driven), ≤4 words, freq ≥ 2.

    # Discover top-level task tokens from _group_path — data-driven, NOT hardcoded
    top_task_tokens: set = set()
    if '_group_path' in can.columns:
        for _gp in can['_group_path'].dropna().astype(str):
            _first = _gp.split(' > ')[0].strip()
            if _first and _first.lower() not in ('ungrouped', 'nan', ''):
                top_task_tokens.add(_first.lower())

    _task_pfx_re = (
        re.compile(
            r'^(?:' + '|'.join(re.escape(t)
                               for t in sorted(top_task_tokens, key=len, reverse=True))
            + r')\s+',
            re.IGNORECASE,
        )
        if top_task_tokens else None
    )

    title_counts: dict = defaultdict(int)
    for raw_text in can[sem_col].fillna('').astype(str):
        desc_part = raw_text
        if 'description:' in raw_text.lower():
            desc_part = re.split(r'description\s*:', raw_text, maxsplit=1,
                                  flags=re.IGNORECASE)[-1].strip()
        full_len  = len(desc_part.strip())
        if full_len < 3:
            continue
        title_raw = re.split(r'[:|]', desc_part)[0].strip()
        # Gate: title must be genuinely shorter than the full description.
        # If title ≥ 80% of full text there is no colon structure → skip.
        if len(title_raw) >= full_len * 0.80:
            continue
        title_clean = re.sub(r'^[\s\d\W]+', '', title_raw).strip()
        if len(title_clean) < 3 or title_clean.replace(' ', '').isdigit():
            continue
        # Strip leading task prefix (data-driven)
        if _task_pfx_re:
            title_clean = _task_pfx_re.sub('', title_clean).strip()
        if len(title_clean) < 3:
            continue
        # ≤4 words: a concept anchor must be a short label, not a sentence
        if len(title_clean.split()) > 4:
            continue
        if _is_bool_noise(title_clean):  # [FIX5]
            continue
        title_counts[title_clean] += 1

    for title, cnt in title_counts.items():
        if cnt < 2:   # must appear in ≥2 variables to be a real shared concept
            continue
        # Reject titles containing underscores — always raw variable/column names
        if '_' in title:
            continue
        key = title.lower()
        if key not in candidates:
            candidates[key] = {
                'label':             title,
                'full_text':         title,
                'frequency':         cnt,
                'source':            'description_title',
                'tfidf_score':       0.95,
                'cross_group_count': 0,
            }
        else:
            candidates[key]['frequency'] = max(candidates[key]['frequency'], cnt)

    # Source 1: Group path components — already structured, highest quality
    for gpath in can['_group_path'].dropna().unique():
        for part in str(gpath).split(' > '):
            part = part.strip()
            if len(part) >= 3 and part.lower() not in ('ungrouped', 'nan', 'none', ''):
                key = part.lower()
                if key not in candidates:
                    candidates[key] = {
                        'label': part, 'full_text': part,
                        'frequency': 0, 'source': 'group_path', 'tfidf_score': 1.0,
                        'cross_group_count': 0,  # [FIX6]
                    }
                candidates[key]['frequency'] += 1

    # Source 2: TF-IDF n-grams from _semantic_text (description values only) [TAX][YAKE]
    texts = can[sem_col].fillna('').astype(str).tolist()
    if texts:
        try:
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3),
                                  max_features=min(600, max_concepts * 4), min_df=1)
            X   = vec.fit_transform(texts)
            terms      = vec.get_feature_names_out()
            mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
            freq_arr   = np.asarray((X > 0).sum(axis=0)).flatten()
            for i in np.argsort(mean_tfidf)[::-1][:max_concepts]:
                term  = terms[i]
                words = term.split()
                if not (len(term) >= 4 and any(c.isalpha() for c in term)
                        and not term.replace(' ', '').isdigit()):
                    continue
                # Skip stop words and structural noise (single-word filter)
                if len(words) == 1 and (term in _STOP or term.lower() in _STRUCT_NOISE):
                    continue
                # [FIX5][TaxoGen] Skip boolean/value-state noise candidates
                if _is_bool_noise(term):
                    continue
                # YAKE: single-word candidates penalised — prefer multi-word phrases
                score_mult = 0.5 if len(words) == 1 else 1.0
                key = term.lower()
                if key not in candidates:
                    candidates[key] = {
                        'label':         term,
                        'full_text':     term,
                        'frequency':     int(freq_arr[i]),
                        'source':        'metadata_tfidf',
                        'tfidf_score':   float(mean_tfidf[i]) * score_mult,
                        'cross_group_count': 0,  # [FIX6]
                    }
        except Exception:
            pass

    # Source 3: Repeated multi-word noun phrases from _semantic_text [TAX]
    phrase_re = re.compile(r'\b([a-z][a-z0-9]{1,}(?:\s+[a-z][a-z0-9]{1,}){1,3})\b')
    phrase_counts = defaultdict(int)
    for text in texts:
        for m in phrase_re.finditer(text.lower()):
            phrase = m.group(1)
            words  = phrase.split()
            if any(w not in _STOP and w not in _STRUCT_NOISE and len(w) >= 3 for w in words):
                phrase_counts[phrase] += 1

    for phrase, count in sorted(phrase_counts.items(), key=lambda x: -x[1]):
        if count >= 2 and len(phrase) >= 5:
            if _is_bool_noise(phrase):   # [FIX5]
                continue
            key = phrase.lower()
            if key not in candidates:
                candidates[key] = {
                    'label':             phrase,
                    'full_text':         phrase,
                    'frequency':         count,
                    'source':            'noun_phrase',
                    'tfidf_score':       0.0,
                    'cross_group_count': 0,  # [FIX6]
                }

    # ── [FIX6][TaxoGen][CAS] Cross-group boilerplate tagging ──────────────────
    # A concept that appears in EVERY top-level group is dataset-wide boilerplate
    # (e.g. "Calculated Assessed Trials" across DMS/MOT/PAL/SWM/…).
    # Count how many distinct top-level groups contain each candidate label.
    # The penalty is applied later in score_concepts_for_cluster().
    # Paper rationale — TaxoGen KDD 2018: contrastive term selection prefers
    # locally dominant, globally rare terms.  Castanet NAACL 2007: facet labels
    # must discriminate between top-level categories.
    if '_group_path' in can.columns:
        top_groups = (
            can['_group_path'].fillna('').astype(str)
               .apply(lambda p: p.split(' > ')[0].strip().lower())
        )
        all_top_groups = [g for g in top_groups.unique() if g not in ('', 'ungrouped', 'nan')]
        n_top_groups   = max(1, len(all_top_groups))

        # Build per-group text corpus for fast membership testing
        group_texts = {}
        for grp in all_top_groups:
            mask = top_groups == grp
            group_texts[grp] = ' '.join(can.loc[mask, sem_col].fillna('').astype(str)).lower()

        for key, cand in candidates.items():
            cand_words = set(re.findall(r'\b[a-z]{3,}\b', cand['label'].lower())) - _STOP
            if not cand_words:
                continue
            count_in_groups = sum(
                1 for grp_text in group_texts.values()
                if all(w in grp_text for w in cand_words)
            )
            cand['cross_group_count'] = count_in_groups
            cand['_n_top_groups']     = n_top_groups  # store for scorer

    # Sort: description_title / group_path first, then by tfidf_score, then by frequency
    _src_priority = {'group_path': 0, 'description_title': 1}
    result = sorted(
        candidates.values(),
        key=lambda x: (_src_priority.get(x['source'], 2),
                        -x['tfidf_score'], -x['frequency'])
    )
    return result[:max_concepts]

# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL CONCEPT SOURCES
# Build a concept TABLE (not just append text). Each entry has a full_text
# that is encoded by SBERT for the N×M alignment matrix.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def wikidata_search(term):
    """Wikidata entity search — returns concatenated descriptions. [GON][HIE]"""
    try:
        resp = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={'action': 'wbsearchentities', 'search': term,
                    'language': 'en', 'format': 'json', 'limit': 3},
            timeout=6, headers={'User-Agent': 'MetadataHierarchyTool/1.0'}
        )
        items = resp.json().get('search', [])
        descs = [it.get('description', '') for it in items if it.get('description')]
        return ' '.join(descs[:2])
    except Exception:
        return ''

@st.cache_data(ttl=3600, show_spinner=False)
def wikidata_broader(term):
    """
    P31=instance_of, P279=subclass_of, P361=part_of from Wikidata SPARQL.
    These are the beyond-is-a relations from [HIE].
    """
    try:
        sr = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={'action': 'wbsearchentities', 'search': term,
                    'language': 'en', 'format': 'json', 'limit': 1},
            timeout=5, headers={'User-Agent': 'MetadataHierarchyTool/1.0'}
        )
        items = sr.json().get('search', [])
        if not items:
            return []
        qid   = items[0]['id']
        sparql = f"""
        SELECT ?rel ?broaderLabel WHERE {{
          VALUES ?prop {{ wdt:P31 wdt:P279 wdt:P361 }}
          wd:{qid} ?prop ?broader .
          BIND(REPLACE(STR(?prop),'.*P','P') AS ?rel)
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en' . }}
        }} LIMIT 5
        """
        resp = requests.get(
            'https://query.wikidata.org/sparql',
            params={'query': sparql, 'format': 'json'},
            headers={'Accept': 'application/json', 'User-Agent': 'MetadataHierarchyTool/1.0'},
            timeout=10
        )
        results = resp.json().get('results', {}).get('bindings', [])
        rel_map = {'P31': 'instance_of', 'P279': 'subclass_of', 'P361': 'part_of'}
        out = []
        for r in results:
            rel = rel_map.get(r.get('rel', {}).get('value', ''), 'related_to')
            lbl = r.get('broaderLabel', {}).get('value', '')
            if lbl:
                out.append((rel, lbl))
        return out
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def wikipedia_summary(term):
    """Wikipedia intro paragraph. Taxonomizer trained on Wikipedia — same corpus. [TAX]"""
    try:
        resp = requests.get(
            'https://en.wikipedia.org/api/rest_v1/page/summary/' + term.replace(' ', '_'),
            timeout=6, headers={'User-Agent': 'MetadataHierarchyTool/1.0'}
        )
        extract = resp.json().get('extract', '')
        return extract[:300] if extract else ''
    except Exception:
        return ''

@st.cache_data(ttl=3600, show_spinner=False)
def pubmed_keywords(query):
    """PubMed enrichment — biomedical domain only. [GON]"""
    try:
        search = requests.get(
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            params={'db': 'pubmed', 'term': query[:100], 'retmax': 3, 'retmode': 'json'},
            timeout=8
        )
        ids = search.json().get('esearchresult', {}).get('idlist', [])
        if not ids:
            return ''
        fetch = requests.get(
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
            params={'db': 'pubmed', 'id': ','.join(ids), 'rettype': 'abstract', 'retmode': 'text'},
            timeout=10
        )
        return fetch.text[:400]
    except Exception:
        return ''

@st.cache_data(ttl=3600, show_spinner=False)
def bioportal_search(phrase, api_key):
    """BioPortal ontology search — biomedical domain, requires free API key. [GON]"""
    if not api_key:
        return []
    try:
        resp = requests.get(
            'https://data.bioontology.org/search',
            params={'q': phrase, 'pagesize': 3, 'display_links': 'false'},
            headers={'Authorization': f'apikey token={api_key}'},
            timeout=8
        )
        results = resp.json().get('collection', [])
        out = []
        for r in results:
            lbl = r.get('prefLabel', '')
            defn = ' '.join(r.get('definition', []))[:200]
            if lbl:
                out.append({
                    'label': lbl, 'full_text': f'{lbl}. {defn}' if defn else lbl,
                    'source': 'bioportal', 'frequency': 0, 'tfidf_score': 0.0,
                    'broader_relations': [],
                })
        return out
    except Exception:
        return []

def wordnet_lookup(phrase):
    """WordNet definition + hypernyms — synonym/hypernym expansion. [GON]"""
    if not _WORDNET_AVAILABLE:
        return None
    try:
        synsets = wn.synsets(phrase.replace(' ', '_'))
        if not synsets:
            synsets = wn.synsets(phrase.split()[0]) if phrase.split() else []
        if not synsets:
            return None
        ss  = synsets[0]
        defn = ss.definition()
        hypernyms = [h.name().replace('_', ' ').split('.')[0] for h in ss.hypernyms()[:3]]
        return {'definition': defn, 'hypernyms': hypernyms}
    except Exception:
        return None

def _is_acronym(phrase):
    """True if phrase is a short all-caps token — high Wikidata polysemy risk. [GON][BLINK]
    These tokens must not be queried raw; use the expanded form instead."""
    p = phrase.strip()
    return p.isupper() and 2 <= len(p) <= 6 and sum(c.isalpha() for c in p) >= 2

@st.cache_data(ttl=86400, show_spinner=False)
def cognitive_atlas_search(term):
    """Cognitive Atlas REST API — domain-specific for cognitive/neurological tasks.
    Preferred over Wikidata for cognitive domain codes. [GON]"""
    try:
        resp = requests.get(
            'https://www.cognitiveatlas.org/api/v-alpha/task',
            params={'search': term, 'format': 'json'},
            timeout=8, headers={'User-Agent': 'MetadataHierarchyTool/1.0'}
        )
        items = resp.json()
        if isinstance(items, list) and items:
            item = items[0]
            name = item.get('name', '')
            defn = item.get('definition_text', '') or item.get('alias', '')
            if name:
                return f'{name}. {defn[:250]}' if defn else name
    except Exception:
        pass
    return ''

def retrieve_concept_table(candidates, domain='general',
                            use_wikidata=True, use_wikipedia=False,
                            use_wordnet=True, use_pubmed=False,
                            bioportal_key='', progress_cb=None,
                            code_expansions=None):
    """
    Build a concept TABLE from candidates + external sources.
    Each entry: {label, full_text, source, frequency, tfidf_score, broader_relations}.
    full_text = label + external description → encoded by SBERT for N×M matrix.
    This is the right-hand side of the Gonçalves N×M alignment matrix. [GON]
    """
    # Start with all candidates as base entries
    table = {}  # label_lower → dict
    for c in candidates:
        key = c['label'].lower()
        table[key] = {
            'label':             c['label'],
            'full_text':         c['label'],
            'source':            c['source'],
            'frequency':         c.get('frequency', 0),
            'tfidf_score':       c.get('tfidf_score', 0.0),
            'broader_relations': [],
        }

    # Enrich top candidates with external sources
    api_candidates = sorted(candidates,
                             key=lambda x: (0 if x['source'] == 'group_path' else 1,
                                            -x.get('tfidf_score', 0), -x.get('frequency', 0)))[:60]
    n_api = len(api_candidates)
    code_expansions = code_expansions or {}

    for i, c in enumerate(api_candidates):
        if progress_cb:
            progress_cb(i / n_api)
        phrase = c['label']
        key    = phrase.lower()

        # Determine the query phrase — never query raw acronyms on Wikidata [GON][BLINK]
        if _is_acronym(phrase):
            exp = code_expansions.get(phrase, {}).get('expansion', '')
            query_phrase = exp if exp else None   # None = skip Wikidata entirely
        else:
            query_phrase = phrase

        # Cognitive Atlas (cognitive/neurological domain — before Wikidata) [GON]
        if domain in ('cognitive', 'neurological', 'biomedical') and query_phrase:
            cat_def = cognitive_atlas_search(query_phrase)
            if cat_def and key in table:
                table[key]['full_text'] = f'{phrase}. {cat_def}'
                table[key]['source']    = 'cognitive_atlas'

        # Wikidata — use expanded form for acronyms, skip if no expansion found
        if use_wikidata and query_phrase is not None:
            wd_desc = wikidata_search(query_phrase)
            wd_rel  = wikidata_broader(query_phrase)
            if key in table:
                if wd_desc and table[key]['source'] not in ('cognitive_atlas',):
                    table[key]['full_text']         = f'{phrase}. {wd_desc}'
                    table[key]['source']            = 'wikidata'
                    table[key]['broader_relations'] = wd_rel

        # WordNet — synonyms, hypernyms, definitions
        if use_wordnet and _WORDNET_AVAILABLE:
            wn_res = wordnet_lookup(phrase)
            if wn_res:
                wn_key = f'wordnet_{key}'
                table[wn_key] = {
                    'label':       phrase,
                    'full_text':   f'{phrase}. {wn_res["definition"]}',
                    'source':      'wordnet',
                    'frequency':   c.get('frequency', 0),
                    'tfidf_score': c.get('tfidf_score', 0.0),
                    'broader_relations': [('related_to', h) for h in wn_res.get('hypernyms', [])],
                }

        # Wikipedia (optional)
        if use_wikipedia and i < 20:
            wiki = wikipedia_summary(phrase)
            if wiki:
                wp_key = f'wikipedia_{key}'
                table[wp_key] = {
                    'label':       phrase,
                    'full_text':   f'{phrase}. {wiki[:200]}',
                    'source':      'wikipedia',
                    'frequency':   c.get('frequency', 0),
                    'tfidf_score': c.get('tfidf_score', 0.0),
                    'broader_relations': [],
                }

        # PubMed (biomedical only, optional)
        if use_pubmed and domain in ('biomedical', 'cognitive') and i < 8:
            pm = pubmed_keywords(phrase)
            if pm:
                pm_key = f'pubmed_{key}'
                table[pm_key] = {
                    'label':       phrase,
                    'full_text':   f'{phrase}. {pm[:200]}',
                    'source':      'pubmed',
                    'frequency':   c.get('frequency', 0),
                    'tfidf_score': c.get('tfidf_score', 0.0),
                    'broader_relations': [],
                }
            time.sleep(0.35)  # NCBI rate limit

        # BioPortal (biomedical only, optional API key)
        if bioportal_key and domain in ('biomedical', 'cognitive') and i < 20:
            for bp in bioportal_search(phrase, bioportal_key):
                bp_key = f"bioportal_{bp['label'].lower()}"
                table[bp_key] = bp

    return list(table.values())

# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT ALIGNMENT — N×M COSINE SIMILARITY [GON]
# Gonçalves et al. build an N×M similarity matrix between metadata field
# embeddings and ontology term embeddings, then rank alignments.
# Here: N=variable clusters, M=concept table entries.
# ─────────────────────────────────────────────────────────────────────────────
def _string_overlap(cluster_texts, concept_label):
    """
    Word overlap between cluster descriptions and concept label words.
    Measures string-level evidence that this concept matches this cluster.
    """
    concept_words = set(re.findall(r'\b[a-z]{3,}\b', concept_label.lower())) - _STOP
    if not concept_words:
        return 0.0
    cluster_combined = ' '.join(cluster_texts).lower()
    cluster_words    = set(re.findall(r'\b[a-z]{3,}\b', cluster_combined)) - _STOP
    overlap = len(concept_words & cluster_words) / len(concept_words)
    return float(overlap)

def _seq_sim(a, b):
    """SequenceMatcher ratio between two strings — for code/label similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def score_concepts_for_cluster(cluster_emb, concept_embs, concept_table, cluster_texts,
                                n_total_vars=None, member_embs=None,
                                sibling_centroids=None, ref_centroids=None,
                                corpus_centroid=None, own_group_centroid=None):
    """
    Fully-semantic, multi-signal concept scoring for one cluster.
    [GON] Gonçalves ESWC 2019 (IDF-weighted embeddings + cosine);
    [TaxoGen KDD 2018] contrastive term selection; [CAS] Castanet facet contrast.

    Every signal is cosine-in-embedding-space — no word-overlap, no hardcoded
    boilerplate lists. All references are data-derived, so it transfers to any set.

        score = 0.30 × fit       (mean cosine of label to THIS cluster's members)
              + 0.35 × contrast  (fit − best cosine to a SIBLING cluster, same task)
              + 0.25 × home      (cosine to OWN-task centroid − mean over all tasks)
              + 0.10 × source_conf
              − url_noise_penalty

    Why three signals, and what each one kills:
      • fit       — must actually describe this cluster.
      • contrast  — kills dataset-wide BOILERPLATE ("Calculated Assessed Trials"):
                    it sits in every sibling cluster too, so fit ≈ sibling-sim →
                    contrast ≈ 0. A real sub-topic ("Total Errors") is in its own
                    cluster but not the latency sibling → positive contrast.
      • home      — kills CROSS-TASK leakage ("Rvp 3 Targets" under DMS): it is far
                    from the DMS group centroid, so (cos to own group − mean over
                    groups) is negative → clipped to 0. A genuine DMS term is at or
                    above the cross-task average → positive. This is task-RELATIVE,
                    unlike a group-agnostic max−mean peak (which wrongly rewarded a
                    sharp RVP-specific label even while labelling a DMS cluster).
    `own_group_centroid` is the centroid of the current task's variables (passed in
    by the caller); `ref_centroids` are all top-level task centroids.

    Returns list of dicts sorted by score descending.
    """
    if concept_embs is None or len(concept_table) == 0:
        return []
    concept_embs = np.asarray(concept_embs, dtype=float)

    # Similarity of each candidate to this cluster's centroid
    emb_sims = cosine_similarity([cluster_emb], concept_embs)[0]

    # ── fit: mean cosine to the cluster's MEMBER embeddings (robust to outliers)
    if member_embs is not None and len(member_embs) > 0:
        fit = cosine_similarity(concept_embs, np.asarray(member_embs, dtype=float)).mean(axis=1)
    else:
        fit = emb_sims

    # ── contrast: fit minus closeness to the nearest SIBLING cluster (same task)
    if sibling_centroids is not None and len(sibling_centroids) > 0:
        sib_sims = cosine_similarity(concept_embs, np.asarray(sibling_centroids, dtype=float))
        contrast = np.clip(fit - sib_sims.max(axis=1), 0.0, 1.0)
    else:
        contrast = np.zeros(len(concept_table))

    # ── home: does the label belong to THIS task more than to tasks on average?
    # Task-relative — measured against the CURRENT group centroid, not a peak.
    home_active = False
    if own_group_centroid is not None and ref_centroids is not None and len(ref_centroids) >= 2:
        own_sim  = cosine_similarity(concept_embs, [own_group_centroid])[:, 0]
        all_mean = cosine_similarity(concept_embs, np.asarray(ref_centroids, dtype=float)).mean(axis=1)
        home = np.clip((own_sim - all_mean) * 3.0, 0.0, 1.0)
        home_active = True
    elif own_group_centroid is not None and corpus_centroid is not None:
        own_sim = cosine_similarity(concept_embs, [own_group_centroid])[:, 0]
        gen     = cosine_similarity(concept_embs, [corpus_centroid])[:, 0]
        home = np.clip((own_sim - gen) * 3.0, 0.0, 1.0)
        home_active = True
    else:
        # No task reference (e.g. single Ungrouped bucket): neutral, don't filter.
        home = np.full(len(concept_table), 0.34)

    src_sc = np.array([SOURCE_CONFIDENCE.get(c.get('source', 'noun_phrase'), 0.55)
                       for c in concept_table])

    # URL / HTML artifact penalty — strips documentation junk, not domain terms
    _url_noise_re = re.compile(
        r'\b(http|href|wiki|neurolex|org|www|definition|category|link|url)\b',
        re.IGNORECASE
    )
    noise_penalty = np.array(
        [0.35 if _url_noise_re.search(c['label']) else 0.0 for c in concept_table]
    )

    total = (0.30 * fit + 0.35 * contrast + 0.25 * home
             + 0.10 * src_sc - noise_penalty)

    # Reported only (provenance/debug) — not scored.
    str_sims = np.array([_string_overlap(cluster_texts, c['label']) for c in concept_table])

    results = []
    for i, concept in enumerate(concept_table):
        # Drop candidates that don't fit this cluster, or (when a task reference
        # exists) that belong to a DIFFERENT task — i.e. home collapsed to 0.
        if float(fit[i]) < 0.12:
            continue
        if home_active and float(home[i]) <= 0.0:
            continue
        results.append({
            'label':             concept['label'],
            'score':             float(total[i]),
            'embedding_sim':     float(emb_sims[i]),
            'coverage':          float(fit[i]),
            'contrast':          float(contrast[i]),
            'specificity':       float(home[i]),
            'string_sim':        float(str_sims[i]),
            'source':            concept.get('source', 'unknown'),
            'broader_relations': concept.get('broader_relations', []),
            '_emb':              concept_embs[i],
        })
    return sorted(results, key=lambda x: -x['score'])

def assign_concept_label(scores, fallback='Group', min_score=0.08,
                         ancestor_names=None, used_sibling_labels=None,
                         top_level_tasks=None, ancestor_embs=None,
                         sibling_label_embs=None, dup_sim=0.82):
    """
    Pick best concept label from scored results.

    Rejection combines STRUCTURAL guards (domain-agnostic, not hardcoding) with
    SEMANTIC ones (embedding cosine):
      Structural:
        - token self-repetition ("Dms Dms")
        - label is a substring of / equal to an ancestor, or vice-versa
          (kills "Dms" and "Dms Recommended Standard" sitting under ancestor "DMS")
        - all of the label's content words already appear in an ancestor label
        - exact match with an already-used sibling label
        - FOREIGN-TASK token: label contains a top-level task name that is NOT the
          current ancestor task (e.g. "Rvp 3 Targets" / "Swm Errors" under DMS).
          Task names are discovered from _group_path — data-driven, not hardcoded.
      Semantic:
        - cosine(label_emb, any ancestor_emb)  > dup_sim → parent paraphrase
        - cosine(label_emb, any sibling_emb)   > dup_sim → sibling paraphrase
    Returns (label, provenance_dict).
    """
    ancestor_set = {str(a).lower().strip() for a in (ancestor_names or [])}
    used_set     = {str(u).lower().strip() for u in (used_sibling_labels or [])}
    anc_embs     = np.asarray(ancestor_embs, dtype=float) if ancestor_embs is not None and len(ancestor_embs) else None
    sib_embs     = list(sibling_label_embs) if sibling_label_embs else []

    # Current task = the ancestor that is itself a top-level task (data-driven)
    _task_set     = {str(t).lower() for t in (top_level_tasks or [])}
    _current_task = next((str(a).lower() for a in (ancestor_names or [])
                          if str(a).lower() in _task_set), None)

    def _is_degenerate(lbl, emb=None):
        """True if label should be rejected."""
        l = lbl.strip().lower()
        # Structural 1: token self-repetition ("Dms Dms", "Swm Swm")
        toks = l.split()
        if len(toks) >= 2 and len(set(toks)) < len(toks):
            return True
        # Structural 2: substring of / equal to an ancestor (or vice-versa)
        for anc in ancestor_set:
            if l == anc or l in anc or anc in l:
                return True
        # Structural 3: all content words already present in an ancestor label
        lbl_words = set(re.findall(r'\b[a-z]{3,}\b', l)) - _STOP
        for anc in ancestor_set:
            anc_words = set(re.findall(r'\b[a-z]{3,}\b', anc)) - _STOP
            if lbl_words and lbl_words.issubset(anc_words):
                return True
        # Structural 4: already used by a sibling group
        if l in used_set:
            return True
        # Structural 5: foreign-task token (cross-task contamination)
        if _current_task and _task_set:
            for task in _task_set:
                if task != _current_task and re.search(r'\b' + re.escape(task) + r'\b', l):
                    return True
        # Semantic parent-duplication: candidate paraphrases an ancestor label
        if emb is not None and anc_embs is not None:
            if float(cosine_similarity([emb], anc_embs).max()) > dup_sim:
                return True
        # Semantic sibling-duplication: candidate paraphrases a chosen sibling label
        if emb is not None and sib_embs:
            if float(cosine_similarity([emb], np.asarray(sib_embs, dtype=float)).max()) > dup_sim:
                return True
        return False

    # Walk ranked scores; skip degenerate candidates
    chosen = None
    for s in scores:
        if s['score'] < min_score:
            break
        candidate = s['label'].strip().title()
        if not _is_degenerate(candidate, s.get('_emb')):
            chosen = s
            break

    if chosen is None:
        return fallback, {
            'node_label': fallback, 'confidence': 0.0,
            'alternatives': [], 'source_evidence': ['tfidf_fallback'],
            'embedding_sim': 0.0, 'string_sim': 0.0,
            'coverage': 0.0, 'contrast': 0.0, 'specificity': 0.0,
        }

    label = chosen['label'].strip().title()
    alts  = [s['label'] for s in scores[1:4]
             if s['label'] != chosen['label']
             and not _is_degenerate(s['label'].strip().title(), s.get('_emb'))]
    provenance = {
        'node_label':    label,
        'confidence':    round(chosen['score'], 3),
        'alternatives':  alts,
        'source_evidence': [chosen['source']],
        'embedding_sim': round(chosen['embedding_sim'], 3),
        'coverage':      round(chosen.get('coverage', 0.0), 3),
        'contrast':      round(chosen.get('contrast', 0.0), 3),
        'specificity':   round(chosen.get('specificity', 0.0), 3),
        'string_sim':    round(chosen['string_sim'], 3),
    }
    return label, provenance

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF FALLBACK CLUSTERING
# Used when concept table is unavailable or similarity is too low.
# ─────────────────────────────────────────────────────────────────────────────
def tfidf_cluster_labels(texts, max_clusters=8):
    """[GON] TF-IDF agglomerative clustering — discriminative label per cluster."""
    n = len(texts)
    if n <= 1:
        return [''] * n
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                              max_features=500, min_df=1)
        X = vec.fit_transform(texts)
        n_clust = min(max_clusters, max(2, n // 3), X.shape[0])
        dist    = cosine_distances(X).astype(float)
        np.fill_diagonal(dist, 0.0)
        labels  = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed',
                                          linkage='average').fit_predict(dist)
        terms   = vec.get_feature_names_out()
        X_dense = X.toarray()
        cnames  = {}
        for k in range(n_clust):
            mask = labels == k
            if not mask.any():
                cnames[k] = f'Group {k+1}'
                continue
            scores = X_dense[mask].mean(axis=0) - (X_dense[~mask].mean(axis=0) if (~mask).any() else 0)
            top    = [i for i in np.argsort(scores)[::-1] if len(terms[i]) > 3]
            cnames[k] = terms[top[0]].title() if top else f'Group {k+1}'
        return [cnames[int(lb)] for lb in labels]
    except Exception:
        return [''] * n

# ─────────────────────────────────────────────────────────────────────────────
# [C8] WORDNET HYPERNYM CHAIN FALLBACK [CAS][TAX]
# Castanet: "carves out a structure from WordNet that reflects the collection."
# Taxonomizer: "labeling inner nodes requires the identification of hypernyms."
# Walks IS-A chain upward from the dominant noun in cluster texts.
# Returns the highest-confidence hypernym that is not in excluded_names.
# ─────────────────────────────────────────────────────────────────────────────
def wordnet_hypernym_fallback(cluster_texts, excluded_names=None):
    """
    [C8][CAS][TAX] Walk WordNet IS-A chain upward from cluster centroid noun.
    Returns the best hypernym that is:
      - not in excluded_names (ancestors, parent label)
      - not a stop word
      - not too generic (not 'entity','object','thing','abstraction','whole')
    Falls back to None if WordNet unavailable or no valid hypernym found.
    """
    if not _WORDNET_AVAILABLE:
        return None
    excluded = {str(n).lower().strip() for n in (excluded_names or [])}
    _too_generic = {'entity', 'object', 'thing', 'abstraction', 'whole',
                    'physical entity', 'psychological feature', 'group',
                    'attribute', 'measure', 'amount', 'number', 'quantity'}

    # Extract most frequent meaningful nouns from cluster texts
    all_text = ' '.join(cluster_texts).lower()
    words = [w for w in re.findall(r'\b[a-z]{4,}\b', all_text)
             if w not in _STOP and w not in _too_generic]
    if not words:
        return None

    from collections import Counter
    freq = Counter(words)
    candidates_words = [w for w, _ in freq.most_common(8)]

    best_label = None
    best_depth = 0  # prefer specific (deeper) hypernyms over generic ones

    for word in candidates_words:
        try:
            synsets = wn.synsets(word, pos=wn.NOUN)
            if not synsets:
                continue
            ss = synsets[0]
            # Walk hypernym chain — collect all hypernyms with their depth
            paths = ss.hypernym_paths()
            for path in paths:
                for depth, hyp_ss in enumerate(reversed(path)):
                    hyp_name = hyp_ss.name().split('.')[0].replace('_', ' ')
                    if (hyp_name.lower() not in excluded
                            and hyp_name.lower() not in _too_generic
                            and hyp_name.lower() not in _STOP
                            and len(hyp_name) > 3
                            and depth > 0):           # skip the synset itself (depth=0)
                        if depth > best_depth:
                            best_depth = depth
                            best_label = hyp_name.title()
                        break                         # use deepest valid hypernym per path
        except Exception:
            continue

    return best_label if best_depth > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# [C3] DISCRIMINATIVE TF-IDF LABEL PER GROUP [GON][TAX]
# Gonçalves: N×M alignment uses text signals from descriptions, not raw codes.
# TaxoGen: "local embedding module for discriminative power at each level."
# Computes TF-IDF across ALL groups so terms unique to THIS group score high.
# ─────────────────────────────────────────────────────────────────────────────
def get_discriminative_tfidf_label(cluster_texts, all_groups_texts):
    """
    [C3][GON][TAX] Return the most discriminative 1-2 word label for cluster_texts
    relative to all_groups_texts (list of text lists from sibling groups).
    Uses TF-IDF contrast: high TF in cluster, low IDF across all groups = discriminative.
    Returns a title-cased string or None.
    """
    try:
        # Build one document per group (cluster + all siblings)
        cluster_doc = ' '.join(cluster_texts)
        sibling_docs = [' '.join(g) for g in all_groups_texts if g]
        all_docs = [cluster_doc] + sibling_docs
        if len(all_docs) < 2:
            return None
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                              max_features=300, min_df=1)
        X = vec.fit_transform(all_docs)
        terms = vec.get_feature_names_out()
        cluster_vec = X[0].toarray()[0]
        # Score = cluster TF-IDF score — mean across sibling docs
        sibling_mean = X[1:].toarray().mean(axis=0) if X.shape[0] > 1 else np.zeros(len(terms))
        contrast = cluster_vec - sibling_mean
        best_idxs = [i for i in np.argsort(contrast)[::-1]
                     if len(terms[i]) > 3 and contrast[i] > 0.01]
        if best_idxs:
            return terms[best_idxs[0]].title()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NODE MANIPULATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def nmap(nodes):    return {int(n['id']): n for n in nodes}
def next_id(nodes): return max([int(n['id']) for n in nodes] or [0]) + 1

def add_child(nodes, parent, child):
    m = nmap(nodes); p = m.get(int(parent))
    if not p: return
    rel = list(p.get('related', []))
    if int(child) not in rel: rel.append(int(child))
    p['related'] = rel

def remove_child(nodes, parent, child):
    m = nmap(nodes); p = m.get(int(parent))
    if p: p['related'] = [x for x in p.get('related', []) if int(x) != int(child)]

def make_agg(id, name, related=None, op='concat', dtype='determine', desc='',
             shown=True, relation_type='belongs_to', provenance=None):
    node = {
        'id':      int(id),
        'name':    str(name),
        'related': [int(x) for x in (related or [])],
        'type':    'aggregation',
        'info':    {'operation': op, 'usedAttributes': [], 'formula': '', 'exec': '',
                    'relation_type':  relation_type,
                    'relation_label': RELATION_TYPES.get(relation_type, 'belongs to')},
        'isShown': bool(shown),
        'desc':    desc or '',
        'dtype':   dtype,
        'recover': True,
    }
    if provenance:
        node['concept_provenance'] = provenance
    return node

def get_node(nodes, id):    return nmap(nodes).get(int(id))
def update_node(nodes, id, **upd):
    for n in nodes:
        if int(n['id']) == int(id): n.update(upd)
    return nodes

def parents(nodes, child):
    return [int(n['id']) for n in nodes if int(child) in [int(x) for x in n.get('related', [])]]

def ancestor_names(nodes, nid):
    """
    [FIX1][HIE] Walk up the tree from nid collecting all ancestor node names.
    HiExpan (Shen et al., KDD 2018) Section 4.3 — Conflict Resolution:
    "avoid assigning a label already present in the path from root to the node."
    Used by hiexpan_depth_expansion_semantic to pass ancestor context to
    assign_concept_label, preventing repeated labels across hierarchy levels.
    """
    m = nmap(nodes)
    result, cur, visited = [], int(nid), set()
    while cur not in visited:
        visited.add(cur)
        pars = parents(nodes, cur)
        if not pars:
            break
        p  = pars[0]
        pn = m.get(p)
        if pn and pn.get('name') and pn.get('type') != 'root':
            result.append(str(pn['name']))
        cur = p
    return result

def descendants(nodes, id):
    m = nmap(nodes); seen = []
    def rec(nid):
        n = m.get(int(nid))
        if not n: return
        for c in n.get('related', []):
            c = int(c)
            if c in seen: continue
            seen.append(c); rec(c)
    rec(id); return seen

def leaf_ids(nodes, id):
    m = nmap(nodes); out = []
    def rec(nid):
        n = m.get(int(nid))
        if not n: return
        if n.get('type') == 'attribute': out.append(int(nid)); return
        for c in n.get('related', []): rec(int(c))
    rec(id); return list(dict.fromkeys(out))

def move_node(nodes, child, new_parent):
    if int(child) == 0 or int(new_parent) == int(child) or int(new_parent) in descendants(nodes, child):
        return nodes
    for p in parents(nodes, child): remove_child(nodes, p, child)
    add_child(nodes, new_parent, child); return nodes

def delete_agg(nodes, id, reattach=True):
    if int(id) == 0: return nodes
    n = get_node(nodes, id)
    if not n or n.get('type') == 'attribute': return nodes
    ps = parents(nodes, id); children = list(n.get('related', []))
    for p in ps:
        remove_child(nodes, p, id)
        if reattach:
            for c in children: add_child(nodes, p, c)
    return [x for x in nodes if int(x['id']) != int(id)]

def build_parent_map(nodes):
    pm = {}
    for n in nodes:
        for c in n.get('related', []):
            cid = int(c)
            if cid not in pm: pm[cid] = int(n['id'])
    return pm

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _eval_cluster_assignments(nodes: list, can: pd.DataFrame) -> list:
    """Map each canonical row to the id of its depth-1 aggregation ancestor."""
    pm = build_parent_map(nodes)
    def depth1(nid: int) -> int:
        while pm.get(nid, -1) not in (-1, 0):
            nid = pm[nid]
        return nid
    lid_to_nid = {n['metadata']['leaf_id']: int(n['id'])
                  for n in nodes if n.get('type') == 'attribute' and 'metadata' in n}
    return [depth1(lid_to_nid[lid]) if lid in lid_to_nid else -1
            for lid in can['_leaf_id']]

def _purity_score(y_true, y_pred) -> float:
    from collections import Counter
    clusters: dict = {}
    for t, p in zip(y_true, y_pred):
        clusters.setdefault(p, []).append(t)
    correct = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct / max(len(y_true), 1)

def _structural_stats(nodes: list) -> dict:
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

def path_rows(nodes):
    m = nmap(nodes); rows = []
    def rec(nid, path, depth):
        n = m.get(int(nid))
        if not n: return
        label = n.get('name', str(nid))
        full  = f'{path} / {label}' if path else label
        lc    = len(leaf_ids(nodes, nid))
        rows.append({'id': int(nid), 'name': label, 'path': full, 'depth': depth,
                     'type': n.get('type', ''), 'leaf_count': lc,
                     'relation': n.get('info', {}).get('relation_label', ''),
                     'choice': f'{full} [{n.get("type","")}, {lc} vars]'})
        for c in n.get('related', []): rec(int(c), full, depth + 1)
    rec(0, '', 0); return rows

def attr_opts(nodes):
    return [{'label': f'{n["name"]} (id={n["id"]})', 'id': int(n['id']), 'node': n}
            for n in nodes if n.get('type') == 'attribute']

def agg_opts(nodes, root=True):
    types = ['aggregation', 'root'] if root else ['aggregation']
    return [{'label': f'{n["name"]} (id={n["id"]})', 'id': int(n['id']), 'node': n}
            for n in nodes if n.get('type') in types]

def _centroid(embs):
    if embs is None or len(embs) == 0: return None
    c = np.mean(embs, axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c

# ─────────────────────────────────────────────────────────────────────────────
# [F4] FACET-GUIDED SUB-SPLITTING HELPERS [CAS][HIE]
#
# Castanet (Stoica & Hearst, 2007): "a single hierarchy conflates dimensions
# and produces hard-to-label, counter-intuitive sub-hierarchies."
# These helpers insert Statistic and Condition tiers below each concept node,
# using the _facet_stat and _facet_cond columns pre-computed by
# precompute_stat_cond_facets(). Only splits when ≥2 valid sub-groups exist
# with ≥2 variables each — consistent with HiExpan's coherence requirement.
# No hardcoding: the column values (Mean/SD/Median, 0/4/12/...) come from data.
# ─────────────────────────────────────────────────────────────────────────────
_MIN_FACET_GROUP = 2  # minimum variables per facet sub-group

def _do_facet_subsplit(sub_can, parent_id, current_path,
                       nodes, leaf_to_id, ensure_path_fn):
    """
    [F4][CAS] Facet sub-split by _facet_cond (numeric condition) only.
    The statistic tier is no longer inserted here — it came from a hardcoded
    statistic vocabulary and is now produced data-drivenly by _nest_by_measure().
    Kept defensive: if a legacy _facet_stat column is present it is still honoured,
    but precompute_stat_cond_facets() no longer produces one.
    """
    # A facet tier that merely repeats the parent concept label (e.g. a "Total"
    # statistic under a "Total" concept) is redundant — skip it.
    _parent_lbl = str(current_path[-1]).lower() if current_path else ''
    def _dups_parent(name):
        nl = str(name).lower().strip()
        return bool(nl) and (nl == _parent_lbl or nl in _parent_lbl or _parent_lbl in nl)

    if '_facet_stat' in sub_can.columns:
        stat_groups = {}
        for sv in sub_can['_facet_stat'].fillna('').unique():
            if str(sv) in ('', 'nan') or _dups_parent(sv):
                continue
            sg = sub_can[sub_can['_facet_stat'] == sv]
            if len(sg) >= _MIN_FACET_GROUP:
                stat_groups[sv] = sg

        if len(stat_groups) >= 2:
            # Identify leftover rows not in any valid stat group
            grouped_idx = pd.concat(stat_groups.values()).index if stat_groups else pd.Index([])
            leftover    = sub_can[~sub_can.index.isin(grouped_idx)]

            for sv, sg in stat_groups.items():
                stat_pid = ensure_path_fn(current_path + [str(sv)],
                                          relation='is_statistic_of')
                _do_cond_subsplit(sg, stat_pid, current_path + [str(sv)],
                                  nodes, leaf_to_id, ensure_path_fn)

            # Leftover variables (no valid stat value) go directly under parent
            for _, row in leftover.iterrows():
                add_child(nodes, parent_id, leaf_to_id[row['_leaf_id']])
            return

    # No valid stat split — try cond split at this level directly
    _do_cond_subsplit(sub_can, parent_id, current_path,
                      nodes, leaf_to_id, ensure_path_fn)


def _do_cond_subsplit(sub_can, parent_id, current_path,
                      nodes, leaf_to_id, ensure_path_fn):
    """
    [F4][CAS] Split by _facet_cond (numeric condition: delay/session/timepoint).
    Castanet (Stoica & Hearst, NAACL 2007) treats a condition as a valid facet when
    it has ≥2 distinct values, but HiExpan (Shen et al., KDD 2018) sibling coherence
    requires each resulting group to hold ≥2 variables — a node with a single child
    is not a coherent group.

    [FIX8] Earlier code (the reverted [FIX3] rule) allowed groups of size 1 whenever
    ≥3 distinct condition values existed, to expose 0/4/12-second delay tiers. On
    AI-MIND this produced one aggregation node per delay value, each wrapping a SINGLE
    variable (e.g. `Standard Deviation > 0 > DMSL0SD`): 53.7% singleton nodes and
    meaningless bare-digit labels. We now require ≥2 variables per condition group
    unconditionally; variables whose condition value is unique fall through and attach
    directly to the statistic node, keeping siblings together instead of fragmenting
    them. This is both more faithful to HiExpan and removes the over-split that the
    granularity-tolerant set-overlap metric flagged.
    """
    if '_facet_cond' in sub_can.columns:
        # Count all distinct non-empty condition values in this sub-group
        all_cond_vals = [cv for cv in sub_can['_facet_cond'].fillna('').unique()
                         if str(cv) not in ('', 'nan')]
        # [FIX8] Require ≥2 variables per condition group (HiExpan sibling coherence).
        min_size = _MIN_FACET_GROUP

        cond_groups = {}
        for cv in all_cond_vals:
            cg = sub_can[sub_can['_facet_cond'] == cv]
            if len(cg) >= min_size:
                cond_groups[cv] = cg

        if len(cond_groups) >= 2:
            grouped_idx = pd.concat(cond_groups.values()).index if cond_groups else pd.Index([])
            leftover    = sub_can[~sub_can.index.isin(grouped_idx)]

            for cv, cg in cond_groups.items():
                cond_pid = ensure_path_fn(current_path + [str(cv)],
                                          relation='has_condition')
                for _, row in cg.iterrows():
                    add_child(nodes, cond_pid, leaf_to_id[row['_leaf_id']])

            for _, row in leftover.iterrows():
                add_child(nodes, parent_id, leaf_to_id[row['_leaf_id']])
            return

    # No facet split possible — attach directly
    for _, row in sub_can.iterrows():
        add_child(nodes, parent_id, leaf_to_id[row['_leaf_id']])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HIERARCHY BUILDER [GON][TAX][HIE][CAS]
#
# Algorithm:
#   1. Create leaf nodes (all variables)
#   2. Group by top-level _group_path (task/domain — structural backbone)
#   3. For each group: embed variables → agglomerative cluster → for each cluster
#      compute centroid → score N×M against concept table → assign best label
#   4. [F4] For each concept cluster: facet sub-split by Statistic → Condition
#   5. Store concept assignment back on each variable in can
# ─────────────────────────────────────────────────────────────────────────────
def _noun_phrases(text, max_words=4):
    """
    Grammatical noun phrases via NLTK POS tagging (used when USE_NOUN_PHRASES=True).
    Returns [] if NLTK / the tagger is unavailable, so the caller falls back to
    n-grams. Phrases are contiguous runs of adjectives/nouns up to max_words long.
    """
    try:
        import nltk
        for _pkg in ('averaged_perceptron_tagger', 'punkt'):
            try:
                nltk.data.find(f'taggers/{_pkg}' if 'tagger' in _pkg else f'tokenizers/{_pkg}')
            except LookupError:
                nltk.download(_pkg, quiet=True)
        toks = nltk.word_tokenize(str(text))
        tags = nltk.pos_tag(toks)
    except Exception:
        return []
    phrases, cur = [], []
    for w, t in tags:
        if t.startswith('NN') or t.startswith('JJ'):
            cur.append(w)
            if len(cur) > max_words:
                cur = cur[-max_words:]
        else:
            if len(cur) >= 1:
                phrases.append(' '.join(cur))
            cur = []
    if cur:
        phrases.append(' '.join(cur))
    return [p for p in phrases if len(p) >= 3]


def _keybert_label(member_texts, cluster_centroid, embedder, ancestor_words=None,
                   corpus_centroid=None, used_labels=None, max_words=4,
                   gen_weight=0.0, diversity=KEYBERT_DIVERSITY, cap=500):
    """
    KeyBERT-style extractive labeller. Extract candidate phrases from the cluster's
    DESCRIPTIONS, embed them, and pick by:
        score = (1 − diversity)·cos(phrase, cluster_centroid)
              −      diversity ·cos(phrase, mean candidate phrase)   # MMR redundancy
    With diversity=0 this is plain cosine-to-centroid (argmax relevance). When
    USE_CTFIDF=True the relevance is modulated by corpus IDF so boilerplate (low IDF)
    is suppressed. Candidates come from noun phrases (USE_NOUN_PHRASES=True) or
    n-grams. Extractive — never hallucinates a label. Returns a title-cased string.
    """
    ancestor_words = ancestor_words or set()
    used = {str(u).lower() for u in (used_labels or [])}
    cand = set()
    for t in member_texts:
        raw = re.sub(r'\([^)]*\)', ' ', str(t))            # drop parentheticals
        nps = _noun_phrases(raw, max_words) if USE_NOUN_PHRASES else []
        if nps:
            for p in nps:
                toks = [w for w in p.lower().split()
                        if w not in _STOP and w not in ancestor_words]
                if toks:
                    cand.add(' '.join(toks))
        else:
            toks = [w for w in re.findall(r'[a-z][a-z\-]+', raw.lower())
                    if w not in _STOP and w not in ancestor_words]
            for nlen in range(1, max_words + 1):
                for i in range(len(toks) - nlen + 1):
                    cand.add(' '.join(toks[i:i + nlen]))
    # Junk filter: drop used labels, pure-number phrases, immediately-repeated words.
    cand = [c for c in cand if len(c) >= 4 and c.lower() not in used
            and not c.replace(' ', '').isdigit()
            and not re.search(r'\b(\w+)\s+\1\b', c.lower())]
    if not cand:
        return ''
    cand = cand[:cap]
    embs = np.asarray(embedder.encode(cand), dtype=float)
    sims = cosine_similarity([cluster_centroid], embs)[0]          # relevance
    if USE_CTFIDF and _CORPUS_IDF:
        mx = max(_CORPUS_IDF.values()) or 1.0
        idf = np.array([min(1.0, _CORPUS_IDF.get(c.lower(), mx) / mx) for c in cand])
        sims = sims * (0.5 + 0.5 * idf)
    if gen_weight and corpus_centroid is not None:
        sims = sims - gen_weight * cosine_similarity([corpus_centroid], embs)[0]
    if diversity > 0 and len(embs) > 1:                            # MMR penalty
        generic = cosine_similarity(embs.mean(axis=0, keepdims=True), embs)[0]
        score = (1.0 - diversity) * sims - diversity * generic
    else:
        score = sims
    return cand[int(np.argmax(score))].title()


def _keybert_candidates(member_texts, ancestor_words=None, used_labels=None,
                        max_words=3, cap=500):
    """
    Extract the KeyBERT CANDIDATE phrases from a cluster's member descriptions —
    the same generation logic as _keybert_label but returns the full candidate list
    (un-ranked) so the caller can score them with the title-seeded scorer. Phrases
    are noun phrases (USE_NOUN_PHRASES=True) or n-grams, with ancestor/task words,
    pure numbers, used labels and immediate repeats filtered out.
    """
    ancestor_words = ancestor_words or set()
    used = {str(u).lower() for u in (used_labels or [])}
    cand = set()
    for t in member_texts:
        raw = re.sub(r'\([^)]*\)', ' ', str(t))
        nps = _noun_phrases(raw, max_words) if USE_NOUN_PHRASES else []
        if nps:
            for p in nps:
                toks = [w for w in p.lower().split()
                        if w not in _STOP and w not in ancestor_words]
                if toks:
                    cand.add(' '.join(toks))
        else:
            toks = [w for w in re.findall(r'[a-z][a-z\-]+', raw.lower())
                    if w not in _STOP and w not in ancestor_words]
            for nlen in range(1, max_words + 1):
                for i in range(len(toks) - nlen + 1):
                    cand.add(' '.join(toks[i:i + nlen]))
    cand = [c for c in cand if len(c) >= 4 and c.lower() not in used
            and not c.replace(' ', '').isdigit()
            and not re.search(r'\b(\w+)\s+\1\b', c.lower())]
    return cand[:cap]


def _concept_title(text):
    """
    Extract the human-written concept TITLE from a metadata description.
    Data-dictionary descriptions read `Title: long definition sentence`. The title
    is the colon-segment immediately BEFORE the longest segment (the definition).
    Structural, no vocabulary — works whether the title is segment 0
    ("DMS Correct Latency SD: The standard deviation...") or later
    ("KEY: SWM Between errors: The number of times..."). Returns the title only.
    """
    t = str(text).strip()
    if not t:
        return ''
    segs = t.split(':')
    if len(segs) < 2:
        title = t
    else:
        lens  = [len(s.strip()) for s in segs]
        def_i = max(range(1, len(segs)), key=lambda i: lens[i])   # longest = definition
        title = segs[def_i - 1].strip() or t
    # A genuine concept title is short. If what we extracted is a full SENTENCE
    # (e.g. MOT has no "Title: def" structure — just prose), it is not a title;
    # return '' so the caller falls back to the embedding concept scorer instead of
    # labelling from a sentence. Length-based, no vocabulary.
    if len(re.findall(r'[A-Za-z]+', title)) > 9:
        return ''
    return title


def _title_cluster_label(member_titles, sibling_title_lists, ancestor_words=None,
                         max_words=4, used_labels=None):
    """
    Label a cluster from the concept TITLES its members share, chosen CONTRASTIVELY
    against sibling clusters (tree-based local-IDF). Titles are concept-dense (no
    boilerplate definition text), so this returns the genuine shared concept —
    "Correct Latency", "Standard Deviation" — never "Calculated Assessed Trials".
    Strips ancestor/task tokens and avoids repeating a parent or a used sibling.
    Returns a title-cased label or ''.
    """
    ancestor_words = {w.lower() for w in (ancestor_words or [])}
    used_labels    = {str(u).lower() for u in (used_labels or [])}

    def _phrases(title):
        t = re.sub(r'\([^)]*\)', ' ', title.lower())      # drop parenthetical conditions
        toks = [w for w in re.findall(r'[a-z][a-z\-]{1,}', t)
                if w not in _STOP and w not in ancestor_words]
        out = set()
        for nlen in range(1, max_words + 1):
            for i in range(len(toks) - nlen + 1):
                out.add(' '.join(toks[i:i + nlen]))
        return out

    M = len(member_titles)
    if M == 0:
        return ''
    member_df = defaultdict(int)
    for ph_set in (_phrases(t) for t in member_titles):
        for ph in ph_set:
            member_df[ph] += 1
    sib_flat = [t for lst in (sibling_title_lists or []) for t in lst]
    S = max(1, len(sib_flat))
    sib_df = defaultdict(int)
    for t in sib_flat:
        for ph in _phrases(t):
            sib_df[ph] += 1

    best, best_score = '', -1.0
    for ph, mdf in member_df.items():
        if mdf < 2:                              # must be shared by ≥2 members
            continue
        words = ph.split()
        if all(w in ancestor_words for w in words):   # don't repeat the parent
            continue
        if ph in used_labels:                          # don't repeat a sibling
            continue
        score = (mdf / M - sib_df.get(ph, 0) / S) * (1.0 + 0.25 * (len(words) - 1))
        if score > best_score:
            best_score, best = score, ph
    return best.title() if best else ''


def _raw_title(text):
    """Title segment, keeping parentheticals (the error TYPE lives in them)."""
    segs = str(text).split(':')
    if len(segs) < 2:
        return str(text).strip()
    lens = [len(s.strip()) for s in segs]
    di = max(range(1, len(segs)), key=lambda i: lens[i])
    return segs[di - 1].strip()


def _label_from_own_title(title, ancestor_words, max_words=4):
    """[Fix5] Label a singleton variable from its OWN title (minus ancestor/task
    words and parentheticals). Returns '' for sentence-like / empty titles."""
    t = re.sub(r'\([^)]*\)', ' ', str(title).lower())
    toks = [w for w in re.findall(r'[a-z][a-z\-]+', t)
            if w not in _STOP and w not in ancestor_words]
    if not toks or len(toks) > 7:          # >7 words ⇒ prose, not a concept title
        return ''
    return ' '.join(toks[:max_words]).title()


def _strip_leading_prose(label):
    """Drop a leading word that is a verb but NOT a noun in WordNet (e.g. 'Include
    Shapes' → 'Shapes', from a prose description) — data-driven, no word list. Keeps
    qualifiers like 'Correct' only when applied where appropriate (sub-split labels)."""
    if not _WORDNET_AVAILABLE:
        return label
    words = label.split()
    while len(words) > 1:
        w = words[0].lower()
        try:
            if wn.synsets(w, pos=wn.VERB) and not wn.synsets(w, pos=wn.NOUN):
                words = words[1:]
            else:
                break
        except Exception:
            break
    return ' '.join(words)


def _subsplit_concept_by_title(nodes):
    """
    [Fix2] Split a concept node's leaves by a distinctive TITLE descriptor when they
    fall into ≥2 groups of ≥2 — e.g. DMS 'Error' → {Incorrect Colour, Incorrect
    Pattern, Distractor} (the type is in the title parenthetical). Greedy prefers
    LONGER descriptors so "incorrect colour"/"incorrect pattern" win over the bare
    "incorrect". The ≥2-groups-of-≥2 gate is what stops a delay over-split: delay
    variants form only ONE group ("second delay") so they are never split out.
    No hardcoded vocabulary.
    """
    pm = build_parent_map(nodes)
    def _anc_words(nid):
        w, x = set(), nid
        m = nmap(nodes)
        while True:
            nd = m.get(x)
            if nd and nd.get('type') != 'root':
                w |= set(re.findall(r'[a-z]{3,}', str(nd.get('name', '')).lower()))
            if x not in pm:
                break
            x = pm[x]
        return w
    for node in [n for n in nodes if n.get('type') == 'aggregation']:
        nid = int(node['id'])
        m = nmap(nodes)
        leaf_children = [int(c) for c in node.get('related', [])
                         if m.get(int(c), {}).get('type') == 'attribute']
        if len(leaf_children) < 4:
            continue
        aw = _anc_words(nid)
        # Tokens present in (nearly) ALL leaves are parent-level, not sub-categories
        # — e.g. "sd" under a Standard Deviation node. Excluding them stops the
        # delay over-split (without them, delay variants form only one group).
        nL = len(leaf_children)
        tok_df = defaultdict(int)
        for cid in leaf_children:
            ln = m[cid]
            title = _raw_title(ln.get('semantic_desc', ln.get('desc', '')))
            for w in set(re.findall(r'[a-z][a-z\-]+', title.lower())):
                if w not in _STOP and w not in aw:
                    tok_df[w] += 1
        common = {w for w, c in tok_df.items() if c > 0.7 * nL}
        p2l = defaultdict(set)
        for cid in leaf_children:
            ln = m[cid]
            title = _raw_title(ln.get('semantic_desc', ln.get('desc', '')))
            toks = [w for w in re.findall(r'[a-z][a-z\-]+', title.lower())
                    if w not in _STOP and w not in aw and w not in common]
            phs = set()
            for nl in (3, 2, 1):
                for i in range(len(toks) - nl + 1):
                    phs.add(' '.join(toks[i:i + nl]))
            for p in phs:
                p2l[p].add(cid)
        covered, groups = set(), []
        for p, cids in sorted(p2l.items(), key=lambda kv: (-len(kv[0].split()), -len(kv[1]))):
            avail = cids - covered
            if len(avail) >= 2:
                groups.append((p, avail)); covered |= avail
        if len(groups) < 2:
            continue
        for p, cids in groups:
            sub_lbl = _strip_leading_prose(p.title()) or p.title()
            snid = next_id(nodes)
            nodes.append(make_agg(snid, sub_lbl, desc=f'Sub-group: {sub_lbl}',
                                  relation_type='belongs_to'))
            add_child(nodes, nid, snid)
            for cid in cids:
                remove_child(nodes, nid, cid)
                add_child(nodes, snid, cid)
    return nodes


def _cluster_and_label(tdf, path_prefix, nodes, leaf_to_id, embedder,
                        concept_table, concept_embs, ensure_path_fn,
                        n_clusters_max, can, ref_centroids=None,
                        corpus_centroid=None):
    """
    Cluster variables in tdf and assign concept labels from concept_table.
    Modifies nodes in-place. Updates can['_concept_label'] for each variable.

    Semantic label selection:
    [GON][CAS] score_concepts_for_cluster ranks candidates by embedding coverage,
               sibling contrast, and dataset-wide specificity (semantic IDF) using
               the member / sibling / group (ref_centroids) embeddings passed in.
    [HIE]      assign_concept_label rejects labels that paraphrase an ancestor or a
               chosen sibling (semantic dedup via embeddings, not word lists).
    [HIE]      Singleton clusters (n=1) attach directly to parent — no group node.
    """
    sem_col = '_semantic_text' if '_semantic_text' in tdf.columns else '_text'
    texts = tdf[sem_col].fillna('').tolist()
    # Concept TITLES (pre-definition) — clean label vocabulary, never boilerplate.
    titles = [_concept_title(t) for t in texts]
    n     = len(tdf)

    if n == 0:
        return

    # Ancestor names + their embeddings for semantic parent-duplication filter [HIE]
    ancestor_names = list(path_prefix)
    ancestor_embs  = (embedder.encode(ancestor_names)
                      if ancestor_names else None)

    # [ChangeC] Discover top-level task tokens from the full dataset (data-driven)
    _top_level_tasks: set = set()
    if '_group_path' in can.columns:
        for _gp in can['_group_path'].dropna().astype(str):
            _f = _gp.split(' > ')[0].strip()
            if _f and _f.lower() not in ('ungrouped', 'nan', ''):
                _top_level_tasks.add(_f.lower())

    _aw_base = set(re.findall(r'[a-z]{3,}', ' '.join(ancestor_names).lower())) | _top_level_tasks

    if n < 3 or concept_embs is None or len(concept_table) == 0:
        # Too few variables to cluster — label each from its own title [Fix5], or
        # KeyBERT over its description when no title exists. ensure_path merges it
        # into an existing concept of the same name.
        pid = ensure_path_fn(path_prefix)
        _small = embedder.encode(texts) if texts else None
        for i, (_, row) in enumerate(tdf.iterrows()):
            lbl = _label_from_own_title(titles[i], _aw_base)
            if not lbl and _small is not None:
                lbl = _keybert_label([texts[i]], _small[i], embedder,
                                     ancestor_words=_aw_base, used_labels=set(),
                                     max_words=2, gen_weight=0.3,
                                     diversity=KEYBERT_DIVERSITY)
            tgt = ensure_path_fn(path_prefix + [lbl]) if lbl and lbl.lower() not in \
                  {a.lower() for a in ancestor_names} else pid
            add_child(nodes, tgt, leaf_to_id[row['_leaf_id']])
        return

    # Embed variables
    var_embs = embedder.encode(texts)
    # Centroid of THIS task/subgroup — reference for the scorer's `home` signal.
    own_group_centroid = _centroid(var_embs)

    # Choose number of clusters adaptively
    n_clust = min(n_clusters_max, max(2, n // 3), n)

    # Agglomerative clustering on variable embeddings [TAX][GON]
    # [FIX7][GON][TAX] Code-family cohesion bias:
    # Variables sharing the same _code_family (e.g. DMSL, SWMBE) are structurally
    # related by the instrument's own naming convention.  Reduce their pairwise
    # cosine distance by a factor of 0.80 so the clusterer prefers to keep them
    # together.  Rationale — Gonçalves ESWC 2019: structural prefix affinity;
    # Taxonomizer IEEE TVCG 2019: compound labels align with code morphology.
    # Factor 0.80 is a cohesion weight, not a fixed threshold — it is applied
    # multiplicatively so the relative ordering of similarities is preserved.
    try:
        dist = cosine_distances(var_embs).astype(float)
        np.fill_diagonal(dist, 0.0)

        # Apply code-family cohesion if _code_family is available
        if '_code_family' in tdf.columns:
            families = tdf['_code_family'].fillna('').astype(str).tolist()
            _COHESION_FACTOR = 0.80   # same-family pairs: distance × 0.80 (pulled together)
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if families[ii] and families[ii] == families[jj]:
                        dist[ii, jj] *= _COHESION_FACTOR
                        dist[jj, ii] *= _COHESION_FACTOR

        c_lbls = AgglomerativeClustering(n_clusters=n_clust, metric='precomputed',
                                          linkage='average').fit_predict(dist)
    except Exception:
        c_lbls = np.zeros(n, dtype=int)
        n_clust = 1

    rows_list = list(tdf.iterrows())

    # [C3] Pre-compute all cluster text groups for discriminative TF-IDF [GON][TAX]
    # and each cluster's centroid (used as sibling-contrast references [CAS]).
    all_cluster_texts  = []
    all_cluster_titles = []
    all_centroids      = []
    for k in range(n_clust):
        mask = c_lbls == k
        cluster_idxs = [i for i, m in enumerate(mask) if m]
        all_cluster_texts.append([texts[i] for i in cluster_idxs] if cluster_idxs else [])
        all_cluster_titles.append([titles[i] for i in cluster_idxs] if cluster_idxs else [])
        all_centroids.append(_centroid(var_embs[mask]) if cluster_idxs else None)

    # Track used sibling labels (string) and their embeddings (semantic dedup) [TAX][GON]
    used_sibling_labels = set()
    sibling_label_embs  = []

    parent_pid = ensure_path_fn(path_prefix)  # get parent node id upfront

    for k in range(n_clust):
        mask         = c_lbls == k
        cluster_idxs = [i for i, m in enumerate(mask) if m]
        if not cluster_idxs:
            continue

        cluster_texts_k = [texts[i] for i in cluster_idxs]
        cluster_emb     = _centroid(var_embs[mask])

        # [Fix5] Singleton: label it from its OWN title and attach under that concept
        # (ensure_path merges it into an existing same-named concept if one exists),
        # instead of dumping it unclassified under the task.
        if len(cluster_idxs) == 1:
            _, row = rows_list[cluster_idxs[0]]
            lbl = _label_from_own_title(titles[cluster_idxs[0]], _aw_base)
            src = 'singleton_title'
            if not lbl and cluster_emb is not None:
                lbl = _keybert_label([cluster_texts_k[0]], cluster_emb, embedder,
                                     ancestor_words=_aw_base,
                                     used_labels=used_sibling_labels,
                                     max_words=2, gen_weight=0.3,
                                     diversity=KEYBERT_DIVERSITY)
                src = 'singleton_keybert'
            if lbl and lbl.lower() not in {a.lower() for a in ancestor_names}:
                tgt = ensure_path_fn(path_prefix + [lbl], relation='belongs_to')
                can.at[row.name, '_concept_label'] = lbl
            else:
                tgt = parent_pid
                can.at[row.name, '_concept_label'] = path_prefix[-1] if path_prefix else 'root'
            add_child(nodes, tgt, leaf_to_id[row['_leaf_id']])
            can.at[row.name, '_concept_score']  = 0.0
            can.at[row.name, '_concept_source'] = src
            continue

        if cluster_emb is not None:
            # Sibling centroids = every OTHER cluster in this parent (contrast ref) [CAS]
            sibling_centroids = [all_centroids[j] for j in range(n_clust)
                                 if j != k and all_centroids[j] is not None]
            scores = score_concepts_for_cluster(
                cluster_emb, concept_embs, concept_table, cluster_texts_k,
                n_total_vars=len(can),
                member_embs=var_embs[mask],
                sibling_centroids=np.array(sibling_centroids) if sibling_centroids else None,
                ref_centroids=ref_centroids,          # all top-level task centroids
                corpus_centroid=corpus_centroid,
                own_group_centroid=own_group_centroid,  # current task → home signal
            )
        else:
            scores = []

        # ── TITLE-SEEDED LABEL SELECTION (Guided KeyBERT) ─────────────────────
        # The label is FORMED FROM THE DESCRIPTIONS: candidates are KeyBERT phrases
        # extracted from the cluster's member descriptions (+ scored concept-table
        # entries). The pre-colon TITLE does NOT override — it is a ranking SEED:
        #   score = α·cos(cand, cluster centroid)   # description fit
        #         + β·cos(cand, title embedding)     # title INFLUENCE (LABEL_W_TITLE)
        #         + γ·contrast(vs siblings)
        #         + δ·external grounding
        # So the displayed label is always a description-derived phrase, pulled toward
        # the human-canonical title phrasing. Set LABEL_W_TITLE=0 for a pure-description
        # ablation. The title phrase is also added as ONE candidate so a clean title can
        # still win on merit (it is usually present verbatim in the descriptions anyway).
        ancestor_words = set(re.findall(r'[a-z]{3,}',
                                        ' '.join(ancestor_names).lower())) | _top_level_tasks
        member_titles_k     = [titles[i] for i in cluster_idxs]
        sibling_title_lists = [all_cluster_titles[j] for j in range(n_clust) if j != k]
        sibling_texts       = [all_cluster_texts[j] for j in range(n_clust) if j != k]

        # Pre-colon title → used only as the SEED ANCHOR (and one candidate), never a
        # direct override.
        title_label = _title_cluster_label(member_titles_k, sibling_title_lists,
                                           ancestor_words=ancestor_words,
                                           used_labels=used_sibling_labels)
        title_emb = (embedder.encode([title_label])[0]
                     if title_label else None)

        # Candidate phrases drawn ONLY from the cluster's DESCRIPTIONS (KeyBERT) plus
        # the pre-colon title. External ontology sources (Cognitive Atlas / Wikidata /
        # WordNet / PubMed) are deliberately NOT candidates — per design they inform the
        # embedding space / semantic understanding only, and must never name a node.
        kb_cands = _keybert_candidates(cluster_texts_k, ancestor_words=ancestor_words,
                                       used_labels=used_sibling_labels, max_words=3)
        pool_src = [(c, 'keybert') for c in kb_cands]
        if title_label:
            pool_src.append((title_label, 'description_title'))
        # Dedup; title's source tag takes priority over keybert when the phrase matches.
        seen_pool = {}
        for lbl, src in pool_src:
            key = lbl.lower()
            if key not in seen_pool or src == 'description_title':
                seen_pool[key] = (lbl, src)
        pool      = [v[0] for v in seen_pool.values()]
        pool_srcs = [v[1] for v in seen_pool.values()]

        keybert_label = kb_cands[0] if kb_cands else ''  # for fallback only

        candidate_scores = []
        if pool and cluster_emb is not None:
            cand_embs = np.asarray(embedder.encode(pool), dtype=float)
            relevance = cosine_similarity([cluster_emb], cand_embs)[0]
            if sibling_centroids:
                sib_sim  = cosine_similarity(cand_embs,
                                             np.asarray(sibling_centroids, dtype=float)).max(axis=1)
                contrast = np.clip(relevance - sib_sim, 0.0, 1.0)
            else:
                contrast = np.zeros(len(pool))
            # Title SEED: cosine of each description-derived candidate to the title.
            if title_emb is not None:
                title_sim = cosine_similarity(cand_embs, [title_emb])[:, 0]
            else:
                title_sim = np.zeros(len(pool))
            for i, cand in enumerate(pool):
                hyb = (LABEL_W_RELEVANCE * float(relevance[i])
                       + LABEL_W_TITLE    * float(title_sim[i])
                       + LABEL_W_CONTRAST * float(contrast[i]))
                candidate_scores.append({
                    'label':             cand,
                    'score':             hyb,
                    'embedding_sim':     float(relevance[i]),
                    'coverage':          float(relevance[i]),
                    'contrast':          float(contrast[i]),
                    'specificity':       0.0,
                    'string_sim':        float(title_sim[i]),  # title seed alignment
                    'source':            pool_srcs[i],
                    'broader_relations': [],
                    '_emb':              cand_embs[i],
                })
            candidate_scores.sort(key=lambda x: -x['score'])

        fallback_label = (title_label
                          or keybert_label
                          or get_discriminative_tfidf_label(cluster_texts_k, sibling_texts)
                          or f'Group {k+1}')

        label, provenance = assign_concept_label(
            candidate_scores,
            fallback=fallback_label,
            min_score=0.0,
            ancestor_names=ancestor_names,
            used_sibling_labels=used_sibling_labels,
            top_level_tasks=_top_level_tasks,
            ancestor_embs=ancestor_embs,
            sibling_label_embs=sibling_label_embs,
        )

        # Skip the node only when there is truly NO concept name (empty title, no
        # scored candidate → a bare "Group k"). Title labels are trusted and kept.
        if (not title_label) and (not candidate_scores) and label.startswith('Group '):
            for ci in cluster_idxs:
                _, row = rows_list[ci]
                add_child(nodes, parent_pid, leaf_to_id[row['_leaf_id']])
                can.at[row.name, '_concept_label']  = path_prefix[-1] if path_prefix else 'root'
                can.at[row.name, '_concept_score']  = 0.0
                can.at[row.name, '_concept_source'] = 'weak_label_direct'
            continue

        # WordNet hypernym — ONLY when there is no title concept name.
        if (not title_label) and (label == fallback_label
                                  or label.lower() in {a.lower() for a in ancestor_names}):
            wn_label = wordnet_hypernym_fallback(cluster_texts_k, excluded_names=ancestor_names)
            if wn_label:
                label = wn_label
                provenance['node_label']      = label
                provenance['source_evidence'] = ['wordnet_hypernym']

        # Guarantee distinct siblings: qualify a colliding label with a distinguishing
        # word from this cluster's own titles (never emit a duplicate sibling).
        if label.lower() in used_sibling_labels:
            from collections import Counter as _Counter
            _cnt = _Counter()
            for _tt in member_titles_k:
                for _w in re.findall(r'[a-z]{3,}', _tt.lower()):
                    if _w not in _STOP and _w not in ancestor_words and _w not in label.lower():
                        _cnt[_w] += 1
            _extra = next((w for w, _ in _cnt.most_common()
                           if f'{label} {w}'.lower() not in used_sibling_labels), None)
            if _extra:
                label = f'{label} {_extra.title()}'
            else:
                _i = 2
                while f'{label} {_i}'.lower() in used_sibling_labels:
                    _i += 1
                label = f'{label} {_i}'

        used_sibling_labels.add(label.lower())  # register for sibling dedup (string + emb)
        try:
            sibling_label_embs.append(embedder.encode([label])[0])
        except Exception:
            pass

        pid = ensure_path_fn(path_prefix + [label],
                              relation='belongs_to', provenance=provenance)

        # Store concept assignment on can (needed by Castanet facets later).
        # Provenance reflects the HYBRID winner (title / keybert / concept_table),
        # not the old semantic-only scorer — so the exported labels CSV is accurate.
        for ci in cluster_idxs:
            _, row = rows_list[ci]
            can.at[row.name, '_concept_label']  = label
            can.at[row.name, '_concept_score']  = provenance.get('confidence', 0.0)
            can.at[row.name, '_concept_source'] = (provenance.get('source_evidence') or ['fallback'])[0]

        # Attach the cluster's variables directly under the concept node. The former
        # Statistic/Condition facet sub-split is removed: the statistic tier came from
        # a hardcoded vocabulary (now produced data-drivenly by _nest_by_measure), and
        # the numeric Condition tier produced bare-digit nodes (0/4/12) that inflated
        # singleton%/n_agg and moved the tree away from gold. Castanet's Condition facet
        # still exists as a separate parallel view via detect_facets() — not a tier.
        for ci in cluster_idxs:
            _, row = rows_list[ci]
            add_child(nodes, pid, leaf_to_id[row['_leaf_id']])


def _remove_phrase(tokens, phrase_tokens):
    """Remove the first contiguous occurrence of phrase_tokens from tokens."""
    nlen = len(phrase_tokens)
    for i in range(len(tokens) - nlen + 1):
        if tokens[i:i + nlen] == phrase_tokens:
            return tokens[:i] + tokens[i + nlen:]
    return [t for t in tokens if t not in phrase_tokens]


def _nest_by_measure(nodes):
    """
    [Fix2] Group concept-sibling nodes that SHARE a measure phrase into a Measure
    parent, renaming each child to its residual statistic. Example under DMS:
        Mean Correct Latency, Median Correct Latency, Correct Latency Standard Deviation
        →  Correct Latency → { Mean, Median, Standard Deviation }
    The measure is simply the phrase shared by ≥2 siblings; the statistic is what
    remains after removing it. No hardcoded statistic list. Adds Measure→Statistic
    depth only where the data supports it; other concepts stay flat.
    """
    pm = build_parent_map(nodes)
    task_ids = [int(n['id']) for n in nodes
                if n.get('type') == 'aggregation' and pm.get(int(n['id'])) == 0]
    for task_id in task_ids:
        while True:
            m = nmap(nodes)
            task = m.get(task_id)
            if not task:
                break
            child_ids = [int(c) for c in task.get('related', [])
                         if m.get(int(c), {}).get('type') == 'aggregation']
            if len(child_ids) < 3:
                break
            labels = {cid: str(m[cid]['name']) for cid in child_ids}
            phrase_children = defaultdict(set)
            for cid, lbl in labels.items():
                toks = [w for w in re.findall(r'[a-z][a-z\-]+', lbl.lower()) if w not in _STOP]
                for nlen in (3, 2):
                    for i in range(len(toks) - nlen + 1):
                        phrase_children[' '.join(toks[i:i + nlen])].add(cid)
            cand = [(ph, cids) for ph, cids in phrase_children.items() if len(cids) >= 2]
            if not cand:
                break
            ph, grouped = max(cand, key=lambda x: (len(x[1]), len(x[0].split())))
            ptoks = ph.split()
            nid = next_id(nodes)
            nodes.append(make_agg(nid, ph.title(), desc=f'Measure: {ph.title()}',
                                  relation_type='belongs_to'))
            add_child(nodes, task_id, nid)
            for cid in list(grouped):
                remove_child(nodes, task_id, cid)
                ctoks = [w for w in re.findall(r'[a-z][a-z\-]+', labels[cid].lower())
                         if w not in _STOP]
                resid = _remove_phrase(ctoks, ptoks)
                if len(resid) == 1:
                    # A lone modifier ("Double", "Within") reads poorly on its own —
                    # qualify it with the measure's most-informative word (longest;
                    # ties → last), e.g. "Double" → "Double Errors". No hardcoding.
                    mword = max(ptoks, key=lambda w: (len(w), ptoks.index(w)))
                    if mword not in resid:
                        resid = resid + [mword]
                if resid:
                    m[cid]['name'] = ' '.join(resid).title()
                    add_child(nodes, nid, cid)
                else:
                    # child label == measure → dissolve it, leaves go under new parent
                    for leaf in list(m[cid].get('related', [])):
                        add_child(nodes, nid, int(leaf))
                    nodes[:] = [x for x in nodes if int(x['id']) != cid]
    return nodes


def _singular(w):
    return w[:-1] if (len(w) > 3 and w.endswith('s') and not w.endswith('ss')) else w


def _nest_by_category(nodes):
    """
    [Fix3] Add a Measure-CATEGORY tier: group a task's concept-sibling nodes by their
    HEAD noun (last significant word, singularised) when ≥2 share it, e.g.
        Total Correct, Percent Correct  →  Correct → { Total, Percent }
        Total Errors, Probability Error →  Errors  → { Total, Probability }
    The HEAD is used (not any shared word) specifically so "Correct Latency"
    (head = Latency) is NOT pulled under "Correct". Children are renamed to the
    residual (label minus the head). No hardcoded category list.
    """
    pm = build_parent_map(nodes)
    task_ids = [int(n['id']) for n in nodes
                if n.get('type') == 'aggregation' and pm.get(int(n['id'])) == 0]
    for task_id in task_ids:
        m = nmap(nodes)
        task = m.get(task_id)
        if not task:
            continue
        child_ids = [int(c) for c in task.get('related', [])
                     if m.get(int(c), {}).get('type') == 'aggregation']
        if len(child_ids) < 3:
            continue
        head_groups, head_forms, labels = defaultdict(list), defaultdict(list), {}
        for cid in child_ids:
            lbl = str(m[cid]['name'])
            labels[cid] = lbl
            words = [w for w in re.findall(r'[a-z][a-z\-]+', lbl.lower()) if w not in _STOP]
            if not words:
                continue
            sg = _singular(words[-1])
            head_groups[sg].append(cid)
            head_forms[sg].append(words[-1])
        for sg, cids in list(head_groups.items()):
            if len(cids) < 2:
                continue
            cat = max(head_forms[sg], key=len).title()      # nicest surface form
            nid = next_id(nodes)
            nodes.append(make_agg(nid, cat, desc=f'Category: {cat}',
                                  relation_type='belongs_to'))
            add_child(nodes, task_id, nid)
            for cid in cids:
                remove_child(nodes, task_id, cid)
                ctoks = [w for w in re.findall(r'[a-z][a-z\-]+', labels[cid].lower())
                         if w not in _STOP]
                resid = [t for t in ctoks if _singular(t) != sg]
                if resid:
                    m[cid]['name'] = ' '.join(resid).title()
                    add_child(nodes, nid, cid)
                else:
                    for leaf in list(m[cid].get('related', [])):
                        add_child(nodes, nid, int(leaf))
                    nodes[:] = [x for x in nodes if int(x['id']) != cid]
            m = nmap(nodes)
    return nodes


def _merge_duplicate_concepts(nodes):
    """
    [Fix] Merge aggregation nodes that share the SAME name within the same task
    (keeping the shallowest), e.g. SWM had a singleton 'Within Errors' AND an
    'Errors Boxes > Within Errors' — both become one flat 'Within Errors'. Removes
    duplicates created when clustering split a concept's variants and #5 / _nest_by_measure
    labelled them identically.
    """
    pm = build_parent_map(nodes)
    def depth(nid):
        d, x = 0, nid
        while x in pm:
            x = pm[x]; d += 1
        return d
    def task_of(nid):
        x = nid
        while True:
            p = pm.get(x)
            if p is None or p == 0:
                return x
            x = p
    groups = defaultdict(list)
    for n in nodes:
        if n.get('type') == 'aggregation' and int(n['id']) in pm:
            groups[(task_of(int(n['id'])), str(n['name']).lower())].append(int(n['id']))
    removed = set()
    for (_t, _name), ids in groups.items():
        ids = [i for i in ids if i not in removed]
        if len(ids) < 2:
            continue
        keeper = min(ids, key=depth)
        m = nmap(nodes)
        for dup in ids:
            if dup == keeper:
                continue
            for c in list(m[dup].get('related', [])):
                remove_child(nodes, dup, int(c)); add_child(nodes, keeper, int(c))
            # Remove dup from whatever node CURRENTLY references it (not the stale pm —
            # earlier post-processes may have re-parented it). Leaving a stale ref makes
            # a dangling child that breaks the Plotly sunburst.
            for pn in nodes:
                if dup in [int(x) for x in pn.get('related', [])]:
                    remove_child(nodes, int(pn['id']), dup)
            removed.add(dup)
    nodes[:] = [n for n in nodes if int(n['id']) not in removed]
    # Defensive: drop any child reference to a node that no longer exists.
    _alive = {int(n['id']) for n in nodes}
    for n in nodes:
        n['related'] = [int(x) for x in n.get('related', []) if int(x) in _alive]
    return nodes


def _prune_empty_aggregations(nodes):
    """Remove aggregation nodes whose subtree contains NO variable (leaf). Empty
    concept nodes are meaningless AND break the Plotly sunburst: every node gets a
    min value of 1, so an empty child makes its parent's value < sum(children) and
    branchvalues='total' refuses to render (blank chart)."""
    m = nmap(nodes)
    def has_leaf(nid, seen):
        if nid in seen:
            return False
        seen.add(nid)
        n = m.get(nid)
        if not n:
            return False
        if n.get('type') == 'attribute':
            return True
        return any(has_leaf(int(c), seen) for c in n.get('related', []))
    empty = {int(n['id']) for n in nodes
             if n.get('type') == 'aggregation' and not has_leaf(int(n['id']), set())}
    if empty:
        nodes[:] = [n for n in nodes if int(n['id']) not in empty]
        alive = {int(n['id']) for n in nodes}
        for n in nodes:
            n['related'] = [int(c) for c in n.get('related', []) if int(c) in alive]
    return nodes


def _dissolve_facet_singletons(nodes):
    """
    Dissolve FACET tier nodes (Statistic / Condition) that wrap a single variable.
    A condition or statistic node with exactly one leaf child carries no grouping
    value — e.g. `Standard Deviation > 0 > DMSL0SD`. We remove such nodes and
    reattach their single child to the node's parent, keeping siblings together.

    Scope is deliberately narrow: only nodes whose relation_type is 'has_condition'
    or 'is_statistic_of' are touched, so genuine single-member CONCEPT nodes that
    carry a distinctive name are preserved (per the chosen policy).
    """
    _FACET_RELS = {'has_condition', 'is_statistic_of'}
    changed = True
    while changed:
        changed = False
        pm = build_parent_map(nodes)
        m  = nmap(nodes)
        for n in list(nodes):
            if n.get('type') != 'aggregation':
                continue
            if n['info'].get('relation_type') not in _FACET_RELS:
                continue
            nid      = int(n['id'])
            children = [int(c) for c in n.get('related', [])]
            # "Single variable" = exactly one child and that child is a leaf attribute.
            if len(children) == 1 and m.get(children[0], {}).get('type') == 'attribute':
                parent = pm.get(nid)
                if parent is None:
                    continue
                add_child(nodes, parent, children[0])
                remove_child(nodes, parent, nid)
                nodes[:] = [x for x in nodes if int(x['id']) != nid]
                changed = True
                break
    return nodes


def build_concept_hierarchy(can, embedder, concept_table, project='metadata_project',
                             n_clusters_per_group=8):
    """
    Build hierarchy using automatic concept label assignment.
    No hardcoded patterns. Labels come from metadata + external concept table.
    [GON] N×M alignment · [TAX] leaf=attribute, node=abstract concept · [HIE] task-first
    """
    nodes     = [{'id': 0, 'name': project, 'desc': 'Root node', 'type': 'root',
                  'dtype': 'root', 'isShown': True, 'related': []}]
    leaf_to_id = {}

    for i, (_, r) in enumerate(can.iterrows(), start=1):
        leaf_to_id[r['_leaf_id']] = i
        nodes.append({
            'id':           i,
            'name':         r['_leaf_label'],
            'dtype':        r['_dtype'],
            'related':      [],
            'isShown':      True,
            'type':         'attribute',
            'desc':         r['_text'],
            'semantic_desc': r.get('_semantic_text', r['_text']),
            'source_file':  r['_source_file'],
            'metadata':     {'leaf_id': r['_leaf_id'], 'group_path': r['_group_path']},
        })

    # Embed concept table once for the whole hierarchy build
    if concept_table:
        concept_texts = [c['full_text'] for c in concept_table]
        concept_embs  = embedder.encode(concept_texts)
    else:
        concept_embs = None

    # ── Dataset-wide reference embeddings for semantic IDF / specificity [GON] ──
    # Encode every variable once, then build one centroid per top-level group.
    # A candidate label that is close to ONE group centroid and far from the rest
    # is discriminative; one close to ALL of them is boilerplate. corpus_centroid
    # is the global mean (generic = central). Both are derived purely from data.
    sem_col_all = '_semantic_text' if '_semantic_text' in can.columns else '_text'

    # Active domain — used by the hybrid label scorer's external-grounding signal.
    global _ACTIVE_DOMAIN
    _ACTIVE_DOMAIN = detect_domain(can)

    # Corpus IDF over description n-grams — KeyBERT c-TF-IDF distinctiveness weight
    # (only consulted when USE_CTFIDF=True). Data-derived, dataset-agnostic.
    global _CORPUS_IDF
    _CORPUS_IDF = {}
    try:
        from sklearn.feature_extraction.text import CountVectorizer as _CV
        _docs = can[sem_col_all].fillna('').astype(str).tolist()
        _cv = _CV(ngram_range=(1, 3), binary=True, lowercase=True,
                  token_pattern=r'[a-z][a-z\-]+')
        _dt = _cv.fit_transform(_docs)
        _dfa = np.asarray(_dt.sum(axis=0)).ravel(); _N = _dt.shape[0]
        _CORPUS_IDF = {p: float(np.log((_N + 1) / (_dfa[i] + 1)) + 1.0)
                       for p, i in _cv.vocabulary_.items()}
    except Exception:
        _CORPUS_IDF = {}

    ref_centroids = corpus_centroid = None
    try:
        all_var_embs = embedder.encode(can[sem_col_all].fillna('').astype(str).tolist())
        corpus_centroid = _centroid(all_var_embs)
        _tops = can['_group_path'].fillna('Ungrouped').apply(
            lambda x: str(x).split(' > ')[0].strip() or 'Ungrouped')
        _cent = []
        for g in _tops.unique():
            gm = (_tops == g).to_numpy()
            if gm.sum() >= 1:
                _cent.append(_centroid(all_var_embs[gm]))
        ref_centroids = np.array(_cent) if len(_cent) >= 2 else None
    except Exception:
        pass

    path_ids = {}

    def ensure_path(parts, relation='belongs_to', provenance=None):
        key = tuple(str(p) for p in parts)
        if key in path_ids:
            return path_ids[key]
        nid = next_id(nodes)
        path_ids[key] = nid
        nodes.append(make_agg(nid, parts[-1],
                               desc=f'Concept group: {" > ".join(str(p) for p in parts)}',
                               relation_type=relation,
                               provenance=provenance))
        parent = 0 if len(parts) == 1 else ensure_path(parts[:-1])
        add_child(nodes, parent, nid)
        return nid

    # Group variables by top-level group path
    work = can.copy()
    work['_top'] = work['_group_path'].apply(
        lambda x: str(x).split(' > ')[0].strip()
        if str(x) not in ('', 'nan', 'Ungrouped') else 'Ungrouped'
    )

    for top_label, tdf in work.groupby('_top', dropna=False, sort=False):
        top_label = str(top_label)

        # Check if sub-group paths already exist (level 2+)
        subgroup_paths = tdf['_group_path'].apply(
            lambda x: ' > '.join(str(x).split(' > ')[1:]).strip()
            if len(str(x).split(' > ')) > 1 else ''
        )
        has_subgroups = subgroup_paths.str.strip().str.len().gt(0).any()

        if has_subgroups:
            # [C4][CAS] UnaryPenalty: count distinct subgroup paths under this top group.
            # Castanet: "eliminate a child whose name appears within the parent's name"
            # and nodes that create unary (1-child) chains weaken the hierarchy.
            # If a structural column creates only ONE branch under this parent,
            # it is a pass-through (e.g. "DMS Recommended Standard" under "DMS")
            # and should be demoted — cluster directly under the top-level node instead.
            distinct_subpaths = subgroup_paths[subgroup_paths.str.strip().str.len().gt(0)].unique()
            n_distinct_subpaths = len(distinct_subpaths)

            if n_distinct_subpaths <= 1:
                # UnaryPenalty triggered — structural column creates only 1 branch.
                # Cluster directly under top_label, skip the variant pass-through. [C4]
                _cluster_and_label(
                    tdf, [top_label], nodes, leaf_to_id, embedder,
                    concept_table, concept_embs, ensure_path,
                    n_clusters_per_group, can, ref_centroids, corpus_centroid
                )
            else:
                # Multiple distinct subgroups — structural column is meaningful, keep it.
                for subpath, sdf in tdf.groupby(subgroup_paths, dropna=False, sort=False):
                    subpath = str(subpath).strip()
                    if subpath:
                        parts = [top_label] + [p.strip() for p in subpath.split(' > ') if p.strip()]
                    else:
                        parts = [top_label]
                    # Cluster and label within this subgroup
                    _cluster_and_label(
                        sdf, parts, nodes, leaf_to_id, embedder,
                        concept_table, concept_embs, ensure_path,
                        n_clusters_per_group, can, ref_centroids, corpus_centroid
                    )
        else:
            # No pre-existing subgroups — cluster all variables under this top group
            _cluster_and_label(
                tdf, [top_label], nodes, leaf_to_id, embedder,
                concept_table, concept_embs, ensure_path,
                n_clusters_per_group, can, ref_centroids, corpus_centroid
            )

    # [Fix2] Nest statistics under their shared measure (Correct Latency → Mean/…),
    # then merge same-named duplicates, THEN sub-split the consolidated concept
    # nodes by a distinctive title descriptor (Error → Incorrect Colour / …). Order
    # matters: sub-splitting last avoids the merge re-parenting sub-nodes oddly.
    _nest_by_measure(nodes)
    _merge_duplicate_concepts(nodes)
    _subsplit_concept_by_title(nodes)
    # Remove empty concept nodes (no variables) — meaningless and they break the
    # branchvalues='total' sunburst (parent value < sum of children → blank render).
    _prune_empty_aggregations(nodes)
    # Dissolve 1-variable Statistic/Condition facet nodes (no grouping value).
    _dissolve_facet_singletons(nodes)
    _prune_empty_aggregations(nodes)
    # NOTE: a head-noun "category" tier (Errors/Correct) was tried and reverted —
    # it regressed setOverlap (0.914→0.836: mis-grouping) and added depth beyond gold.
    # _nest_by_category() is kept defined but intentionally NOT called.

    _alive = {int(n['id']) for n in nodes}
    for n in nodes:
        n['related'] = [x for x in dict.fromkeys(int(x) for x in n.get('related', []))
                        if x in _alive]   # dedup + drop dangling refs (sunburst safety)

    return nodes

# ─────────────────────────────────────────────────────────────────────────────
# HIEXPAN-INSPIRED REFINEMENT [HIE]
# ─────────────────────────────────────────────────────────────────────────────
def _leaf_texts(nodes, nid, text_cache):
    return [text_cache[i] for i in leaf_ids(nodes, nid) if i in text_cache]

def _build_emb_cache(nodes, embedder, text_cache):
    """
    [HIE] Pre-compute ALL leaf embeddings in one batch call.
    Paper-correct: HiExpan pre-computes entity representations once upfront,
    then all expansion passes reuse the cache — no re-encoding per leaf per pass.
    """
    ids   = [int(n['id']) for n in nodes
             if n['type'] == 'attribute' and text_cache.get(int(n['id']), '').strip()]
    if not ids:
        return {}
    texts = [text_cache[i] for i in ids]
    embs  = embedder.encode(texts)   # ONE batch call for everything
    return {nid: embs[i] for i, nid in enumerate(ids)}

def hiexpan_sibling_coherence(nodes, embedder, text_cache, emb_cache=None):
    """[HIE] Mean pairwise cosine similarity of attribute children per group node."""
    report = []
    m = nmap(nodes)
    if emb_cache is None:
        emb_cache = _build_emb_cache(nodes, embedder, text_cache)
    for node in nodes:
        if node['type'] not in ('aggregation', 'root'): continue
        attr_ch = [int(c) for c in node.get('related', [])
                   if m.get(int(c), {}).get('type') == 'attribute']
        if len(attr_ch) < 2: continue
        embs_ = np.array([emb_cache[cid] for cid in attr_ch if cid in emb_cache])
        if len(embs_) < 2: continue
        sims     = cosine_similarity(embs_)
        n_       = len(embs_)
        mask_    = np.triu(np.ones((n_, n_), dtype=bool), k=1)
        mean_sim = float(sims[mask_].mean()) if mask_.any() else 1.0
        report.append({'node_id': int(node['id']), 'name': node['name'],
                       'n_attr_children': n_, 'coherence_score': round(mean_sim, 3),
                       'is_incoherent': mean_sim < 0.25})
    return sorted(report, key=lambda x: x['coherence_score'])

def hiexpan_width_expansion(nodes, embedder, text_cache, threshold=0.45, emb_cache=None):
    """[HIE] Move each leaf to the sibling group with highest centroid similarity.
    Uses pre-built emb_cache — no per-leaf encode() calls (paper-correct, fast)."""
    import copy
    nodes = copy.deepcopy(nodes)
    m     = nmap(nodes)
    if emb_cache is None:
        emb_cache = _build_emb_cache(nodes, embedder, text_cache)

    measure_nodes = [n for n in nodes if n['type'] == 'aggregation'
                     and any(m.get(int(c), {}).get('type') == 'attribute'
                             for c in n.get('related', []))]
    if len(measure_nodes) < 2: return nodes, 0

    # Build group centroids from cached embeddings — no new encode() calls
    node_centroids = {}
    for mn in measure_nodes:
        leaf_ids_ = [int(c) for c in leaf_ids(nodes, int(mn['id']))
                     if int(c) in emb_cache]
        if leaf_ids_:
            embs_ = np.array([emb_cache[i] for i in leaf_ids_])
            node_centroids[int(mn['id'])] = _centroid(embs_)

    if len(node_centroids) < 2: return nodes, 0
    cent_ids   = list(node_centroids.keys())
    cent_array = np.stack([node_centroids[i] for i in cent_ids])

    n_moves = 0
    for leaf in [n for n in nodes if n['type'] == 'attribute']:
        lid = int(leaf['id'])
        if lid not in emb_cache: continue
        leaf_emb        = emb_cache[lid]          # cached — no encode() call
        current_parents = parents(nodes, lid)
        current_measure = [p for p in current_parents if p in node_centroids]
        if not current_measure: continue
        cur_p    = current_measure[0]
        sims     = cent_array.dot(leaf_emb)
        best_idx = int(np.argmax(sims))
        best_p   = cent_ids[best_idx]
        if best_p != cur_p and float(sims[best_idx]) > node_centroids[cur_p].dot(leaf_emb) + 0.02:
            nodes = move_node(nodes, lid, best_p)
            n_moves += 1
    return nodes, n_moves

def hiexpan_depth_expansion_semantic(nodes, embedder, text_cache, concept_table,
                                     concept_embs, n_subclusters=3, emb_cache=None,
                                     coherence_threshold=0.45, top_level_tasks=None):
    """
    [HIE][C1][C2] Depth expansion — embedding-based, no hardcoded patterns.
    Uses pre-built emb_cache — no per-group encode() calls (paper-correct, fast).

    [C2][TAX][RAPTOR] Quality gate added:
      - Balance check: largest sub-cluster must be ≤ 70% of total (prevents degenerate splits)
      - Min size: each sub-cluster must have ≥ 2 variables (HiExpan coherence requires siblings)
      - coherence_threshold: tunable per recursive pass (lowered across passes for deeper trees)
    """
    import copy
    nodes = copy.deepcopy(nodes)
    m     = nmap(nodes)
    if emb_cache is None:
        emb_cache = _build_emb_cache(nodes, embedder, text_cache)
    n_exp = 0

    for agg in [n for n in nodes if n['type'] == 'aggregation']:
        attr_ch = [int(c) for c in agg.get('related', [])
                   if m.get(int(c), {}).get('type') == 'attribute']
        if len(attr_ch) < 3:
            continue

        # Check coherence using cached embeddings — no new encode() calls
        cached_ids  = [cid for cid in attr_ch if cid in emb_cache]
        if len(cached_ids) < 2:
            continue
        embs = np.array([emb_cache[cid] for cid in cached_ids])
        sims = cosine_similarity(embs)
        n_   = len(cached_ids)
        mask_ = np.triu(np.ones((n_, n_), dtype=bool), k=1)
        coherence = float(sims[mask_].mean()) if mask_.any() else 1.0

        # [F6][HIE][RAPTOR] Revised depth-expansion gate:
        # Original: skip if coherence ≥ threshold — WRONG for narrow-vocab domains
        # (CANTAB/HCP have high cosine similarity even across sub-types).
        # New rule: skip ONLY when coherent AND small AND low vocabulary diversity.
        # HiExpan: expands wide nodes; RAPTOR: splits while BIC improves.
        # A coherent-but-large node with diverse sub-vocabularies is a good parent
        # whose children have not yet been discovered — depth expansion IS needed.
        if coherence >= coherence_threshold:
            # Allow depth expansion for large nodes with vocabulary diversity
            if len(attr_ch) < 6:
                continue  # Truly small coherent cluster — no further split needed
            # Compute vocabulary range: max unique non-stop tokens minus min
            vocab_sizes = []
            for cid in cached_ids:
                txt = text_cache.get(cid, '').lower()
                toks = set(re.findall(r'\b[a-z]{4,}\b', txt)) - _STOP
                vocab_sizes.append(len(toks))
            vocab_range = max(vocab_sizes) - min(vocab_sizes) if vocab_sizes else 0
            if vocab_range < 3:
                continue  # Low internal diversity — truly homogeneous, stop here

        # Re-cluster the leaves
        k_sub = min(n_subclusters, max(2, len(attr_ch) // 3))
        try:
            dist   = cosine_distances(embs).astype(float)
            np.fill_diagonal(dist, 0.0)
            sub_lbs = AgglomerativeClustering(n_clusters=k_sub, metric='precomputed',
                                              linkage='average').fit_predict(dist)
        except Exception:
            continue

        # [C2][TAX][RAPTOR] Quality gate: reject unbalanced or degenerate splits
        cluster_sizes = [int((sub_lbs == sk).sum()) for sk in range(k_sub)]
        total_size    = sum(cluster_sizes)
        max_cluster   = max(cluster_sizes) if cluster_sizes else 0
        # Balance: largest cluster ≤ 70% of total
        if total_size > 0 and max_cluster / total_size > 0.70:
            continue  # Degenerate split — one cluster dominates, no real gain
        # Min-size: every sub-cluster must have ≥ 2 variables (HiExpan sibling coherence)
        if any(s < 2 for s in cluster_sizes):
            continue

        # Remove direct leaf connections from this agg node
        agg_id = int(agg['id'])
        for cid in attr_ch:
            remove_child(nodes, agg_id, cid)

        # Per sub-cluster leaf ids + their TITLES (so HiExpan labels from titles too,
        # not the boilerplate definition path).
        sub_cids_by_sk  = [[cached_ids[i] for i, m_ in enumerate(sub_lbs == sk) if m_]
                           for sk in range(k_sub)]
        sub_titles_by_sk = [[_concept_title(text_cache.get(cid, '')) for cid in cids]
                            for cids in sub_cids_by_sk]
        agg_ancestors = ancestor_names(nodes, agg_id) + [agg['name']]
        _anc_words = set(re.findall(r'[a-z]{3,}', ' '.join(agg_ancestors).lower()))
        _used_sub  = set()

        # Create sub-nodes — TITLE label wins; concept scoring only as fallback.
        for sk in range(k_sub):
            sub_cids = sub_cids_by_sk[sk]
            if not sub_cids:
                continue
            sub_mask  = sub_lbs == sk
            sub_texts = [text_cache.get(cid, '') for cid in sub_cids]
            sub_emb   = _centroid(embs[sub_mask])

            title_label = _title_cluster_label(
                sub_titles_by_sk[sk],
                [sub_titles_by_sk[j] for j in range(k_sub) if j != sk],
                ancestor_words=_anc_words, used_labels=_used_sub)

            if title_label:
                label = title_label
                provenance = {'node_label': label, 'source_evidence': ['description_title'],
                              'confidence': 0.0, 'alternatives': []}
            elif sub_emb is not None and concept_embs is not None and concept_table:
                scores = score_concepts_for_cluster(sub_emb, concept_embs, concept_table, sub_texts)
                label, provenance = assign_concept_label(
                    scores, fallback=f'{agg["name"]} {sk+1}',
                    ancestor_names=agg_ancestors, top_level_tasks=top_level_tasks,
                )
            else:
                label     = f'{agg["name"]} {sk+1}'
                provenance = None
            _used_sub.add(str(label).lower())

            nid = next_id(nodes)
            nodes.append(make_agg(nid, label,
                                   desc=f'Sub-group of {agg["name"]}: {label}',
                                   relation_type='belongs_to',
                                   provenance=provenance))
            add_child(nodes, agg_id, nid)
            for cid in sub_cids:
                add_child(nodes, nid, cid)
        n_exp += 1

    return nodes, n_exp

def hiexpan_global_optimization(nodes, embedder, text_cache, n_passes=2, emb_cache=None):
    """[HIE] Global optimization — repeated width expansion passes until convergence.
    Reuses emb_cache — no new encode() calls across passes."""
    total = 0
    for _ in range(n_passes):
        nodes, moves = hiexpan_width_expansion(nodes, embedder, text_cache,
                                               threshold=0.40, emb_cache=emb_cache)
        total += moves
        if moves == 0: break
    return nodes, total

def run_hiexpan(nodes, can, embedder, concept_table=None, concept_embs=None,
                max_depth_passes=4):
    """
    [HIE][C1][C2] Run all HiExpan passes with a single pre-built embedding cache.
    Paper-correct: encode all leaves ONCE, reuse across coherence / width / depth / global.
    This reduces HiExpan from O(n_passes × n_leaves) encode calls to O(1).

    [C1][HIE][TAX] Recursive depth expansion loop:
    HiExpan: "builds the taxonomy by recursively expanding all these sets."
    TaxoGen: "splitting a coarse topic into fine-grained ones" is iterative.
    Progressive coherence thresholds: [0.45, 0.38, 0.30, 0.22] — each pass allows
    finer splits, pushing the hierarchy deeper until max_depth_passes or convergence.
    """
    text_cache = {int(n['id']): str(n.get('semantic_desc', n.get('desc', '')))
                  for n in nodes if n['type'] == 'attribute'}

    # ── Pre-encode ALL leaves once (HiExpan paper: pre-compute entity representations)
    emb_cache = _build_emb_cache(nodes, embedder, text_cache)

    report = {}
    report['coherence_before']      = hiexpan_sibling_coherence(
        nodes, embedder, text_cache, emb_cache=emb_cache)
    nodes, n_width                  = hiexpan_width_expansion(
        nodes, embedder, text_cache, emb_cache=emb_cache)
    report['width_expansion_moves'] = n_width

    # [C1][HIE][TAX] Recursive depth expansion — progressive threshold schedule
    # Pass 1: threshold=0.45 (broad splits)
    # Pass 2: threshold=0.38 (medium splits)
    # Pass 3: threshold=0.30 (fine splits)
    # Pass 4: threshold=0.22 (very fine — only if still incoherent)
    # [FIX1] DEPTH-EXPANSION DISABLED. It split concept nodes (Total Correct,
    # Percent Correct, Error) by DELAY condition into sub-clusters whose titles
    # differ only by a parenthetical number — so the title labeler found nothing
    # distinctive and fell back to the boilerplate "Calculated Assessed Trials"
    # candidate. Those repetitive "same children" tiers are removed by not running
    # this pass; the hierarchy stays Task → concept → leaves. Measure→Statistic
    # depth is handled separately from the title composition (Fix 2), not here.
    total_depth_exp = 0
    pass_idx = -1

    report['depth_expansion_nodes'] = total_depth_exp
    report['depth_expansion_passes'] = pass_idx + 1

    nodes, n_global                 = hiexpan_global_optimization(
        nodes, embedder, text_cache, emb_cache=emb_cache)
    report['global_optimization_moves'] = n_global
    report['coherence_after']       = hiexpan_sibling_coherence(
        nodes, embedder, text_cache, emb_cache=emb_cache)
    return nodes, report

# ─────────────────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION TABLE [HIE]
# Variables where the top-2 concept assignments differ by < 0.05 in score.
# ─────────────────────────────────────────────────────────────────────────────
def compute_conflict_table(can, nodes):
    """
    [HIE] Full conflict resolution table.
    For each low-confidence variable, computes similarity to current parent centroid
    and to all sibling group centroids — shows top-2 alternative placements.
    """
    pm  = build_parent_map(nodes)
    m   = nmap(nodes)

    # Build centroid cache for all aggregation nodes
    agg_nodes = [n for n in nodes if n.get('type') == 'aggregation']
    text_cache = {int(n['id']): str(n.get('desc', ''))
                  for n in nodes if n.get('type') == 'attribute'}

    # TF-IDF similarity proxy (no embedder available here — use text overlap)
    def _sim_to_group(var_text, agg_node):
        agg_texts = [text_cache.get(int(c), '')
                     for c in agg_node.get('related', [])
                     if m.get(int(c), {}).get('type') == 'attribute']
        if not agg_texts:
            return 0.0
        combined = ' '.join(agg_texts).lower()
        var_words = set(re.findall(r'\b[a-z]{3,}\b', var_text.lower())) - _STOP
        group_words = set(re.findall(r'\b[a-z]{3,}\b', combined)) - _STOP
        if not var_words:
            return 0.0
        return len(var_words & group_words) / len(var_words)

    rows = []
    for _, row in can.iterrows():
        score = float(row.get('_concept_score', 0.0))
        if not (0 < score < 0.25):
            continue
        lid   = row['_leaf_id']
        lid_n = [n for n in nodes if n.get('metadata', {}).get('leaf_id') == lid]
        if not lid_n:
            continue
        nid        = int(lid_n[0]['id'])
        parent_id  = pm.get(nid)
        parent_n   = m.get(parent_id, {})
        parent_nm  = parent_n.get('name', '')
        var_text   = str(row.get('_text', ''))

        # Sibling groups = all aggregation nodes under same grandparent
        grandparent_id = pm.get(parent_id)
        sibling_groups = [
            n for n in agg_nodes
            if pm.get(int(n['id'])) == grandparent_id and int(n['id']) != parent_id
        ]
        sib_sims = sorted(
            [{'name': sn['name'], 'sim': round(_sim_to_group(var_text, sn), 3)}
             for sn in sibling_groups],
            key=lambda x: -x['sim']
        )
        cur_sim  = round(_sim_to_group(var_text, parent_n), 3) if parent_n else 0.0
        alt1     = sib_sims[0] if len(sib_sims) > 0 else {'name': '—', 'sim': 0.0}
        alt2     = sib_sims[1] if len(sib_sims) > 1 else {'name': '—', 'sim': 0.0}
        decision = ('Move to alt-1' if alt1['sim'] > cur_sim + 0.10
                    else 'Review manually' if alt1['sim'] > cur_sim
                    else 'Keep current')
        rows.append({
            'variable':            row['_leaf_label'],
            'concept_label':       row.get('_concept_label', ''),
            'concept_score':       round(score, 3),
            'current_parent':      parent_nm,
            'current_sim':         cur_sim,
            'alt_parent_1':        alt1['name'],
            'alt_sim_1':           alt1['sim'],
            'alt_parent_2':        alt2['name'],
            'alt_sim_2':           alt2['sim'],
            'decision':            decision,
            'source':              row.get('_concept_source', ''),
        })
    cols = ['variable', 'concept_label', 'concept_score',
            'current_parent', 'current_sim',
            'alt_parent_1', 'alt_sim_1',
            'alt_parent_2', 'alt_sim_2',
            'decision', 'source']
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# ─────────────────────────────────────────────────────────────────────────────
# CASTANET PARALLEL FACETED HIERARCHIES [CAS]
# Uses automatic concept assignments — no hardcoded dimension patterns.
# ─────────────────────────────────────────────────────────────────────────────
def detect_facets(can, code_expansions=None):
    """
    [CAS] Auto-detect semantic facet dimensions from the actual metadata structure.
    Each facet is only added when meaningful (>1 distinct value in the data).
    No hardcoded labels — facet names and groups come entirely from the data.

    Detects (in order, only if data supports them):
      Task          — top-level group path
      Variant       — second-level group path
      Measure Type  — automatically assigned concept label (N×M alignment)
      Statistic     — detected from description text (mean/sd/median/percent etc.)
      Condition     — numeric code suffix (delay/timepoint conditions)
      Outcome Type  — outcome/error keywords detected from description text
      Scale/Precision — metadata columns (decimal places, unit, format)
      Code Family   — structural prefix groups
    """
    facets = []
    code_expansions = code_expansions or {}
    sem_col = '_semantic_text' if '_semantic_text' in can.columns else '_text'

    # ── Facet: Task (top-level group path) ────────────────────────────────────
    top_groups = can['_group_path'].apply(
        lambda x: str(x).split(' > ')[0].strip()
        if str(x) not in ('', 'nan', 'Ungrouped') else 'Ungrouped'
    )
    if top_groups.nunique() > 1:
        can['_facet_task'] = top_groups
        facets.append({
            'name': 'Task',
            'desc': 'Variables grouped by their top-level task or domain.',
            'fn':   (lambda col: lambda row: str(row.get(col, 'Ungrouped')))('_facet_task'),
            'relation': 'belongs_to',
        })

    # ── Facet: Variant (second-level group path, if present) ──────────────────
    def _second_level(gpath):
        parts = str(gpath).split(' > ')
        return parts[1].strip() if len(parts) > 1 else ''
    second = can['_group_path'].apply(_second_level)
    if second.ne('').any() and second.nunique() > 1:
        can['_facet_variant'] = second.where(second != '', 'No Variant')
        facets.append({
            'name': 'Variant',
            'desc': 'Variables grouped by their second-level structural variant.',
            'fn':   (lambda col: lambda row: str(row.get(col, 'No Variant')))('_facet_variant'),
            'relation': 'belongs_to',
        })

    # ── Facet: Measure Type (concept label from N×M alignment) ────────────────
    has_concepts = can['_concept_label'].fillna('').ne('').any()
    if has_concepts and can['_concept_label'].nunique() > 1:
        facets.append({
            'name': 'Measure Type',
            'desc': 'Variables grouped by automatically discovered concept label (N×M embedding alignment).',
            'fn':   lambda row: str(row['_concept_label']) if str(row['_concept_label']) not in ('', 'nan') else 'Unclassified',
            'relation': 'related_to',
        })

    # ── Facet: Statistic (detected from description text) ─────────────────────
    _stat_re = re.compile(
        r'\b(mean|average|median|standard deviation|std|percent|proportion|'
        r'probability|total|sum|count|maximum|minimum|range|variance|'
        r'coefficient|ratio|rate|frequency)\b', re.IGNORECASE
    )
    _stat_norm = {
        'average': 'Mean', 'std': 'Standard Deviation', 'proportion': 'Percent',
        'sum': 'Total', 'count': 'Total', 'frequency': 'Rate',
    }
    def _extract_stat(row):
        hits = _stat_re.findall(str(row.get(sem_col, row.get('_text', ''))).lower())
        if not hits: return ''
        h = hits[0].lower()
        return _stat_norm.get(h, h.title())
    stat_col = can.apply(_extract_stat, axis=1)
    if stat_col.ne('').any() and stat_col.nunique() > 1:
        can['_facet_stat'] = stat_col.where(stat_col != '', 'Other')
        facets.append({
            'name': 'Statistic',
            'desc': 'Variables grouped by statistical summary type detected from descriptions.',
            'fn':   (lambda col: lambda row: str(row.get(col, 'Other')))('_facet_stat'),
            'relation': 'is_statistic_of',
        })

    # ── Facet: Condition (numeric code suffix) ─────────────────────────────────
    _num_re = re.compile(r'(\d+)')
    def _extract_cond(row):
        hits = _num_re.findall(str(row['_leaf_label']).split('/')[0].strip())
        return hits[0] if hits else ''
    cond_col = can.apply(_extract_cond, axis=1)
    if cond_col.ne('').any() and cond_col.nunique() > 1:
        can['_facet_cond'] = cond_col.where(cond_col != '', 'No Condition')
        # Name facet from most common unit word in descriptions
        _unit_re  = re.compile(r'\b(second|msec|millisecond|month|week|day|year|trial|block|session|delay)\b', re.IGNORECASE)
        all_text  = ' '.join(can[sem_col].fillna('').astype(str).tolist()).lower()
        unit_hits = _unit_re.findall(all_text)
        fname     = (max(set(unit_hits), key=unit_hits.count).title() + ' Condition') if unit_hits else 'Condition'
        facets.append({
            'name': fname,
            'desc': 'Variables grouped by numeric condition variant in variable codes.',
            'fn':   (lambda col: lambda row: str(row.get(col, 'No Condition')))('_facet_cond'),
            'relation': 'has_condition',
        })

    # ── Facet: Outcome/Error Type (from description keywords) ─────────────────
    _out_re = re.compile(r'\b(error|errors|miss|false alarm|omission|commission|incorrect|outcome|penalty)\b', re.IGNORECASE)
    def _extract_outcome(row):
        hits = _out_re.findall(str(row.get(sem_col, row.get('_text', ''))).lower())
        return hits[0].title() if hits else ''
    out_col = can.apply(_extract_outcome, axis=1)
    if out_col.ne('').any() and out_col.nunique() > 1:
        can['_facet_outcome'] = out_col.where(out_col != '', 'Other')
        facets.append({
            'name': 'Outcome Type',
            'desc': 'Variables grouped by outcome/error type detected from description text.',
            'fn':   (lambda col: lambda row: str(row.get(col, 'Other')))('_facet_outcome'),
            'relation': 'has_measure',
        })

    # ── Facet: Scale/Precision (from _raw metadata columns) ───────────────────
    _prec_re = re.compile(r'\b(decimal|precision|unit|scale|format)\b', re.IGNORECASE)
    if '_raw' in can.columns:
        sample_raw = can['_raw'].dropna().iloc[0] if len(can) > 0 else {}
        prec_cols  = [c for c in (sample_raw.keys() if isinstance(sample_raw, dict) else [])
                      if _prec_re.search(str(c))]
        if prec_cols:
            def _extract_prec(row):
                raw = row.get('_raw', {})
                if not isinstance(raw, dict): return ''
                for pc in prec_cols:
                    v = str(raw.get(pc, '')).strip()
                    if v and v.lower() not in ('nan', 'none', ''): return v
                return ''
            prec_col = can.apply(_extract_prec, axis=1)
            if prec_col.ne('').any() and prec_col.nunique() > 1:
                can['_facet_prec'] = prec_col.where(prec_col != '', 'Unspecified')
                facets.append({
                    'name': 'Scale/Precision',
                    'desc': 'Variables grouped by decimal places or unit of measurement.',
                    'fn':   (lambda col: lambda row: str(row.get(col, 'Unspecified')))('_facet_prec'),
                    'relation': 'belongs_to',
                })

    # ── Facet: Code Family ─────────────────────────────────────────────────────
    has_families = can['_code_family'].fillna('').ne('').any()
    if has_families and can['_code_family'].nunique() > 1:
        facets.append({
            'name': 'Code Family',
            'desc': 'Variables grouped by variable-code structural prefix.',
            'fn':   lambda row: str(row['_code_family']) if str(row['_code_family']) not in ('', 'nan') else 'Other',
            'relation': 'belongs_to',
        })

    # Fallback: TF-IDF semantic clusters if fewer than 2 facets detected
    if len(facets) < 2:
        texts = can[sem_col].fillna('').tolist()
        for nc in [5, 8]:
            lbls    = tfidf_cluster_labels(texts, max_clusters=nc)
            lbl_col = f'_tfidf_cluster_{nc}'
            can[lbl_col] = lbls
            facets.append({
                'name':     f'Semantic Cluster (k={nc})',
                'desc':     f'TF-IDF agglomerative clustering into {nc} groups.',
                'fn':       (lambda col: lambda row: str(row.get(col, 'Other')))(lbl_col),
                'relation': 'related_to',
            })

    return facets

def build_facet_hierarchy(can, facet, project='root'):
    """[CAS] Single-level facet hierarchy: Root → Group → Leaf."""
    nodes     = [{'id': 0, 'name': project, 'type': 'root', 'dtype': 'root',
                  'isShown': True, 'related': [], 'desc': f"Facet: {facet['name']}"}]
    group_ids = {}
    for i, (_, row) in enumerate(can.iterrows(), start=1):
        group_label = str(facet['fn'](row))
        if group_label not in group_ids:
            gid = len(nodes)
            group_ids[group_label] = gid
            nodes.append(make_agg(gid, group_label,
                                   desc=f"{facet['name']}: {group_label}",
                                   relation_type=facet['relation']))
            add_child(nodes, 0, gid)
        lid = len(nodes)
        nodes.append({'id': lid, 'name': str(row['_leaf_label']), 'dtype': str(row['_dtype']),
                      'related': [], 'isShown': True, 'type': 'attribute',
                      'desc': str(row['_text']), 'source_file': str(row['_source_file']),
                      'metadata': {'leaf_id': str(row['_leaf_id']),
                                   'group_path': str(row['_group_path'])}})
        add_child(nodes, group_ids[group_label], lid)
    for n in nodes:
        n['related'] = list(dict.fromkeys([int(x) for x in n.get('related', [])]))
    return nodes

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
RELATION_COLORS = {
    'has_measure': '#4C72B0', 'is_statistic_of': '#DD8452', 'has_condition': '#55A868',
    'part_of': '#C44E52', 'instance_of': '#8172B2', 'subclass_of': '#937860',
    'belongs_to': '#8C8C8C', 'related_to': '#CCB974', '': '#8C8C8C',
}

def _node_color(n):
    if n.get('type') == 'root':      return '#c44e52'
    if n.get('type') == 'attribute': return '#4C72B0'
    if n.get('type') == 'collapsed': return '#bbbbbb'
    return RELATION_COLORS.get(n.get('info', {}).get('relation_type', ''), '#8C8C8C')

def _wrap_hover(text, width=90):
    """Word-wrap plain text into HTML lines for Plotly hover tooltips."""
    text = str(text).replace('<', '&lt;').replace('>', '&gt;')
    words, lines, line, length = text.split(), [], [], 0
    for w in words:
        if length + len(w) + 1 > width and line:
            lines.append(' '.join(line))
            line, length = [w], len(w)
        else:
            line.append(w); length += len(w) + 1
    if line:
        lines.append(' '.join(line))
    return '<br>'.join(lines)

def _rich_hover(n, nodes):
    """Full, word-wrapped hover tooltip (name, provenance, complete description).
    Shared by every view so the treemap and node-link tooltips are as readable as
    the sunburst's — leaves show their full semantic_desc, no truncation."""
    nid = int(n['id']); lc = len(leaf_ids(nodes, nid))
    rel  = n.get('info', {}).get('relation_label', '') if n.get('type') == 'aggregation' else ''
    prov = n.get('concept_provenance', {})
    raw_desc = (n.get('semantic_desc') or n.get('desc', '')) \
               if n.get('type') == 'attribute' else n.get('desc', '')
    desc_html = _wrap_hover(raw_desc)
    alts = ', '.join(prov.get('alternatives', []))
    src  = ', '.join(prov.get('source_evidence', []))
    return (f'<b>{n.get("name","")}</b><br>Type: {n.get("type","")}<br>'
            f'Relation: {rel}<br>Variables: {lc}'
            + (f'<br>Confidence: {prov.get("confidence","")} | Source: {src}'
               f'<br>Alternatives: {alts}' if prov else '')
            + f'<br><br>{desc_html}')

def plot_sunburst(nodes, max_depth=4):
    pm = build_parent_map(nodes)
    ids, labels, parents_, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id']); lc = len(leaf_ids(nodes, nid))
        ids.append(str(nid)); labels.append(str(n.get('name', ''))[:40])
        parents_.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(max(1, lc))
        hover.append(_rich_hover(n, nodes))
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents_, values=values,
                                 branchvalues='total', hovertext=hover, hoverinfo='text',
                                 maxdepth=max_depth, insidetextorientation='radial',
                                 marker=dict(colorscale='Blues', line=dict(width=1, color='white')),
                                 leaf=dict(opacity=0.85)))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10),
                      title='Click sector to drill down — hover for concept provenance')
    return fig

def plot_treemap(nodes):
    pm = build_parent_map(nodes)
    ids, labels, parents_, values, hover = [], [], [], [], []
    for n in nodes:
        nid = int(n['id']); lc = len(leaf_ids(nodes, nid))
        ids.append(str(nid)); labels.append(str(n.get('name', ''))[:40])
        parents_.append('' if nid == 0 else str(pm.get(nid, 0)))
        values.append(max(1, lc))
        hover.append(_rich_hover(n, nodes))
    fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents_, values=values,
                                branchvalues='total', hovertext=hover, hoverinfo='text',
                                textinfo='label+value',
                                marker=dict(colorscale='Blues', line=dict(width=1, color='white'))))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def plot_facets_parallel(facet_trees):
    """[CAS] Side-by-side sunbursts — one per facet dimension."""
    names = list(facet_trees.keys()); nf = len(names)
    if nf == 0: return go.Figure()
    cols  = min(3, nf); rows = (nf + cols - 1) // cols
    specs = [[{'type': 'sunburst'} for _ in range(cols)] for _ in range(rows)]
    fig   = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=names)
    for idx, (fname, fnodes) in enumerate(facet_trees.items()):
        row = idx // cols + 1; col = idx % cols + 1
        pm  = build_parent_map(fnodes)
        ids, labels, parents_, values, hover = [], [], [], [], []
        for n_ in fnodes:
            nid = int(n_['id']); lc = len(leaf_ids(fnodes, nid))
            ids.append(f'{fname}_{nid}'); labels.append(str(n_.get('name', ''))[:28])
            parents_.append('' if nid == 0 else f'{fname}_{pm.get(nid, 0)}')
            values.append(max(1, lc))
            hover.append(f'<b>{n_.get("name","")}</b><br>Variables: {lc}')
        fig.add_trace(go.Sunburst(ids=ids, labels=labels, parents=parents_, values=values,
                                   branchvalues='total', hovertext=hover, hoverinfo='text',
                                   maxdepth=2, leaf=dict(opacity=0.8),
                                   marker=dict(line=dict(width=1, color='white'))),
                      row=row, col=col)
    fig.update_layout(height=420 * rows, margin=dict(l=10, r=10, t=50, b=10),
                      title_text='Castanet Parallel Faceted Hierarchies — same variables, different views',
                      title_font_size=13)
    return fig

def display_graph(nodes, max_depth=4, show_hidden=False):
    m = nmap(nodes); dnodes = {}; edges = []; counter = 10 ** 9
    def rec(nid, depth):
        nonlocal counter
        n = m.get(int(nid))
        if not n: return
        if not show_hidden and n.get('isShown') is False and depth > 0: return
        dnodes[int(nid)] = n
        if depth >= max_depth and n.get('related'):
            counter += 1; cid = counter
            dnodes[cid] = {'id': cid, 'name': f'… {len(leaf_ids(nodes,nid))} variables',
                           'type': 'collapsed', 'dtype': 'determine', 'related': [],
                           'desc': f'Collapsed: {n.get("name")}', 'isShown': True}
            edges.append((int(nid), cid)); return
        for c in n.get('related', []):
            ch = m.get(int(c))
            if not ch: continue
            if not show_hidden and ch.get('isShown') is False: continue
            edges.append((int(nid), int(c))); rec(int(c), depth + 1)
    rec(0, 0); return list(dnodes.values()), edges

def positions(dnodes, edges):
    """
    Reingold-Tilford style layout.
    x = depth × horizontal_scale (breathing room between levels)
    y = subtree-aware vertical placement with 1.8 spacing per leaf
    Aggregation nodes centered over their children's y range.
    """
    H_SCALE  = 3.0   # horizontal gap between depth levels
    V_SPACE  = 1.8   # vertical gap between leaf slots

    children = defaultdict(list)
    for p, c in edges:
        children[p].append(c)

    pos = {}
    counter = {'v': 0}

    def rec(nid, depth):
        ch = children.get(nid, [])
        if not ch:
            # Leaf — assign next vertical slot
            y_pos = counter['v'] * V_SPACE
            counter['v'] += 1
            pos[nid] = (depth * H_SCALE, y_pos)
            return y_pos
        child_ys = [rec(c, depth + 1) for c in ch]
        # Parent centered over children range
        y_pos = float(np.mean(child_ys))
        pos[nid] = (depth * H_SCALE, y_pos)
        return y_pos

    rec(0, 0)
    return pos


def plot_node_link(nodes, max_depth, show_hidden, show_leaf_labels):
    """
    Node-link tree with Reingold-Tilford layout.
    Paper: Taxonomizer recommends Sunburst as primary view for large hierarchies.
    Node-link is supplementary — best for exploring structure at moderate depth.
    """
    dnodes, edges = display_graph(nodes, max_depth, show_hidden)
    pos = positions(dnodes, edges)

    # Edges: elbow-style (horizontal then vertical)
    ex, ey = [], []
    for p, c in edges:
        if p not in pos or c not in pos: continue
        x0, y0 = pos[p]
        x1, y1 = pos[c]
        # Draw: parent → midpoint horizontally → child vertically → child
        xm = (x0 + x1) / 2
        ex += [x0, xm, xm, x1, None]
        ey += [y0, y0, y1, y1, None]
    traces = [go.Scatter(x=ex, y=ey, mode='lines',
                          line=dict(width=1, color='#c8c8c8'),
                          hoverinfo='skip', showlegend=False)]

    # Nodes — split aggregation and leaf into two traces for cleaner rendering
    agg_xs, agg_ys, agg_labels, agg_colors, agg_hover = [], [], [], [], []
    lf_xs,  lf_ys,  lf_labels,  lf_colors,  lf_hover  = [], [], [], [], []

    for n in dnodes:
        nid = int(n['id'])
        if nid not in pos: continue
        x, y   = pos[nid]
        lc     = len(leaf_ids(nodes, nid))
        lab    = n.get('name', str(nid))
        htxt   = _rich_hover(n, nodes)
        col    = _node_color(n)

        if n.get('type') in ('root', 'aggregation'):
            display_lab = (lab + (f' ({lc})' if lc else ''))[:50]
            agg_xs.append(x); agg_ys.append(y)
            agg_labels.append(display_lab)
            agg_colors.append(col); agg_hover.append(htxt)
        else:
            display_lab = lab[:40] if show_leaf_labels else ''
            lf_xs.append(x); lf_ys.append(y)
            lf_labels.append(display_lab)
            lf_colors.append(col); lf_hover.append(htxt)

    if agg_xs:
        traces.append(go.Scatter(
            x=agg_xs, y=agg_ys, mode='markers+text',
            text=agg_labels, textposition='middle right',
            hovertext=agg_hover, hoverinfo='text',
            marker=dict(size=16, color=agg_colors,
                        line=dict(color='white', width=2)),
            showlegend=False
        ))
    if lf_xs:
        traces.append(go.Scatter(
            x=lf_xs, y=lf_ys, mode='markers+text',
            text=lf_labels, textposition='middle right',
            hovertext=lf_hover, hoverinfo='text',
            marker=dict(size=7, color=lf_colors,
                        symbol='circle', opacity=0.75,
                        line=dict(color='white', width=1)),
            showlegend=False
        ))

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
            text='Tip: Sunburst (above) is better for large hierarchies [Taxonomizer]',
            xref='paper', yref='paper', x=0.0, y=1.01,
            showarrow=False, font=dict(size=11, color='grey'),
            align='left'
        )]
    )
    return fig

def semantic_map(can):
    texts  = can['_text'].fillna('').astype(str).tolist()
    labels = can['_leaf_label'].astype(str).tolist()
    groups = can['_group_path'].fillna('Ungrouped').astype(str).apply(lambda x: x.split(' > ')[0])
    X      = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                             max_features=1000).fit_transform(texts)
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(X) if X.shape[1] >= 2 \
             else np.zeros((len(texts), 2))
    fig = go.Figure()
    for g in list(dict.fromkeys(groups)):
        mask = groups == g; idx = np.where(mask.values)[0]
        fig.add_trace(go.Scatter(x=coords[mask, 0], y=coords[mask, 1], mode='markers',
                                  name=str(g), text=[labels[i] for i in idx],
                                  hovertext=[f'<b>{labels[i]}</b><br>{texts[i][:400]}' for i in idx],
                                  hoverinfo='text', marker=dict(size=8, opacity=0.85)))
    fig.update_layout(height=600, plot_bgcolor='white', paper_bgcolor='white')
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT STATE
# ─────────────────────────────────────────────────────────────────────────────
for _key in ['nodes', 'canonical', 'configs', 'embedder', 'facet_trees',
             'hiexpan_report', 'concept_table', 'domain']:
    if _key not in st.session_state:
        st.session_state[_key] = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('1. Input')
    uploads  = st.file_uploader('Upload metadata file(s)',
                                 type=['csv','tsv','txt','xlsx','xls','json','md','markdown'],
                                 accept_multiple_files=True)
    existing = st.file_uploader('Load existing hierarchy JSON', type=['json'])

    st.header('2. Generation')
    project    = st.text_input('Root / project name', value='metadata_project')
    max_rows   = st.slider('Max variables', 10, 3000, 600, 10)
    merge_files = st.checkbox('Merge uploaded files', value=True)
    n_clusters = st.slider('Max clusters per group', 2, 16, 8, 1,
                            help='Maximum number of concept sub-groups per top-level group.')

    st.header('3. Semantic embedder')
    model_choice = st.selectbox('Embedding model',
                                 ['all-MiniLM-L6-v2', 'all-mpnet-base-v2',
                                  'paraphrase-MiniLM-L6-v2', 'TF-IDF (no ST)'],
                                 help='[TAX][GON] Sentence-BERT for dense semantic embeddings.')

    max_concepts  = st.slider('Max candidate concepts', 30, 300, 120, 10,
                               help='How many candidate concepts to extract from metadata text.')
    st.caption('HiExpan refinement runs automatically after every build. '
               'Wikidata / Wikipedia / PubMed activate automatically for biomedical, '
               'cognitive, and neurological domains.')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXISTING HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────
if existing is not None:
    try:
        obj = json.loads(existing.getvalue().decode('utf-8', errors='replace'))
        if isinstance(obj, list):
            st.session_state.nodes = obj
            st.success('Loaded hierarchy JSON.')
        else:
            st.error('Hierarchy JSON must be a list of nodes.')
    except Exception as e:
        st.error(f'Could not load: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if uploads:
    paths    = save_uploads(uploads)
    raw_by   = {}; cfg_by = {}; prof_by = {}

    st.subheader('Step 1 — Inspect files')
    for p in paths:
        try:
            df = load_any(p); warn = False
            if probably_raw(df):
                df = raw_to_metadata(df); warn = True
            cfg, prof = detect_roles(df)
            raw_by[p.name] = df; cfg_by[p.name] = cfg; prof_by[p.name] = prof
            with st.expander(f'📄 {p.name}', expanded=False):
                if warn:
                    st.warning('Looked like raw data — columns converted to metadata rows.')
                st.write(f'Rows: **{len(df):,}**, Columns: **{len(df.columns)}**')
                st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f'Failed to load {p.name}: {e}')

    st.subheader('Step 2 — Confirm column roles')
    configs = {}
    for name, df in raw_by.items():
        with st.expander(f'⚙️ {name}', expanded=True):
            cols = list(df.columns); auto = cfg_by[name]
            c1, c2 = st.columns(2)
            with c1:
                leaf  = st.multiselect('Leaf column(s)', cols,
                                        default=[c for c in auto['leaf_cols'] if c in cols],
                                        key=f'leaf_{name}')
                group = st.multiselect('Group/Task column(s)', cols,
                                        default=[c for c in auto['group_cols'] if c in cols],
                                        key=f'group_{name}')
            with c2:
                text  = st.multiselect('Description column(s)', cols,
                                        default=[c for c in auto['text_cols'] if c in cols],
                                        key=f'text_{name}')
                meta  = st.multiselect('Type/unit column(s)', cols,
                                        default=[c for c in auto['metadata_cols'] if c in cols],
                                        key=f'meta_{name}')
            prev = list(dict.fromkeys(leaf + group + text + meta))
            if prev:
                st.dataframe(df[prev].head(6), use_container_width=True)
            configs[name] = {'leaf_cols': leaf, 'group_cols': group,
                             'text_cols': text, 'metadata_cols': meta}

    if st.button('🌳 Build Approach 1 hierarchy', type='primary'):
        try:
            # ── Step A: Build canonical schemas per file ───────────────────────
            cans = [build_canonical(df.head(max_rows), configs[name], name)
                    for name, df in raw_by.items()]

            # ── Step A.5: Domain check before merging multiple files ──────────
            if merge_files and len(cans) > 1:
                file_domains = [detect_domain(c) for c in cans]
                unique_domains = list(dict.fromkeys(file_domains))
                if len(unique_domains) > 1:
                    names_str = ', '.join(
                        f'{n} → {d}' for n, d in zip(list(raw_by.keys()), file_domains)
                    )
                    st.warning(
                        f'Files appear to be from different domains: {names_str}. '
                        f'Merging may produce a mixed hierarchy. '
                        f'Uncheck **Merge uploaded files** to process separately.'
                    )
                can = pd.concat(cans, ignore_index=True)
            else:
                can = cans[0]
            if len(can) > max_rows:
                can = can.head(max_rows).copy()

            # ── Step B: Code family + acronym expansion ───────────────────────
            with st.spinner('Detecting variable code families and expanding acronyms...'):
                can = cluster_codes_by_prefix(can)
                n_families = can['_code_family'].ne('').sum()
                if n_families > 0:
                    st.info(f'Detected {can["_code_family"].nunique()} code families '
                            f'({n_families} coded variables).')
                code_expansions = expand_variable_codes(can)
                if code_expansions:
                    st.info(f'Identified {len(code_expansions)} acronym/segment expansions.')
                st.session_state['code_expansions'] = code_expansions

            # ── Step B.5: Patch _semantic_text with acronym expansions ──────────
            # [C7][GON][LOB] Description text is the dominant semantic signal.
            # Gonçalves: N×M alignment right-hand side uses concept descriptions, not codes.
            # Lobo: "generate additional context for column names to aid matching."
            # Fix: prepend ONLY when expansion adds genuinely new semantic content.
            # Do NOT prepend if the expansion token is already present in the description
            # (prevents "DMS DMS Correct Latency..." doubling that caused "Dms Dms" labels).
            # [F1] Word-boundary coverage check for _patch_semantic.
            # Previous guard used raw substring match: "DMS Correct Latency SD"
            # not in description even when description has "DMS Correct Latency
            # Standard Deviation" — because "SD" ≠ "Standard Deviation".
            # New check: expansion is "covered" if ≥60% of its non-stop words
            # appear as whole words in the description. If covered, skip prepend.
            _patch_stop = {'the','a','an','is','are','was','to','of','in',
                           'on','at','for','with','by','and','or','as'}
            def _exp_covered(exp_str, base_lower):
                words = [w for w in exp_str.lower().split()
                         if w not in _patch_stop and len(w) > 2]
                if not words:
                    return True
                found = sum(
                    1 for w in words
                    if re.search(r'\b' + re.escape(w) + r'\b', base_lower)
                )
                return found / len(words) >= 0.60

            def _patch_semantic(row, exps):
                base = str(row.get('_semantic_text', row['_text']))
                base_lower = base.lower()
                code = str(row['_leaf_label']).strip().split('/')[0]
                seg_tok = re.compile(r'([A-Z]{2,}|\d+)')
                segments = seg_tok.findall(code)
                new_parts = []
                for s in segments:
                    exp = exps.get(s, {}).get('expansion', '')
                    # Only prepend if expansion adds genuinely new information
                    # [F1] word-boundary check: skip if ≥60% of exp words already present
                    if exp and not _exp_covered(exp, base_lower):
                        new_parts.append(exp)
                if new_parts:
                    return f'{" ".join(new_parts)} {base}'
                # Family expansion — same word-boundary guard
                fam = str(row.get('_code_family', ''))
                if fam and fam in exps:
                    fam_exp = exps[fam].get('expansion', '')
                    if fam_exp and not _exp_covered(fam_exp, base_lower):
                        return f'{fam_exp} {base}'
                return base
            can['_semantic_text'] = can.apply(
                lambda r: _patch_semantic(r, code_expansions), axis=1
            )

            # ── Step C: Load SBERT embedder (always attempt; fallback graceful) ─
            with st.spinner('Loading SBERT embedding model...'):
                model_name = model_choice if model_choice != 'TF-IDF (no ST)' else 'all-MiniLM-L6-v2'
                emb = SemanticEmbedder(model_name=model_name)
                ok, msg = emb.load()   # always attempt SBERT load
                if ok:
                    st.success(f'SBERT loaded: {msg}')
                else:
                    st.warning(f'SBERT unavailable — {msg}. Using TF-IDF+SVD fallback.')
            st.session_state.embedder = emb

            # ── Step D: Detect domain ─────────────────────────────────────────
            domain = detect_domain(can)
            st.session_state.domain = domain
            _bio_domains = ('biomedical', 'cognitive', 'neurological')
            _use_external = domain in _bio_domains
            st.info(f'Detected domain: **{domain}**'
                    + (' — Wikidata / Wikipedia / PubMed activated' if _use_external else ''))

            # ── Step E: Extract candidate concepts from metadata ──────────────
            with st.spinner('Extracting candidate concepts from metadata text...'):
                candidates = extract_candidate_concepts_from_metadata(can, max_concepts=max_concepts)
                st.info(f'Extracted {len(candidates)} candidate concepts from metadata.')

            # ── Step F: Build concept table ────────────────────────────────────
            # Biomedical / cognitive / neurological → enrich via Wikidata + PubMed
            # Wikipedia excluded: too slow for interactive use; Wikidata covers same ground
            # All other domains → local-only (no HTTP calls)
            if _use_external:
                with st.spinner(f'Enriching concept table via Wikidata / PubMed ({domain} domain)...'):
                    pb = st.progress(0)
                    concept_table = retrieve_concept_table(
                        candidates, domain=domain,
                        use_wikidata=True, use_wikipedia=False,
                        use_wordnet=True,  use_pubmed=True,
                        bioportal_key='',
                        progress_cb=lambda x: pb.progress(x),
                        code_expansions=code_expansions,
                    )
                    pb.empty()
                    n_wd = sum(1 for c in concept_table if 'wikidata' in c.get('source', ''))
                    n_pm = sum(1 for c in concept_table if 'pubmed'   in c.get('source', ''))
                    st.success(f'Concept table: {len(concept_table)} entries '
                               f'(Wikidata: {n_wd}, PubMed: {n_pm})')
            else:
                concept_table = [
                    {'label': c['label'], 'full_text': c['label'],
                     'source': c.get('source', 'metadata_tfidf'),
                     'frequency': c.get('frequency', 0),
                     'tfidf_score': c.get('tfidf_score', 0.0),
                     'broader_relations': []}
                    for c in candidates
                ]
                st.success(f'Concept table: {len(concept_table)} entries (local metadata — no external calls)')
            st.session_state.concept_table = concept_table

            # ── Step F.5: Fit shared vector space for TF-IDF fallback ─────────
            # CRITICAL: must encode variables + concepts in the SAME space for
            # N×M cosine similarity to be valid. No-op when SBERT is active.
            with st.spinner('Fitting shared embedding space...'):
                var_texts     = can['_semantic_text'].fillna('').astype(str).tolist()
                concept_texts = [c['full_text'] for c in concept_table]
                emb.fit_joint(var_texts + concept_texts)

            # ── Step F.6: Pre-compute Statistic and Condition facets ─────────
            # [F3][F5][CAS] These columns are needed inside _cluster_and_label
            # for facet sub-splitting. They must be computed BEFORE Step G.
            # detect_facets / build_castanet_facets runs AFTER hierarchy build
            # (Step I), so we pre-compute only _facet_cond here. The statistic tier
            # is produced data-drivenly later by _nest_by_measure (no hardcoded vocab).
            with st.spinner('Pre-computing Condition facets [CAS]...'):
                can = precompute_stat_cond_facets(can)
                n_cond  = can['_facet_cond'].ne('').sum()
                st.info(f'Facet pre-computation: {n_cond} variables with Condition. '
                        f'Statistic depth is derived from concept titles (_nest_by_measure).')

            # ── Step G: Build concept hierarchy (N×M alignment) ──────────────
            with st.spinner('Building concept hierarchy via N×M alignment [GON][TAX]...'):
                nodes = build_concept_hierarchy(
                    can, emb, concept_table,
                    project=project,
                    n_clusters_per_group=n_clusters,
                )

            # ── Step H: HiExpan refinement (always automatic) ─────────────────
            with st.spinner('Running HiExpan refinement [HIE]...'):
                if concept_table:
                    c_embs = emb.encode([c['full_text'] for c in concept_table])
                else:
                    c_embs = None
                nodes, report = run_hiexpan(nodes, can, emb, concept_table, c_embs)
                st.session_state.hiexpan_report = report
                wmoves = report.get('width_expansion_moves', 0)
                dexp   = report.get('depth_expansion_nodes', 0)
                gmoves = report.get('global_optimization_moves', 0)
                st.success(f'HiExpan complete — width moves: {wmoves}, '
                           f'depth expansions: {dexp}, global moves: {gmoves}')

            # ── Step I: Castanet facets ───────────────────────────────────────
            with st.spinner('Building Castanet parallel facets [CAS]...'):
                facets      = detect_facets(can, code_expansions=code_expansions)
                facet_trees = {f['name']: build_facet_hierarchy(can, f, project)
                               for f in facets}
                st.session_state.facet_trees = facet_trees

            # ── Step J: Evaluation metrics ────────────────────────────────────
            n_total   = len(can)
            n_aligned = can['_concept_score'].gt(0.08).sum()
            n_lowconf = can['_concept_score'].between(0, 0.25, inclusive='right').sum()
            n_fallbk  = can['_concept_score'].eq(0.0).sum()
            n_family  = can['_code_family'].ne('').sum()
            avg_conf  = float(can['_concept_score'].mean())
            cov_pct   = round(100 * n_aligned / max(n_total, 1), 1)
            st.session_state['eval_metrics'] = {
                'total_variables':        n_total,
                'alignment_coverage_%':   cov_pct,
                'avg_label_confidence':   round(avg_conf, 3),
                'fallback_count':         int(n_fallbk),
                'fallback_rate_%':        round(100 * n_fallbk / max(n_total, 1), 1),
                'low_confidence_count':   int(n_lowconf),
                'variables_with_family':  int(n_family),
                'code_family_%':          round(100 * n_family / max(n_total, 1), 1),
                'concept_table_size':     len(concept_table),
                'wikidata_entries':       sum(1 for c in concept_table if 'wikidata' in c.get('source', '')),
                'pubmed_entries':         sum(1 for c in concept_table if 'pubmed'   in c.get('source', '')),
                'acronym_expansions':     len(code_expansions),
                'hiexpan_width_moves':    wmoves,
                'hiexpan_depth_exp':      dexp,
                'hiexpan_global_moves':   gmoves,
            }

            # ── Build concept-label provenance DataFrame (4th export) ──────────
            prov_rows_build = []
            for _n in nodes:
                if _n.get('type') == 'aggregation' and _n.get('concept_provenance'):
                    _p = _n['concept_provenance']
                    prov_rows_build.append({
                        'Node':          _n['name'],
                        'Confidence':    _p.get('confidence', ''),
                        'Source':        ', '.join(_p.get('source_evidence', [])),
                        'Embedding sim': _p.get('embedding_sim', ''),
                        'Alternatives':  ', '.join(_p.get('alternatives', [])[:3]),
                    })
            prov_df = pd.DataFrame(prov_rows_build) if prov_rows_build else pd.DataFrame()

            st.session_state.canonical   = can
            st.session_state.configs     = configs
            st.session_state.nodes       = nodes
            st.session_state['prov_df']  = prov_df


            n_l = len([n for n in nodes if n.get('type') == 'attribute'])
            n_i = len([n for n in nodes if n.get('type') == 'aggregation'])
            st.success(f'Built: {n_l} variables, {n_i} internal nodes, '
                       f'{len(facets)} facets — {cov_pct}% concept-aligned.')

        except Exception as e:
            st.error(f'Build failed: {e}')
            import traceback; st.code(traceback.format_exc())

if st.session_state.nodes is None:
    st.info('Upload a metadata file and click **Build Approach 1 hierarchy**.')
    st.stop()

nodes         = st.session_state.nodes
can           = st.session_state.canonical
facet_trees   = st.session_state.facet_trees or {}
hiexpan_report = st.session_state.hiexpan_report or {}
concept_table = st.session_state.concept_table or []

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(['🌳 LoD tree', '🔲 Faceted view', '🧬 HiExpan report',
                '⚠️ Conflicts', '✏️ Edit', '🔍 Search',
                '🗺️ Semantic map', '📋 Metadata', '⬇️ Export', 'ℹ️ Method',
                '📊 Evaluation'])

# ── Tab 0: LoD tree ───────────────────────────────────────────────────────────
with tabs[0]:
    # ── Visualization controls (above chart — easy to find) ───────────────────
    vc1, vc2, vc3, vc4 = st.columns([2, 2, 1, 1])
    with vc1:
        viz_mode = st.radio('View mode',
                            ['Sunburst (drill-down)', 'Treemap', 'Node-link tree'],
                            horizontal=True, index=0,
                            help='Sunburst recommended for large hierarchies [Taxonomizer]. Node-link best for exploring structure at smaller depth.')
    with vc2:
        depth = st.slider('Depth (Level of Detail)', 1, 8, 3, 1)
    with vc3:
        show_leaf_labels = st.checkbox('Leaf labels', value=False)
    with vc4:
        show_hidden = st.checkbox('Hidden nodes', value=False)
    st.divider()

    if viz_mode == 'Sunburst (drill-down)':
        st.caption('Hover for concept provenance (confidence, source, alternatives). Click to drill down.')
        st.plotly_chart(plot_sunburst(nodes, depth), use_container_width=True)
    elif viz_mode == 'Treemap':
        st.plotly_chart(plot_treemap(nodes), use_container_width=True)
    else:
        st.plotly_chart(plot_node_link(nodes, depth, show_hidden, show_leaf_labels),
                        use_container_width=True)
    pr  = path_rows(nodes)
    max_d = max((r['depth'] for r in pr), default=0)
    c1, c2, c3 = st.columns(3)
    c1.metric('Variables',      len([n for n in nodes if n.get('type') == 'attribute']))
    c2.metric('Internal nodes', len([n for n in nodes if n.get('type') == 'aggregation']))
    c3.metric('Max depth',      max_d)
    emb_ = st.session_state.embedder
    if emb_:
        st.caption(f'Embedding backend: **{emb_.backend}** | Domain: **{st.session_state.domain or "unknown"}**')

    # Code expansion table
    code_exp = st.session_state.get('code_expansions', {})
    if code_exp:
        with st.expander(f'Acronym / code segment expansions ({len(code_exp)} found)', expanded=False):
            exp_rows = [{'Segment': seg, 'Expansion': v['expansion'],
                         'Evidence': ', '.join(v['evidence'])}
                        for seg, v in code_exp.items()]
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True)

    # Concept label provenance for internal nodes
    prov_rows = []
    for n in nodes:
        if n.get('type') == 'aggregation' and n.get('concept_provenance'):
            p = n['concept_provenance']
            prov_rows.append({
                'Node': n['name'],
                'Confidence': p.get('confidence', ''),
                'Source': ', '.join(p.get('source_evidence', [])),
                'Embedding sim': p.get('embedding_sim', ''),
                'Alternatives': ', '.join(p.get('alternatives', [])[:3]),
            })
    if prov_rows:
        with st.expander('Concept label provenance for internal nodes', expanded=False):
            st.dataframe(pd.DataFrame(prov_rows), use_container_width=True)

# ── Tab 1: Faceted view ───────────────────────────────────────────────────────
with tabs[1]:
    st.subheader('Castanet Parallel Faceted Hierarchies')
    st.markdown(
        '**[CAS]** Each sunburst organises the same variables by a different dimension. '
        'Concept facet uses automatically assigned labels from embedding alignment.'
    )
    if facet_trees:
        st.plotly_chart(plot_facets_parallel(facet_trees), use_container_width=True)
        st.markdown('### Per-facet detail')
        sel_facet = st.selectbox('Inspect facet tree', list(facet_trees.keys()))
        ft = facet_trees[sel_facet]
        st.plotly_chart(plot_sunburst(ft, max_depth=3), use_container_width=True)
        n_groups = len([n for n in ft if n.get('type') == 'aggregation'])
        st.info(f'Facet **{sel_facet}**: {n_groups} groups, '
                f'{len([n for n in ft if n.get("type")=="attribute"])} variables')
    else:
        st.info('Build the hierarchy first to see faceted views.')

# ── Tab 2: HiExpan report ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader('HiExpan Refinement Report')
    if hiexpan_report:
        c1, c2, c3 = st.columns(3)
        c1.metric('Width expansion moves',   hiexpan_report.get('width_expansion_moves', 0))
        c2.metric('Depth expansion nodes',   hiexpan_report.get('depth_expansion_nodes', 0))
        c3.metric('Global optimization moves', hiexpan_report.get('global_optimization_moves', 0))
        st.markdown('### Sibling coherence — before refinement (worst first)')
        before = hiexpan_report.get('coherence_before', [])
        if before:
            st.dataframe(pd.DataFrame(before), use_container_width=True)
        st.markdown('### Sibling coherence — after refinement')
        after = hiexpan_report.get('coherence_after', [])
        if after:
            st.dataframe(pd.DataFrame(after), use_container_width=True)
            b_mean = np.mean([r['coherence_score'] for r in before]) if before else float('nan')
            a_mean = np.mean([r['coherence_score'] for r in after])
            st.metric('Mean coherence improvement',
                       f'{a_mean:.3f}', delta=f'{a_mean - b_mean:+.3f}')
    else:
        st.info('HiExpan runs automatically. Build the hierarchy to see results.')

    # ── Evaluation metrics ─────────────────────────────────────────────────────
    ev = st.session_state.get('eval_metrics', {})
    if ev:
        st.markdown('---')
        st.subheader('Evaluation Metrics')
        st.markdown(
            'These metrics help evaluate how well the automatic concept alignment worked. '
            'For thesis evaluation, compare against a manually curated hierarchy.'
        )
        ea, eb, ec, ed = st.columns(4)
        ea.metric('Alignment coverage', f'{ev.get("alignment_coverage_%", 0)}%',
                  help='% of variables with concept score > 0.08 (non-fallback)')
        eb.metric('Avg label confidence', f'{ev.get("avg_label_confidence", 0):.3f}',
                  help='Mean concept score across all variables (0–1)')
        ec.metric('Low-confidence placements', ev.get('low_confidence_count', 0),
                  help='Variables with concept score 0–0.25 (review in Conflicts tab)')
        ed.metric('Fallback rate', f'{ev.get("fallback_rate_%", 0)}%',
                  help='% of variables that got score=0 (no concept matched above threshold)')
        e2a, e2b, e2c = st.columns(3)
        e2a.metric('Concept table size', ev.get('concept_table_size', 0))
        e2b.metric('Variables w/ code family', f'{ev.get("code_family_%", 0)}%')
        e2c.metric('Acronym expansions', ev.get('acronym_expansions', 0))
        st.caption(
            '**Thesis interpretation:** Alignment coverage > 70% indicates the concept table '
            'adequately covers the metadata domain. Fallback rate > 30% suggests the domain vocabulary '
            'is sparse — try a biomedical/cognitive dataset to activate Wikidata + PubMed enrichment. '
            'Low-confidence > 20% suggests HiExpan depth expansion created subclusters with ambiguous boundaries.'
        )

# ── Tab 3: Conflict resolution ────────────────────────────────────────────────
with tabs[3]:
    st.subheader('Conflict Resolution — Low-Confidence Placements')
    st.markdown(
        'Variables with concept assignment confidence < 0.25 may belong to multiple groups. '
        'Review and use the **Edit** tab to move them if needed.'
    )
    if can is not None:
        conflict_df = compute_conflict_table(can, nodes)
        if len(conflict_df):
            st.dataframe(conflict_df, use_container_width=True)
        else:
            st.success('No low-confidence placements detected.')
    else:
        st.info('Build the hierarchy first.')

# ── Tab 4: Edit ───────────────────────────────────────────────────────────────
with tabs[4]:
    rows_ = path_rows(nodes); choice_to_id = {r['choice']: r['id'] for r in rows_}
    selected = st.selectbox('Select node', list(choice_to_id.keys()))
    sid = choice_to_id[selected]; node = get_node(nodes, sid)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('### Node properties')
        new_name  = st.text_input('Name', value=node.get('name', ''), key=f'nm{sid}')
        new_desc  = st.text_area('Description', value=node.get('desc', ''), key=f'ds{sid}', height=100)
        dtype_vals = ['root', 'number', 'string', 'determine', 'unknown']
        cur = node.get('dtype', 'determine')
        new_dtype = st.selectbox('Data type', dtype_vals,
                                  index=dtype_vals.index(cur) if cur in dtype_vals else 3,
                                  key=f'dt{sid}')
        new_shown = st.checkbox('Shown', value=bool(node.get('isShown', True)), key=f'sh{sid}')
        if node.get('type') == 'aggregation':
            rel_opts = list(RELATION_TYPES.keys())
            cur_rel  = node.get('info', {}).get('relation_type', 'belongs_to')
            new_rel  = st.selectbox('Relation type', rel_opts,
                                     index=rel_opts.index(cur_rel) if cur_rel in rel_opts else 0,
                                     format_func=lambda k: f'{k} — {RELATION_TYPES[k]}',
                                     key=f'rel{sid}')
        if node.get('concept_provenance'):
            prov = node['concept_provenance']
            st.markdown('**Concept provenance:**')
            st.json(prov)
        if st.button('Save changes'):
            info = dict(node.get('info', {}))
            if node.get('type') == 'aggregation':
                info['relation_type']  = new_rel
                info['relation_label'] = RELATION_TYPES.get(new_rel, '')
            st.session_state.nodes = update_node(nodes, sid, name=new_name, desc=new_desc,
                                                  dtype=new_dtype, isShown=new_shown, info=info)
            st.rerun()
    with c2:
        st.markdown('### Move / add / delete')
        if node.get('type') in ['root', 'aggregation']:
            with st.form('add_grp'):
                cname = st.text_input('New child name', value='New Group')
                crel  = st.selectbox('Relation type', list(RELATION_TYPES.keys()))
                cdesc = st.text_area('Description', value='')
                if st.form_submit_button('Add child'):
                    nid_ = next_id(nodes)
                    nodes.append(make_agg(nid_, cname, desc=cdesc, relation_type=crel))
                    add_child(nodes, sid, nid_)
                    st.session_state.nodes = nodes; st.rerun()
        pops = [o for o in agg_opts(nodes, True) if o['id'] != sid]
        if sid != 0 and pops:
            tgt     = st.selectbox('Move under', [o['label'] for o in pops])
            tgt_id  = next(o['id'] for o in pops if o['label'] == tgt)
            if st.button('Move node'):
                st.session_state.nodes = move_node(nodes, sid, tgt_id); st.rerun()
        if node.get('type') == 'aggregation':
            rea = st.checkbox('Reattach children when deleting', value=True)
            if st.button('Delete aggregation'):
                st.session_state.nodes = delete_agg(nodes, sid, rea); st.rerun()
    st.markdown('### Children')
    cns = [get_node(nodes, c) for c in node.get('related', [])]
    st.dataframe(pd.DataFrame([{'id': c.get('id'), 'name': c.get('name'),
                                 'type': c.get('type'),
                                 'relation': c.get('info', {}).get('relation_label', ''),
                                 'desc': str(c.get('desc', ''))[:120]}
                                for c in cns if c]), use_container_width=True)

# ── Tab 5: Search ─────────────────────────────────────────────────────────────
with tabs[5]:
    q = st.text_input('Search name, description, relation, type')
    out_ = []
    for n in nodes:
        hay = ' '.join([str(n.get(k, '')) for k in ['name', 'desc', 'dtype', 'type']]
                       + [n.get('info', {}).get('relation_label', '')]).lower()
        if not q or q.lower() in hay:
            out_.append({'id': n.get('id'), 'name': n.get('name'), 'type': n.get('type'),
                         'relation': n.get('info', {}).get('relation_label', ''),
                         'n_children': len(n.get('related', [])),
                         'desc': str(n.get('desc', ''))[:200]})
    st.dataframe(pd.DataFrame(out_), use_container_width=True)

# ── Tab 6: Semantic map ───────────────────────────────────────────────────────
with tabs[6]:
    if can is None or len(can) < 3:
        st.info('Semantic map available after build.')
    else:
        st.plotly_chart(semantic_map(can), use_container_width=True)

# ── Tab 7: Metadata ───────────────────────────────────────────────────────────
with tabs[7]:
    if can is None:
        st.info('Available after build.')
    else:
        show_cols = [c for c in can.columns if c != '_raw']
        st.dataframe(can[show_cols], use_container_width=True)

# ── Tab 8: Export ─────────────────────────────────────────────────────────────
with tabs[8]:
    # Name downloads after the uploaded CSV (fall back to the project name).
    if uploads:
        _base = safe_name(Path(uploads[0].name).stem)
    else:
        _base = safe_name(project)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            'Hierarchy JSON',
            data=json.dumps(nodes, indent=2, ensure_ascii=False).encode('utf-8'),
            file_name=f'{_base}_approach1_hierarchy.json',
            mime='application/json',
            use_container_width=True,
        )
    with col2:
        if facet_trees:
            st.download_button(
                'Facets JSON',
                data=json.dumps(facet_trees, indent=2, ensure_ascii=False).encode('utf-8'),
                file_name=f'{_base}_approach1_facets.json',
                mime='application/json',
                use_container_width=True,
            )

    col3, col4 = st.columns(2)
    with col3:
        if can is not None:
            st.download_button(
                'Canonical CSV',
                data=can.drop(columns=['_raw'], errors='ignore').to_csv(index=False).encode('utf-8'),
                file_name=f'{_base}_approach1_canonical.csv',
                mime='text/csv',
                use_container_width=True,
            )
    with col4:
        _prov_df = st.session_state.get('prov_df', pd.DataFrame())
        if not _prov_df.empty:
            st.download_button(
                'Concept labels CSV',
                data=_prov_df.to_csv(index=False).encode('utf-8'),
                file_name=f'{_base}_approach1_concept_labels.csv',
                mime='text/csv',
                use_container_width=True,
            )

    st.divider()
    # ── Save directly into the project's outputs/approach_1/ folder ────────────
    _out_dir = Path(__file__).resolve().parent / 'outputs' / 'approach_1'
    st.markdown('### Save to project folder')
    st.caption(
        'The download buttons above go to your browser’s Downloads folder (a browser '
        f'restriction). This button instead writes the files into `{_out_dir}` with the '
        'dataset name — convenient for `evaluate_all.py`.'
    )
    if st.button('💾 Save all to outputs/approach_1/', type='primary',
                 use_container_width=True):
        try:
            _out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            (_out_dir / f'{_base}_approach1_hierarchy.json').write_text(
                json.dumps(nodes, indent=2, ensure_ascii=False), encoding='utf-8')
            saved.append(f'{_base}_approach1_hierarchy.json')
            if facet_trees:
                (_out_dir / f'{_base}_approach1_facets.json').write_text(
                    json.dumps(facet_trees, indent=2, ensure_ascii=False), encoding='utf-8')
                saved.append(f'{_base}_approach1_facets.json')
            if can is not None:
                can.drop(columns=['_raw'], errors='ignore').to_csv(
                    _out_dir / f'{_base}_approach1_canonical.csv', index=False)
                saved.append(f'{_base}_approach1_canonical.csv')
            _prov_df2 = st.session_state.get('prov_df', pd.DataFrame())
            if not _prov_df2.empty:
                _prov_df2.to_csv(_out_dir / f'{_base}_approach1_concept_labels.csv', index=False)
                saved.append(f'{_base}_approach1_concept_labels.csv')
            st.success(f'Saved to `{_out_dir}`:\n\n- ' + '\n- '.join(saved))
        except Exception as _e:
            st.error(f'Could not save: {_e}')

    st.divider()
    st.markdown('### Hierarchy preview (first 5 nodes)')
    st.json(nodes[:5])

# ── Tab 9: Method ─────────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown(f"""
## Method — Approach 1

### Algorithm (no hardcoded domain labels)

| Step | What happens | Paper |
|---|---|---|
| 1. Canonical schema | Every metadata file → unified `_text` object | [GON] |
| 2. Code family detection | Variable-code structural prefix clustering (DMSL*, SWMBE*) | [GON] |
| 3. Domain detection | Auto-detect biomedical / cognitive / finance / environment / general | — |
| 4. Candidate concept extraction | TF-IDF n-grams + noun phrases + group-path terms from **input data only** | [GON][TAX] |
| 5. External concept table | Wikidata + WordNet + Wikipedia + PubMed + BioPortal → concept TABLE | [GON][TAX][HIE] |
| 6. Concept embedding | SBERT encodes variables + concept table entries | [TAX][GON] |
| 7. N×M cosine similarity | Variables × concepts: embedding + string + frequency + source score | [GON] |
| 8. Concept label assignment | Best-scoring concept label + confidence + alternatives + provenance | [GON] |
| 9. Hierarchy construction | Task/group-first backbone + automatic concept sub-groups | [TAX][HIE] |
| 10. HiExpan refinement | Sibling coherence, width expansion, depth expansion (embedding-based), global opt | [HIE] |
| 11. Castanet facets | Concept · Task · Code family · Data type parallel views | [CAS] |

**Detected domain:** `{st.session_state.domain or 'not yet detected'}`
**Concept table size:** `{len(concept_table)} entries`
**WordNet available:** `{'yes' if _WORDNET_AVAILABLE else 'no — run: pip install nltk'}`

### Why no hardcoded patterns?

The previous version used regex lists (`MEASURE_PATTERNS`, `STAT_PATTERNS`, `CONDITION_PATTERNS`)
to label hierarchy nodes. These only worked for the AI-Mind neuropsychology dataset.

This version discovers concept labels **automatically**:
- Extracts candidate terms from **whatever metadata text the user provides**
- Validates them against **universal external knowledge** (Wikidata, WordNet, Wikipedia)
- Selects the best label by **embedding cosine similarity** — the Gonçalves N×M alignment step

For an AI-Mind dataset, the pipeline will discover "correct latency", "standard deviation",
"0 second delay" — because those phrases appear in the data. For a climate dataset,
it will discover "temperature anomaly", "precipitation rate" — again, from the data.
No domain vocabulary is assumed or hardcoded.

### Scoring formula (per cluster) [GON]

```
score(cluster, concept) =
  0.50 × SBERT cosine similarity (cluster centroid ↔ concept embedding)
+ 0.20 × word overlap (cluster description words ∩ concept label words)
+ 0.15 × frequency (concept appears in N metadata rows / max frequency)
+ 0.10 × source confidence (Wikidata=0.88, WordNet=0.83, Wikipedia=0.78, ...)
+ 0.05 × hierarchy evidence (concept has P31/P279/P361 relations in Wikidata)
```

### External sources

| Source | Domain | What it provides |
|---|---|---|
| **Wikidata** (always) | Any | Structured descriptions, P31/P279/P361 broader relations |
| **WordNet** (default) | Any | Definitions, hypernyms, synonyms |
| **Wikipedia** (optional) | Any | Full text definitions |
| **PubMed** (optional) | Biomedical/Cognitive | Abstract text for domain embeddings |
| **BioPortal** (optional, key) | Biomedical | Ontology class labels and definitions |
""")

# ── Tab 10: Evaluation ─────────────────────────────────────────────────────────
with tabs[10]:
    import hierarchy_eval as he

    st.subheader('Hierarchy Quality Evaluation')

    can_eval   = st.session_state.get('canonical', pd.DataFrame())
    nodes_eval = st.session_state.get('nodes', [])

    if can_eval.empty or not nodes_eval:
        st.info('Build a hierarchy first — metrics appear here after the build completes.')
    else:
        st.caption(
            'The group column is a *construction input* (Gonçalves text object + concept '
            'alignment), so it cannot be ground truth. The primary metrics below are '
            '**reference-free** — they assess the hierarchy itself, no gold standard.'
        )

        with st.spinner('Computing reference-free metrics…'):
            tm   = he.traco_metrics(nodes_eval)
            npmi = he.npmi_coherence(nodes_eval, can_eval['_text'].tolist())

        # ── PRIMARY: reference-free hierarchy quality ─────────────────────────
        st.markdown('#### Primary — reference-free hierarchy quality')
        p1, p2, p3 = st.columns(3)
        p1.metric('Parent–child coherence', tm['pc_coherence'],
                  help='TraCo (Wu et al., AAAI 2024). Children correctly nest under parent theme.')
        p2.metric('Sibling diversity', tm['sibling_diversity'],
                  help='TraCo (Wu et al., AAAI 2024). Higher = distinct siblings; LOW = redundant.')
        p3.metric('NPMI label coherence', npmi,
                  help='Lau et al., EACL 2014. Label terms genuinely co-occur in the data.')
        st.caption(f'Embedding backend: **{tm["encoder"]}**.')

        # ── Approach-1-specific alignment metrics ─────────────────────────────
        em = st.session_state.get('eval_metrics', {})
        if em:
            st.markdown('#### Concept-alignment metrics  (Approach-1 specific)')
            a1, a2, a3 = st.columns(3)
            a1.metric('Alignment coverage', f"{em.get('alignment_coverage_pct', 0):.1f}%",
                      help='% of variables assigned an external concept label')
            a2.metric('Avg label confidence', f"{em.get('avg_label_confidence', 0):.3f}",
                      help='Mean concept-alignment cosine score across all variables')
            a3.metric('Fallback rate', f"{em.get('fallback_rate_pct', 0):.1f}%",
                      help='% of variables that used TF-IDF fallback instead of external concept')

        # ── Structural statistics ─────────────────────────────────────────────
        st.markdown('#### Structural statistics')
        sm = he.structural_stats(nodes_eval)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric('Aggregation nodes', sm['n_aggregation_nodes'])
        s2.metric('Max leaf depth',    sm['max_depth'])
        s3.metric('Avg leaf depth',    sm['avg_leaf_depth'])
        s4.metric('Avg branching',     sm['avg_branching_factor'])
        s5.metric('Singleton nodes',   f"{sm['singleton_nodes_%']}%")

        # ── SECONDARY: group preservation (caveated) ──────────────────────────
        st.markdown('#### Secondary — group-structure preservation *(descriptive)*')
        st.caption(
            '⚠️ The group column was an **input** to construction, so these are NOT accuracy '
            'metrics — only how much the hierarchy still reflects the pre-existing group column.'
        )
        gp = he.group_preservation(nodes_eval, can_eval)
        g1, g2, g3 = st.columns(3)
        g1.metric('NMI', gp['NMI']);  g2.metric('ARI', gp['ARI']);  g3.metric('Purity', gp['Purity'])
