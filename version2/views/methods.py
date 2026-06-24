"""
methods.py — single source of truth for method naming, descriptions and display
config, shared by the Demo View (viewer.py) and the Build pages (run_*.py).

Metadata Hierarchy Explorer — TFM 2026.

The internal keys ("Baseline" / "Approach 1" / "Approach 2") are kept stable on
purpose: the pre-built output filenames and the thesis cross-references depend on
them. The user-facing *title* is what gets shown in the app.
"""
from __future__ import annotations

METHOD_ORDER = ["Baseline", "Approach 1", "Approach 2"]

METHODS: dict[str, dict] = {
    "Baseline": {
        "title": "Baseline: Taxonomizer Semantic Space Hierarchy",
        "tag":   "Baseline · Word2Vec semantic space + agglomerative clustering "
                 "(Mahmood & Mueller, IEEE TVCG 2019)",
        "color":     "Greens",
        "compress":  False,
        "node_link": True,
        "about": (
            "Classical clustering baseline. Word2Vec skip-gram embeddings of the "
            "attribute names build a cosine semantic space, then balanced Ward "
            "agglomerative clustering produces the tree; node labels are the most "
            "discriminative terms per cluster. No external knowledge bases and no "
            "neural language models — a deliberately simple reference point."
        ),
    },
    "Approach 1": {
        "title": "Approach 1: External Concept Alignment Hierarchy",
        "tag":   "Approach 1 · SBERT + Gonçalves N×M alignment + HiExpan + Castanet facets",
        "color":     "Blues",
        "compress":  False,
        "node_link": True,
        "about": (
            "Aligns each variable to concepts drawn from external knowledge bases. "
            "SBERT embeddings and an N×M concept-similarity matrix (Gonçalves 2019) "
            "match variables to candidate concepts retrieved from Wikidata, Wikipedia, "
            "WordNet and BioPortal; HiExpan refines the tree and Castanet builds "
            "parallel facets. External enrichment activates automatically for "
            "biomedical, cognitive and neurological domains."
        ),
    },
    "Approach 2": {
        "title": "Approach 2: Dataset Constrained Multi Aspect Hierarchy",
        "tag":   "Approach 2 · FASTopic + phrase-slot mining (Wu et al. NeurIPS 2024)",
        "color":     "Viridis",
        "compress":  True,
        "node_link": True,
        "about": (
            "Builds the hierarchy using only evidence inside the dataset — no external "
            "knowledge. Group structure anchors the top levels, phrase-slot mining and "
            "FASTopic (Wu et al. 2024) discover semantic aspects, and per-aspect "
            "clustering forms the branches. Labels are generated deterministically and "
            "are fully auditable; an optional local LLM may re-phrase them under a "
            "strict grounding check."
        ),
    },
}

# Reverse lookup: display title -> internal key.
TITLE_TO_KEY = {m["title"]: k for k, m in METHODS.items()}
TITLES = [METHODS[k]["title"] for k in METHOD_ORDER]


def title(key: str) -> str:
    return METHODS[key]["title"]


def tag(key: str) -> str:
    return METHODS[key]["tag"]


def about(key: str) -> str:
    return METHODS[key]["about"]
