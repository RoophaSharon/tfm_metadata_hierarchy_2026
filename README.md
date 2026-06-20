---
title: Metadata Hierarchy Explorer
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: demo.py
pinned: false
license: mit
---

# Metadata Hierarchy Construction — TFM

Master's thesis prototype: automatic hierarchy construction from data-dictionary metadata.
Three algorithms are implemented for comparison.

## Live demo

The deployed app opens on a **pre-built results viewer** (`demo.py`) showing the
AI-MIND and HCP hierarchies for all three approaches — no upload needed. Use the
sidebar to switch approach/dataset and the Level-of-Detail controls to adjust depth.

To **build a hierarchy from your own CSV**, open the **Baseline**, **Approach 1**, or
**Approach 2** page from the left sidebar and upload a file. (Approach 2's optional
local-LLM label refinement runs only on a local machine with Ollama; in the cloud it
falls back to the deterministic pipeline automatically.)

## Approaches

- **Baseline** — Pure clustering baseline. Plain TF-IDF / Word2Vec embeddings + hierarchical
  clustering. Documented in `README_baseline.md`.

- **Approach 1** — Global embedding pipeline. Uses SBERT + N×M concept-table alignment
  (Gonçalves 2019) + HiExpan refinement (Shen et al. KDD 2018) + Castanet parallel facets.
  Optionally retrieves concept context from Wikidata / Wikipedia / WordNet / BioPortal.

- **Approach 2** — Dataset-constrained multi-aspect hierarchy. Algorithmic pipeline with no
  domain hardcoding:
  1. Group-anchored L1/L2 (from detected metadata column structure — BISE 2026)
  2. Phrase-slot mining (IE-style slot induction) for descriptions with regular structure
  3. **FASTopic** semantic aspect discovery (Wu et al. NeurIPS 2024) — replaces NMF
  4. NMF lexical fallback for small groups
  5. GMM + BIC for small clusters, MiniBatchKMeans + silhouette for large ones
  6. Deterministic 5-stage label generation (description prefix → group anchor → IDF filter
     → bigram-preferred TF-IDF → optional LLM refinement)
  7. **Optional local-LLM label refinement** via Ollama + Qwen 2.5 (TopicTag pattern, DocEng
     2024). Strict grounding check rejects labels not derived from CSV evidence. Per-node
     provenance recorded.
  8. TraCo-inspired hierarchy diagnostics (AAAI 2024)

  No facet trees — single coherent LoD tree.

See each script's "Method" tab in the running app for the full algorithm and paper references.

## Paper stack

| Component | Paper |
|---|---|
| Multi-aspect taxonomy scaffold | Zhu et al. 2025, EMNLP |
| Canonical metadata text objects | Gonçalves et al. 2019, ESWC |
| Semantic aspect discovery | Wu et al. 2024 (FASTopic), NeurIPS, arXiv:2405.17978 |
| Phrase-slot mining | IE / slot-induction literature (ACM CSUR 2022) |
| LLM label refinement pattern | Eren et al. 2024 (TopicTag), DocEng, arXiv:2407.19616 |
| Local LLM (used for refinement) | Qwen Team 2024 (Qwen 2.5), arXiv:2412.15115 |
| Hierarchy quality diagnostics | Wu et al. 2024 (TraCo), AAAI, arXiv:2401.14113 |
| Group-anchored entry strategy | Motamedi, Novalija, Rei 2026, Springer BISE |
| Multidimensional taxonomy motivation | Kargupta et al. 2025 (TaxoAdapt), ACL |
| Future-work semantic consistency | SC-Taxo 2026, arXiv:2605.00620 |
| Concept-label evaluation framework | Kejriwal et al. 2022 (TICL), EAAI |

## Project layout

```
Hierarchy tool/
├── baseline.py          # Pure clustering baseline (Streamlit app)
├── approach_1.py        # Approach 1 (Streamlit app)
├── approach_2.py        # Approach 2 (Streamlit app)
├── approach_1.ipynb     # Approach 1 reproducible notebook
├── approach_2.ipynb     # Approach 2 reproducible notebook
├── baseline.ipynb       # Baseline reproducible notebook
├── launcher.py          # Run all three apps simultaneously on different ports
├── data/                # Sample input CSVs (AI-MIND, HCP, etc.)
├── outputs/             # Generated hierarchies (JSON)
└── requirements.txt
```

## Running locally

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or 3.11 recommended.

### 2. (Approach 2 only) Install Ollama for the local-LLM label refinement layer

**This is optional — Approach 2 produces deterministic labels without it.**  If you want
the optional TopicTag-style LLM label refinement:

1. Download and install Ollama from https://ollama.com/download
2. Open Ollama once so the background service starts (icon in the system tray)
3. Pull the recommended model:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```
   (For higher quality at higher RAM cost: `ollama pull qwen2.5:7b-instruct`.)
4. Verify the server is reachable:
   - In a browser open `http://localhost:11434/api/tags`
   - Or run `ollama list`

When Approach 2 starts it auto-detects Ollama and the "Refine labels with LLM" checkbox
defaults to ON. Uncheck any time. The deterministic pipeline is the canonical thesis
result; the LLM is an optional re-phraser of evidence already in the CSV.

To override the default URL or model:

```bash
# Optional environment variables
set OLLAMA_URL=http://localhost:11434/v1
set OLLAMA_MODEL=qwen2.5:3b-instruct
```

Or change them live in the Approach 2 sidebar.

### 3. Run one app at a time

```bash
streamlit run baseline.py
# or
streamlit run approach_1.py
# or
streamlit run approach_2.py
```

Each opens at http://localhost:8501 by default.

### 4. Run all three apps simultaneously (for side-by-side comparison)

```bash
python launcher.py
```

This opens three browser tabs:

- http://localhost:8501 — Baseline
- http://localhost:8502 — Approach 1
- http://localhost:8503 — Approach 2

Press **Enter** in the launcher terminal to stop all servers.

## Using the apps

1. Upload one or more metadata CSV / TSV / XLSX / JSON files in the sidebar.
2. Confirm the auto-detected column roles (leaf / group / text / meta).
3. Click **Build hierarchy**.
4. Inspect the LoD tree, evaluation metrics, label provenance (Approach 2), and export JSON.

Sample data is in `data/`:
- `ai-mind-variable-descriptions(in).csv`
- `HCP_S1200_DataDictionary_Oct_30_2023.csv`

## Outputs

- **Baseline / Approach 1** export two JSON files compatible with the VIANNA viewer:
  - `*_lod.json` — primary LoD tree
  - `*_facets.json` — parallel Castanet facet trees

- **Approach 2** exports a single LoD JSON:
  - `*_approach2_lod.json` — primary LoD tree (every aggregation node carries
    `label_provenance` with source stage, confidence, and evidence terms)

Filenames are derived from the uploaded CSV file name, so different CSVs export under
different filenames into `outputs/approach 2/`.

Existing output examples are in `outputs/approach 1/` and `outputs/approach 2/`.

## Defensibility highlights for Approach 2

- **No domain hardcoding.** Slot names, group anchors, and labels are all derived from the
  detected metadata columns + the uploaded CSV — no hand-curated domain vocabulary.
- **Deterministic by default.** Tree topology and all five label-generation stages are
  reproducible from the input CSV alone. Local LLM is opt-in.
- **Grounded LLM refinement.** Every LLM-proposed label must pass a strict grounding
  check — every word in the label must appear in the extracted evidence. Failed proposals
  are rejected and the deterministic label is used instead. Per-node provenance lets
  you answer "did the LLM invent this?" with hard evidence.
- **Local-only LLM.** Qwen 2.5 runs on the thesis machine via Ollama. No external API
  calls, no third-party data sharing, no key management.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FASTopic not installed` warning | `pip install fastopic` (also installs `torch`) |
| `openai` package missing | `pip install openai` |
| `Ollama not reachable` in sidebar | Open the Ollama app from Start menu; the service runs in the system tray |
| Model not found | `ollama pull qwen2.5:3b-instruct` |
| Build very slow with LLM on | Expected for HCP — ~15–40 min on CPU with a 3B model. Disable LLM for fast iteration. |
| `LLM-labeled nodes: 0/N` after build | The grounding check rejected every LLM proposal. Check the **🔍 Label Provenance** tab — counts under `llm_rejected = True` show what happened. |
| Hierarchy too shallow | Increase `Max LoD tree depth` slider (top of sidebar in Approach 2) |

## License

For thesis evaluation only.
