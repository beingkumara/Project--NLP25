# NLP Project — Information Retrieval Engine (Cranfield Dataset)

This project implements and evaluates multiple Information Retrieval (IR) methods on the Cranfield dataset, including TF-IDF VSM, BM25, LSA, Query Expansion, Pseudo-Relevance Feedback (PRF), and Explicit Semantic Analysis (ESA). It includes a full NLP preprocessing pipeline and rigorous statistical evaluation.

---

## Project Structure

```
Project--NLP25/
├── README.md
├── Project_Report.pdf
└── Project/
    └── template_code_part2/
        └── template_code_part2/          ← ALL commands must be run from here
            ├── cranfield/                 ← Dataset (queries, docs, qrels)
            │   ├── cran_docs.json
            │   ├── cran_queries.json
            │   └── cran_qrels.json
            ├── output/                    ← Generated plots and JSON results
            ├── main.py                    ← Main search engine entry point
            ├── part3.py                   ← Baseline VSM evaluation
            ├── part5.py                   ← Multi-method comparison + stats
            ├── test_oov.py                ← Out-of-vocabulary handling test
            ├── find_vsm_flaw.py           ← VSM failure case analysis
            ├── query_class_analysis.py    ← Query classification analysis
            ├── evaluation.py
            ├── informationRetrieval.py
            ├── informationRetrievalBM25.py
            ├── informationRetrievalLSA.py
            ├── informationRetrievalPRF.py
            ├── informationRetrievalESA.py
            ├── informationRetrievalESACranfield.py
            ├── queryExpansion.py
            ├── sentenceSegmentation.py
            ├── tokenization.py
            ├── inflectionReduction.py
            ├── stopwordRemoval.py
            └── util.py
```

---

## Step 1 — Prerequisites

- **Python 3.8 or higher** is required. Check your version:
  ```bash
  python --version
  ```
  If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).

---

## Step 2 — Create a Virtual Environment

Navigate to the working directory first — **all commands in this README must be run from this directory**:

```bash
cd "Project/template_code_part2/template_code_part2"
```

Create the virtual environment:

```bash
python -m venv nlp_env
```

This creates a folder called `nlp_env/` inside the current directory.

---

## Step 3 — Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
nlp_env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
nlp_env\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source nlp_env/bin/activate
```

After activation, your terminal prompt will show `(nlp_env)` at the start, confirming the environment is active.

> To deactivate the virtual environment at any time, run:
> ```bash
> deactivate
> ```

---

## Step 4 — Install Python Dependencies

With the virtual environment **active**, install all required packages:

```bash
pip install numpy scipy scikit-learn matplotlib nltk spacy
```

**What each package is used for:**

| Package | Purpose |
|---|---|
| `numpy` | Matrix operations for LSA (SVD decomposition) |
| `scipy` | Wilcoxon signed-rank statistical tests in part5.py |
| `scikit-learn` | TruncatedSVD for Latent Semantic Analysis |
| `matplotlib` | Generating evaluation plots (PNG files) |
| `nltk` | Tokenization, stemming, lemmatization, stopwords, WordNet |
| `spacy` | Sentence segmentation using the spaCy NLP pipeline |

---

## Step 5 — Download NLTK Data

Some NLTK components require additional data downloads. Run this **once** after installing packages:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

**What each NLTK resource is used for:**

| Resource | Used by |
|---|---|
| `punkt` / `punkt_tab` | Punkt sentence tokenizer in `sentenceSegmentation.py` |
| `wordnet` | WordNet lemmatizer and query expansion via synonyms |
| `stopwords` | Stopword removal in `stopwordRemoval.py` |
| `averaged_perceptron_tagger` / `_eng` | POS tagging for WordNet lemmatization |

---

## Step 6 — Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

This installs the small English model used by `SentenceSegmentation.spacySegmenter()`.

---

## Step 7 — Run the Main Search Engine (main.py)

Make sure you are in `Project/template_code_part2/template_code_part2/` with the virtual environment active.

### Full Dataset Evaluation (default mode)

Runs the TF-IDF VSM search engine on all 225 Cranfield queries and evaluates with Precision, Recall, F-score, MAP, nDCG, and MRR at k=1–10:

```bash
python main.py
```

Outputs to the `output/` folder:
- `output/eval_plot.png` — line chart of all metrics across k values

### Custom Interactive Query Mode

Enter your own query and get the top 5 ranked documents:

```bash
python main.py -custom
```

The engine will prompt you to type a query, then print ranked results with title, author, and a text snippet.

### Optional Arguments

| Argument | Default | Options | Description |
|---|---|---|---|
| `-segmenter` | `punkt` | `naive`, `punkt` | Sentence segmentation method |
| `-tokenizer` | `ptb` | `naive`, `ptb` | Tokenization method |
| `-dataset` | `cranfield/` | any path | Path to the Cranfield dataset folder |
| `-out_folder` | `output/` | any path | Folder where plots are saved |

**Example with custom arguments:**
```bash
python main.py -segmenter punkt -tokenizer ptb
```

---

## Step 8 — Run the Evaluation Files

All evaluation scripts must be run from the `template_code_part2/template_code_part2/` directory with the virtual environment active.

---

### part3.py — Baseline TF-IDF VSM Evaluation

Evaluates the baseline TF-IDF Vector Space Model and computes all IR metrics (Precision, Recall, F-score, MAP, nDCG, MRR) at k=1–10.

```bash
python part3.py
```

**Outputs to `output/`:**
- `eval_plot_part3.png` — metric curves across k values
- `metrics_part3.json` — all numeric metric values in JSON format

**Expected runtime:** ~30–60 seconds

---

### part5.py — Multi-Method Comparison with Statistical Analysis

This is the most comprehensive evaluation script. It benchmarks **6 IR methods** against each other:

1. VSM (TF-IDF baseline)
2. BM25 (with 5-fold cross-validation for hyperparameters k1 and b)
3. LSA (ablation study over n_components = 50, 100, 200, 300)
4. VSM + WordNet Query Expansion
5. BM25 + Pseudo-Relevance Feedback (PRF)
6. ESA with WordNet concept space
7. ESA with Cranfield concept space

It also performs:
- Wilcoxon signed-rank hypothesis testing (α = 0.05) between all method pairs
- Cohen's d effect size calculation
- Per-query Average Precision analysis
- Failure mode analysis (queries where all methods fail)
- Query classification by length and density

```bash
python part5.py
```

**Outputs to `output/`:**

| File | Description |
|---|---|
| `part5_final_metrics_k10.json` | MAP, nDCG, MRR at k=10 for all methods |
| `part5_wilcoxon_tests.json` | Wilcoxon p-values and Cohen's d between all method pairs |
| `part5_lsa_ablation.json` | LSA MAP scores for each n_components setting |
| `part5_per_query_analysis.json` | Per-query AP for each method |
| `part5_failure_analysis.json` | Queries where all methods fail |
| `part5_case_studies.json` | Qualitative case studies of best/worst queries |
| `part5_graded_ndcg_k10.json` | Graded nDCG values (relevance grades 1–2) |
| `part5_full_per_query_ap.json` | Full per-query AP breakdown |
| `part5_comparison_map_ndcg.png` | Bar chart: MAP and nDCG comparison |
| `part5_final_bar_chart.png` | Final metric comparison bar chart |
| `part5_lsa_ablation.png` | LSA ablation study plot |
| `part5_length_class_chart.png` | Performance by query length class |
| `part5_density_class_chart.png` | Performance by query density class |

**Expected runtime:** ~5–15 minutes (BM25 cross-validation and LSA ablation are the slowest steps)

---

### test_oov.py — Out-of-Vocabulary Handling Test

Tests how the IR system handles queries containing unknown/made-up words (words not present in the document collection). Runs two queries:
- A query with valid known terms
- A query with nonsense OOV words (e.g., "flippity floppity")

```bash
python test_oov.py
```

**Output:** Printed to terminal — ranked results for each query, showing how the system gracefully handles zero-match terms without crashing.

**Expected runtime:** ~30–60 seconds

---

### find_vsm_flaw.py — VSM Failure Case Analysis

Identifies queries where the baseline TF-IDF VSM completely fails — specifically cases where a known-relevant document is ranked beyond position 100 (i.e., effectively not retrieved).

```bash
python find_vsm_flaw.py
```

**Output:** Printed to terminal — displays the query text and the body of each missed relevant document, revealing the vocabulary mismatch or other structural failure causing the low rank.

**Expected runtime:** ~30–60 seconds

---

## Full Run Order (Recommended)

To reproduce the complete project pipeline from scratch, run these commands in order:

```bash
# 1. Navigate to the working directory
cd "Project/template_code_part2/template_code_part2"

# 2. Create and activate virtual environment
python -m venv nlp_env
nlp_env\Scripts\Activate.ps1        # Windows PowerShell
# source nlp_env/bin/activate       # macOS/Linux

# 3. Install all dependencies
pip install numpy scipy scikit-learn matplotlib nltk spacy

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# 5. Download spaCy model
python -m spacy download en_core_web_sm

# 6. Run main search engine (baseline evaluation + eval_plot.png)
python main.py

# 7. Run Part 3 baseline evaluation
python part3.py

# 8. Run Part 5 full multi-method comparison (takes several minutes)
python part5.py

# 9. Run OOV test
python test_oov.py

# 10. Run VSM flaw analysis
python find_vsm_flaw.py
```

All generated plots and JSON results will be in the `output/` folder.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'X'`**
Make sure the virtual environment is activated (you should see `(nlp_env)` in your prompt) and re-run `pip install numpy scipy scikit-learn matplotlib nltk spacy`.

**`LookupError: Resource punkt not found`**
Re-run the NLTK download command from Step 5.

**`OSError: [E050] Can't find model 'en_core_web_sm'`**
Re-run `python -m spacy download en_core_web_sm` with the virtual environment active.

**`FileNotFoundError` for cranfield JSON files**
Make sure you are running all scripts from inside `Project/template_code_part2/template_code_part2/`, not from the repo root.

**Scripts hang or are very slow**
`part5.py` is legitimately slow due to cross-validation and LSA ablation. Allow up to 15 minutes. All other scripts should finish in under 2 minutes.
