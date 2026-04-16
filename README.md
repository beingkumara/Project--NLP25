# NLP Information Retrieval Project

This project implements a comprehensive Information Retrieval (IR) system designed to process, index, and search through the **Cranfield dataset**. It explores various IR models, from basic vector space models to advanced semantic search techniques, and provides rigorous evaluation metrics.

## What this Project Does (Simple Terms)
At its heart, this project is a specialized search engine for scientific documents. It takes a collection of research papers (the Cranfield dataset) and a set of queries, then calculates which papers are most relevant to each query. 

It implements several "ranking" strategies:
- **Baseline (TF-IDF)**: Matches words exactly between the query and documents.
- **BM25**: A more advanced matching algorithm that better handles term frequency.
- **LSA (Latent Semantic Analysis)**: Uses math to find "hidden" meanings, allowing it to match documents even if they don't share the exact same words as the query.
- **ESA (Explicit Semantic Analysis)**: Compares queries and documents against external concepts (like WordNet) to understand semantic intent.
- **Query Expansion & PRF**: Automatically adds related words to your search to improve results.

The system then compares these methods to see which one performs best using standard metrics like Precision (how many results were correct) and Recall (how many correct results were found).

---

## Getting Started

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
Once the virtual environment is active, install the required Python packages:

```bash
pip install nltk spacy matplotlib numpy scipy scikit-learn
```

Additionally, you need to download the necessary NLP models and data:

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK datasets
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

---

## How to Run the Files

All core execution scripts are located in the project's source directory. Navigate there first:

```bash
cd template_code_part2/template_code_part2/
```

### Main Evaluation (Part 3)
Computes Precision, Recall, F-score, MAP, and nDCG for \(k = 1\) to \(10\). It generates an evaluation plot in the `output/` folder.
```bash
python part3.py
```

### Advanced Comparative Analysis (Part 5)
Runs a full comparison between the Baseline, BM25, LSA, ESA, and Pseudo-Relevance Feedback models. It includes:
- **Ablation Studies**: Tuning hyperparameters like the number of components in LSA.
- **Statistical Testing**: Uses Wilcoxon Signed-Rank tests to prove if improvements are mathematically significant.
```bash
python part5.py
```

### Out-Of-Vocabulary (OOV) Testing
Evaluates how the search engine handles queries containing words it has never seen before.
```bash
python test_oov.py
```

### Interactive Search
To run the main search pipeline or use custom interactive queries: - Will fail, need to write the main file properly
```bash
python main.py
```

### Outputs
Standard results, JSON data, and comparison graphs are automatically saved to the `output/` directory within the source folder.
