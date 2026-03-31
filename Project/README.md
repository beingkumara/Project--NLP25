# NLP Assignment (Parts 2, 3, and 4)

This repository contains the code for Parts 2, 3, and 4 of the NLP assignment. The core implementation scripts are located in the `template_code_part2/template_code_part2/` directory.

## Dependencies

Make sure you have Python 3 installed. The code depends on the following external packages:

- `nltk`
- `spacy`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install nltk spacy matplotlib
```

Additionally, you need to download the required NLTK data and the spaCy language model:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

> **Note:** The `venv/` folder has been intentionally excluded from this repository. Please create your own virtual environment if desired before installing the dependencies.

## How to Run the Code

First, open your terminal and navigate to the directory containing the project scripts:

```bash
cd template_code_part2/template_code_part2/
```

### Part 2: Main Search Engine Pipeline

To run the complete Information Retrieval evaluation pipeline on the provided Cranfield dataset:

```bash
python main.py
```

**Optional Arguments for `main.py`:**
- `-dataset`: Path to the dataset folder (default: `cranfield/`)
- `-out_folder`: Path to save outputs (default: `output/`)
- `-segmenter`: Choose either `naive` or `punkt` (default: `punkt`)
- `-tokenizer`: Choose either `naive` or `ptb` (default: `ptb`)
- `-custom`: Run the search engine with a custom interactive query instead of the full dataset.

### Part 3: Metrics Evaluation

To compute Precision, Recall, F-score, MAP, and nDCG for \(k = 1\) to \(10\) on the Cranfield dataset, and to generate the evaluation plot (saved as `output/eval_plot_part3.png`):

```bash
python part3.py
```

### Part 4: VSM Flaw Analysis & OOV Testing

**Vector Space Model (VSM) Flaw Analysis:**  
To find an empirical example demonstrating a flaw in the VSM (where a highly relevant document is given a rank worse than 100):

```bash
python find_vsm_flaw.py
```

**Out-Of-Vocabulary (OOV) Testing:**  
To evaluate how the system handles queries with partially known and completely unknown (OOV) terms:

```bash
python test_oov.py
```

### Outputs
Running the scripts will automatically create an `output/` directory (if it doesn't already exist) where intermediary files, calculated metric JSON files, and evaluation graphs will be saved.
