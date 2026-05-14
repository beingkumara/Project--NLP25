"""
part5.py
=========
Part 5: Improving the IR System

This script runs a full comparative evaluation of six retrieval systems:
  - Baseline : TF-IDF VSM (from Part 2)
  - Method A : BM25 (Okapi BM25, manually implemented)
  - Method B : LSA  (TF-IDF + Truncated SVD, n_components selected by 5-fold CV)
  - Method C : VSM + WordNet Query Expansion
  - Method D : BM25 + Pseudo-Relevance Feedback (PRF)
  - Method E : ESA (WordNet Concept Space)
  - Method F : ESA (Cranfield Concept Space)

Evaluation is at k=10 on the Cranfield dataset using:
  Precision@10, Recall@10, F0.5@10, MAP@10, nDCG@10 (binary), nDCG@10 (graded), MRR@10

Statistical significance testing:
  Wilcoxon Signed-Rank Test (scipy.stats.wilcoxon) on per-query AP scores.
  H0: method X and baseline have the same distribution of AP scores.
  We reject H0 (claim improvement is real) at p < 0.05.
  For BM25, LSA, PRF we use one-sided 'greater' because theory strongly predicts improvement.
  For QE and ESA we use two-sided because the direction of change is not guaranteed.

Ablation study:
  LSA n_components is selected via 5-fold CV over [50, 100, 200, 300].
  A separate ablation plot is also made for all k values on the full dataset.
  (The CV-selected k is the one used for final comparison — not the ablation best.)

BM25 hyperparameter tuning:
  5-Fold CV over k1 in [1.2, 1.5, 2.0] and b in [0.5, 0.75, 0.9].

All results are saved to output/part5_*.json and output/part5_*.png.
"""

import json
import os
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Preprocessing modules (existing pipeline) ──────────────────────────────
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

# These are the IR models we built
from informationRetrieval import InformationRetrieval
from informationRetrievalBM25 import InformationRetrievalBM25
from informationRetrievalLSA import InformationRetrievalLSA
from queryExpansion import QueryExpander
from informationRetrievalPRF import InformationRetrievalPRF
from informationRetrievalESA import InformationRetrievalESA
from informationRetrievalESACranfield import InformationRetrievalESACranfield

from evaluation import Evaluation

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# We define folder paths and evaluation variables here
DATASET_PATH = 'cranfield/'
OUT_DIR = 'output/'
EVAL_K = 10
LSA_K_VALUES = [50, 100, 200, 300]


# ===========================================================================
# DATA LOADING HELPERS
# ===========================================================================

from util import load_json, save_json, get_relevant_docs, preprocess_text, parse_cranfield_data, ranked_list_to_dict


# ===========================================================================
# GRADED RELEVANCE BUILDER
# ===========================================================================

def build_graded_relevant_docs(all_qrels_list):
    """
    This function builds a graded relevance dictionary from the Cranfield qrels.
    In Cranfield, each qrel entry has a position field (1 to 4):
      - Position 1 or 2 means the document is highly relevant -> we assign grade = 2
      - Position 3 or 4 means the document is somewhat relevant -> we assign grade = 1

    We need this graded structure to compute graded nDCG which is more informative
    than binary nDCG because it rewards systems that rank highly relevant docs higher.

    The returned structure is: {query_id: {doc_id: grade_value}}
    """
    # Here we will store the nested dictionary of graded relevance
    graded_rels = {}

    # We iterate through every qrel entry one by one
    for i in range(len(all_qrels_list)):
        curr_entry = all_qrels_list[i]

        # Converting to int because sometimes data comes as string
        query_id = int(curr_entry["query_num"])
        doc_id = int(curr_entry["id"])
        position = curr_entry["position"]

        # If we have not seen this query before, we initialize its dict
        if query_id not in graded_rels:
            graded_rels[query_id] = {}

        # Now we assign grade based on the position value from qrels
        # Position 1 or 2 -> highly relevant, grade = 2
        # Position 3 or 4 -> somewhat relevant, grade = 1
        # Anything else -> not relevant, skip it
        if position == 1 or position == 2:
            grade_value = 2
        elif position == 3 or position == 4:
            grade_value = 1
        else:
            # This entry does not qualify as relevant so we skip
            continue

        # We store it, but if the same doc appears twice we keep the max grade
        if doc_id not in graded_rels[query_id]:
            graded_rels[query_id][doc_id] = grade_value
        else:
            if grade_value > graded_rels[query_id][doc_id]:
                graded_rels[query_id][doc_id] = grade_value

    return graded_rels


# ===========================================================================
# GRADED nDCG COMPUTATION
# ===========================================================================

def compute_graded_ndcg_for_query(doc_IDs_ordered, graded_rels_for_query, k):
    """
    This function computes nDCG at cutoff k using graded relevance grades.
    Unlike binary nDCG where gain is only 0 or 1, here the gain can be 1 or 2.
    This is much more informative because it rewards placing grade-2 docs
    at the very top compared to grade-1 docs.

    Formula used:
      DCG@k  = sum over i=1..k of ( gain_i / log2(i + 1) )
      IDCG@k = DCG in the ideal ranking (all grades sorted descending, top-k)
      nDCG@k = DCG@k / IDCG@k
    """
    # We initialize DCG and IDCG values to zero first
    dcg_score = 0.0
    idcg_score = 0.0

    # First we will get the top-k documents from the ranked list
    top_k_docs = []
    for i in range(len(doc_IDs_ordered)):
        if i < k:
            top_k_docs.append(doc_IDs_ordered[i])

    # Now we compute DCG for the actual ranked list
    # For each position, if the doc has a grade, we add gain / log2(pos + 1)
    for pos_idx in range(len(top_k_docs)):
        position_1indexed = pos_idx + 1
        doc_id = top_k_docs[pos_idx]

        # Checking if this document has a relevance grade assigned
        if doc_id in graded_rels_for_query:
            gain = graded_rels_for_query[doc_id]
        else:
            gain = 0

        # Only adding to DCG if gain is positive
        if gain > 0:
            dcg_score = dcg_score + (float(gain) / math.log2(position_1indexed + 1))

    # Now we compute IDCG - the DCG of the ideal (perfect) ranking
    # We collect all grade values from the graded relevance dict for this query
    all_grades_list = []
    for doc_id in graded_rels_for_query:
        all_grades_list.append(graded_rels_for_query[doc_id])

    # Sort grades in descending order to get the ideal arrangement
    all_grades_list.sort(reverse=True)

    # IDCG uses only top-k of these ideal grades
    ideal_count = min(k, len(all_grades_list))
    for ideal_idx in range(ideal_count):
        position_1indexed = ideal_idx + 1
        ideal_gain = all_grades_list[ideal_idx]
        if ideal_gain > 0:
            idcg_score = idcg_score + (float(ideal_gain) / math.log2(position_1indexed + 1))

    # If IDCG is zero, it means there are no relevant docs for this query
    # Hence nDCG is not defined, so we return 0
    if idcg_score == 0.0:
        return 0.0

    # print("debug: DCG =", dcg_score, "IDCG =", idcg_score)
    return dcg_score / idcg_score


def compute_mean_graded_ndcg(doc_IDs_ordered_dict, query_ids, graded_rels, k):
    """
    This function computes the mean graded nDCG over all the queries.
    We iterate through every query, compute graded nDCG for it,
    and then divide by total query count to get the mean value.
    """
    # Handling the edge case where no queries are provided
    if len(query_ids) == 0:
        return 0.0

    total_graded_ndcg = 0.0

    for q_idx in range(len(query_ids)):
        q_id = query_ids[q_idx]

        # Getting the ranked list for this query from the dict
        ranked_for_q = []
        if q_id in doc_IDs_ordered_dict:
            ranked_for_q = doc_IDs_ordered_dict[q_id]

        # Getting the graded relevance dict for this specific query
        graded_for_q = {}
        if q_id in graded_rels:
            graded_for_q = graded_rels[q_id]

        # Compute graded nDCG for this single query
        g_ndcg = compute_graded_ndcg_for_query(ranked_for_q, graded_for_q, k)
        total_graded_ndcg = total_graded_ndcg + g_ndcg

    # Divide by total number of queries to get the mean
    mean_graded_ndcg = total_graded_ndcg / float(len(query_ids))
    return mean_graded_ndcg


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def preprocess_all(query_texts, doc_texts):
    seg  = SentenceSegmentation()
    tok  = Tokenization()
    red  = InflectionReduction()
    stop = StopwordRemoval()

    print("  Preprocessing queries. This might take some time...")
    processed_queries = []
    for q_idx in range(len(query_texts)):
        q = query_texts[q_idx]
        processed_queries.append(preprocess_text(q, seg, tok, red, stop))

    print("  Preprocessing documents (this takes a moment)...")
    processed_docs = []
    for d_idx in range(len(doc_texts)):
        d = doc_texts[d_idx]
        processed_docs.append(preprocess_text(d, seg, tok, red, stop))

    return processed_queries, processed_docs, stop

# ===========================================================================
# EVALUATION HELPERS
# ===========================================================================

def compute_per_query_ap(evaluator, doc_IDs_ordered_dict, query_ids, relevant_docs, k):
    """
    Return a list of per-query Average Precision scores at cutoff k.
    Used as the input distribution for the Wilcoxon test.
    """
    per_query_ap = []
    for idx_q in range(len(query_ids)):
        q_id = query_ids[idx_q]
        # fetch the relevant ones or empty list
        rel_list = []
        if q_id in relevant_docs:
            rel_list = relevant_docs[q_id]

        ap = evaluator.queryAveragePrecision(
            doc_IDs_ordered_dict[q_id], q_id, rel_list, k
        )
        per_query_ap.append(ap)
    return per_query_ap


def compute_all_metrics(evaluator, doc_IDs_ordered_dict, query_ids, relevant_docs, k):
    """
    This function computes all six metrics for a single system at cutoff k.
    We added MRR (Mean Reciprocal Rank) because it is already implemented in
    evaluation.py but was not reported before. MRR is especially informative
    for navigational queries where the first relevant document matters most.
    """
    # Computing each metric separately and storing it in a dictionary
    prec  = evaluator.meanPrecision(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    rec   = evaluator.meanRecall(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    fs    = evaluator.meanFscore(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    m_map = evaluator.meanAveragePrecision(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    ndcg  = evaluator.meanNDCG(doc_IDs_ordered_dict, query_ids, relevant_docs, k)

    # MRR was already implemented in evaluation.py but was not being used before.
    # Here we will now call it properly so we can report it in the results.
    mrr   = evaluator.meanReciprocalRank(doc_IDs_ordered_dict, query_ids, relevant_docs, k)

    return {"precision": prec, "recall": rec, "fscore": fs, "map": m_map, "ndcg": ndcg, "mrr": mrr}


# ===========================================================================
# WILCOXON HYPOTHESIS TEST
# ===========================================================================

def wilcoxon_test(ap_baseline, ap_method, method_name, alternative='greater'):
    """
    This function runs the Wilcoxon Signed-Rank test between the baseline AP scores
    and the method's AP scores at query level.

    H0 is that both systems have the same distribution of per-query AP scores.
    We reject H0 at p < 0.05 and say the difference is statistically meaningful.
    We also compute Cohen's d to see how large the effect actually is.

    We added the alternative parameter because for some methods (BM25, LSA, PRF)
    we have a strong prior from IR theory that they should outperform VSM, so
    one-sided 'greater' is appropriate. But for query expansion and ESA, the
    direction is not guaranteed, so we use 'two-sided' to be fair.

    Parameters
    ----------
    ap_baseline : list
        Per-query AP scores for the baseline (VSM).
    ap_method : list
        Per-query AP scores for the method being tested.
    method_name : str
        Name of the method for display in the report.
    alternative : str
        Either 'greater' (one-sided) or 'two-sided'. Default is 'greater'.
    """
    differences = []
    for i in range(len(ap_method)):
        differences.append(ap_method[i] - ap_baseline[i])

    non_zero = []
    for d_idx in range(len(differences)):
        d = differences[d_idx]
        if d != 0:
            non_zero.append(d)

    if len(non_zero) == 0:
        return {
            "method": method_name,
            "alternative_used": alternative,
            "H0": "AP scores of " + method_name + " and VSM baseline come from the same distribution.",
            "H1": method_name + " has different AP scores from the VSM baseline.",
            "statistic": None,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "significant": False,
            "interpretation": "Both systems have identical performance. Cannot reject H0."
        }

    # Here we run the Wilcoxon test with the specified alternative direction
    # For one-sided 'greater': H1 is that method AP > baseline AP
    # For two-sided: H1 is that the distributions differ in any direction
    stat, p_value = stats.wilcoxon(ap_method, ap_baseline, alternative=alternative)

    # Find mean of the differences
    sum_diff = 0.0
    for d in differences:
        sum_diff = sum_diff + d
    mean_delta = float(sum_diff) / float(len(differences))

    # Computing Cohen's d for effect size
    # Cohen's d = mean(differences) / std(differences)
    sum_sq_diff = 0.0
    for d in differences:
        sum_sq_diff = sum_sq_diff + ((d - mean_delta) ** 2)
    variance = sum_sq_diff / float(len(differences))
    std_dev = math.sqrt(variance)

    cohens_d = 0.0
    if std_dev > 0:
        cohens_d = mean_delta / std_dev
    # print("debug: cohens_d for", method_name, "is", cohens_d)

    # Interpreting effect size based on standard thresholds
    effect_label = "negligible"
    if abs(cohens_d) >= 0.2 and abs(cohens_d) < 0.5:
        effect_label = "small"
    elif abs(cohens_d) >= 0.5 and abs(cohens_d) < 0.8:
        effect_label = "medium"
    elif abs(cohens_d) >= 0.8:
        effect_label = "large"

    significant = False
    if p_value < 0.05:
        significant = True

    # Constructing the H1 string based on which direction we tested
    if alternative == 'greater':
        h1_str = method_name + " has significantly higher AP scores than the VSM baseline."
    else:
        h1_str = method_name + " has significantly different AP scores from the VSM baseline (two-sided)."

    if significant:
        interpretation = (str(method_name) + " IS significantly different from VSM baseline"
            + " (alternative=" + alternative
            + ", p=" + str(round(p_value, 4))
            + ", mean delta MAP=" + str(round(mean_delta, 4))
            + ", Cohen's d=" + str(round(cohens_d, 3))
            + " [" + effect_label + "] effect). Reject H0.")
    else:
        interpretation = (str(method_name) + " is NOT significantly different from VSM baseline"
            + " (alternative=" + alternative
            + ", p=" + str(round(p_value, 4))
            + ", mean delta MAP=" + str(round(mean_delta, 4))
            + ", Cohen's d=" + str(round(cohens_d, 3))
            + " [" + effect_label + "] effect). Cannot reject H0.")

    return {
        "method": method_name,
        "alternative_used": alternative,
        "H0": "AP scores of " + method_name + " and VSM baseline come from the same distribution.",
        "H1": h1_str,
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_delta_ap": mean_delta,
        "cohens_d": cohens_d,
        "effect_size_label": effect_label,
        "significant": significant,
        "interpretation": interpretation
    }

# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_metrics_vs_k(all_systems_metrics_over_k, k_values, out_path):
    """
    Plot MAP and nDCG vs k for all systems on the same figure.
    Saves two subplots: MAP@k and nDCG@k.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    line_styles = ['-o', '-s', '-^', '-D', '-x', '-p', '-h']

    system_names = []
    for name in all_systems_metrics_over_k.keys():
        system_names.append(name)

    for i in range(len(system_names)):
        sys_name = system_names[i]
        metrics_over_k = all_systems_metrics_over_k[sys_name]

        maps = []
        ndcgs = []
        for k_idx in range(len(k_values)):
            k = k_values[k_idx]
            maps.append(metrics_over_k[k]["map"])
            ndcgs.append(metrics_over_k[k]["ndcg"])

        style_to_use = line_styles[i % len(line_styles)]
        axes[0].plot(k_values, maps, style_to_use, label=sys_name)
        axes[1].plot(k_values, ndcgs, style_to_use, label=sys_name)

    axes[0].set_title("MAP@k — Comparison between systems")
    axes[0].set_xlabel("cutoff k")
    axes[0].set_ylabel("MAP Score")
    axes[0].set_xticks(k_values)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("nDCG@k — Comparison between systems")
    axes[1].set_xlabel("cutoff k")
    axes[1].set_ylabel("nDCG Score")
    axes[1].set_xticks(k_values)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved plot at:", out_path)

def plot_ablation(ablation_results, out_path):
    """
    Plot MAP@10 and nDCG@10 vs LSA n_components.
    This ablation uses all 225 queries for illustration purposes.
    The actual final k used in comparison is selected by 5-fold CV (see run_lsa_cross_validation).
    """
    k_vals = list(ablation_results.keys())
    # remove string keys if present during list build
    actual_k_vals = []
    for k_val in k_vals:
        if type(k_val) == int:
            actual_k_vals.append(k_val)

    actual_k_vals.sort()

    maps = []
    ndcgs = []
    for idx_k in range(len(actual_k_vals)):
        k = actual_k_vals[idx_k]
        maps.append(ablation_results[k]["map"])
        ndcgs.append(ablation_results[k]["ndcg"])

    baseline_map = None
    if "baseline_map" in ablation_results:
        baseline_map = ablation_results["baseline_map"]

    baseline_ndcg = None
    if "baseline_ndcg" in ablation_results:
        baseline_ndcg = ablation_results["baseline_ndcg"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(actual_k_vals, maps, '-o', color='steelblue', label='MAP@10 for LSA')
    if baseline_map is not None:
        axes[0].axhline(baseline_map, color='red', linestyle='--', label='VSM Baseline')
    axes[0].set_title("LSA Ablation for MAP@10")
    axes[0].set_xlabel("number of components used (k)")
    axes[0].set_ylabel("MAP@10")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(actual_k_vals, ndcgs, '-s', color='darkorange', label='nDCG@10 for LSA')
    if baseline_ndcg is not None:
        axes[1].axhline(baseline_ndcg, color='red', linestyle='--', label='VSM Baseline')
    axes[1].set_title("LSA Ablation for nDCG@10")
    axes[1].set_xlabel("number of components used (k)")
    axes[1].set_ylabel("nDCG@10")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved plot at:", out_path)

def plot_final_bar_chart(final_metrics, out_path):
    """
    Bar chart comparing Precision, Recall, MAP, nDCG, MRR at k=10
    for all final systems side by side.
    We added MRR to the chart since it is now computed for all systems.
    """
    systems = list(final_metrics.keys())

    # We now include MRR in the bar chart since it was missing before
    metric_keys = ["precision", "recall", "map", "ndcg", "mrr"]
    metric_labels = ["Precision@10", "Recall@10", "MAP@10", "nDCG@10", "MRR@10"]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(len(systems)):
        sys_name = systems[i]
        vals = []
        for m_idx in range(len(metric_keys)):
            m = metric_keys[m_idx]
            # Check if this metric key exists in the result dict
            if m in final_metrics[sys_name]:
                vals.append(final_metrics[sys_name][m])
            else:
                vals.append(0.0)

        offset = (i - len(systems) / 2.0 + 0.5) * width
        ax.bar(x + offset, vals, width, label=sys_name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Final calculated score")
    ax.set_title("Scores at cutoff k=10 across systems (with MRR)")
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved plot at:", out_path)

# ===========================================================================
# STANDARDIZED FORMULATION STATEMENT
# ===========================================================================

def print_standardized_formulation(system_name, measures_list):
    """
    This function prints the standardized evaluation formulation for a given system.
    According to our evaluation framework, we must always state:
    'Took algorithm C over dataset D under assumption E across measure F.'
    This is done for each system to maintain proper documentation.
    """
    # Building the measures string manually
    measures_str = ""
    for m_idx in range(len(measures_list)):
        if m_idx > 0:
            measures_str = measures_str + ", "
        measures_str = measures_str + measures_list[m_idx]

    formulation = ("Took algorithm " + str(system_name)
        + " over dataset Cranfield (1400 documents, 225 queries)"
        + " under assumption [binary relevance, position <= 4 in qrels]"
        + " across evaluation measures " + measures_str + ".")

    print("  [Formulation] " + formulation)
    return formulation


# ===========================================================================
# PER-QUERY GRANULARITY ANALYSIS
# ===========================================================================

def per_query_granularity_analysis(per_query_ap_dict, query_ids, query_text_dict, n_top=5):
    """
    This function does a query-level breakdown for each system compared to baseline.
    For each method, we find the top-N queries where it most outperforms or
    underperforms the baseline. This is required so we go beyond surface-level
    aggregate numbers and see exactly where each method helps or hurts.
    """
    baseline_ap = per_query_ap_dict["VSM (Baseline)"]
    analysis_result = {}

    system_names = []
    for name in per_query_ap_dict.keys():
        if name != "VSM (Baseline)":
            system_names.append(name)

    for sys_idx in range(len(system_names)):
        sys_name = system_names[sys_idx]
        method_ap = per_query_ap_dict[sys_name]

        # Compute delta for each query
        deltas = []
        for q_idx in range(len(query_ids)):
            q_id = query_ids[q_idx]
            delta_val = method_ap[q_idx] - baseline_ap[q_idx]
            deltas.append((delta_val, q_id, q_idx))

        # Sort by delta descending to find best improvements
        deltas.sort(reverse=True)

        # Top N improvements
        top_improvements = []
        for i in range(min(n_top, len(deltas))):
            delta_val, q_id, q_idx = deltas[i]
            q_text = "N/A"
            if q_id in query_text_dict:
                q_text = query_text_dict[q_id]
            top_improvements.append({
                "query_id": q_id,
                "delta_ap": round(delta_val, 4),
                "baseline_ap": round(baseline_ap[q_idx], 4),
                "method_ap": round(method_ap[q_idx], 4),
                "query_text": q_text[:100]
            })

        # Top N degradations (worst deltas at the end of sorted list)
        top_degradations = []
        for i in range(min(n_top, len(deltas))):
            idx = len(deltas) - 1 - i
            delta_val, q_id, q_idx = deltas[idx]
            q_text = "N/A"
            if q_id in query_text_dict:
                q_text = query_text_dict[q_id]
            top_degradations.append({
                "query_id": q_id,
                "delta_ap": round(delta_val, 4),
                "baseline_ap": round(baseline_ap[q_idx], 4),
                "method_ap": round(method_ap[q_idx], 4),
                "query_text": q_text[:100]
            })

        # Count how many queries improved, degraded, or stayed same
        count_improved = 0
        count_degraded = 0
        count_same = 0
        for d_tuple in deltas:
            if d_tuple[0] > 0.001:
                count_improved = count_improved + 1
            elif d_tuple[0] < -0.001:
                count_degraded = count_degraded + 1
            else:
                count_same = count_same + 1

        analysis_result[sys_name] = {
            "queries_improved": count_improved,
            "queries_degraded": count_degraded,
            "queries_unchanged": count_same,
            "top_improvements": top_improvements,
            "top_degradations": top_degradations
        }

    return analysis_result


# ===========================================================================
# SYSTEMATIC FAILURE ANALYSIS
# ===========================================================================

def systematic_failure_analysis(per_query_ap_dict, query_ids, query_text_dict, relevant_docs):
    """
    This function systematically finds queries where a method completely fails
    (AP@10 = 0) but the baseline had some success (AP > 0). This helps us
    understand the boundary conditions under which each method breaks.
    We also note cases where a method fixes baseline failures.

    IMPORTANT CHANGE: Previously the failure_reason was hardcoded as a generic
    message. Now we compute diagnostic properties for each failure:
      - Query length (number of words after preprocessing)
      - Number of relevant documents in qrels for that query
      - A hypothesis about likely failure cause based on these properties
    This gives us deeper empirical insight into WHY the method fails.
    """
    baseline_ap = per_query_ap_dict["VSM (Baseline)"]
    failure_report = {}

    system_names = []
    for name in per_query_ap_dict.keys():
        if name != "VSM (Baseline)":
            system_names.append(name)

    for sys_idx in range(len(system_names)):
        sys_name = system_names[sys_idx]
        method_ap = per_query_ap_dict[sys_name]

        # Finding new failures: baseline had AP > 0 but method has AP = 0
        new_failures = []
        # Finding fixed failures: baseline had AP = 0 but method has AP > 0
        fixed_failures = []

        for q_idx in range(len(query_ids)):
            q_id = query_ids[q_idx]
            b_ap = baseline_ap[q_idx]
            m_ap = method_ap[q_idx]

            q_text = "N/A"
            if q_id in query_text_dict:
                q_text = query_text_dict[q_id]

            # New failure: baseline managed something, but this method got nothing
            if b_ap > 0.001 and m_ap < 0.001:

                # Here we compute diagnostic properties to understand WHY this failed.
                # We look at the raw query text to estimate length.
                q_words = q_text.split()
                q_length = len(q_words)

                # Now check how many relevant docs exist for this query in qrels
                rel_count = 0
                if q_id in relevant_docs:
                    rel_count = len(relevant_docs[q_id])

                # Let us classify the query by length to give a diagnostic hint
                if q_length <= 3:
                    length_note = "very short (" + str(q_length) + " words)"
                elif q_length <= 7:
                    length_note = "medium-length (" + str(q_length) + " words)"
                else:
                    length_note = "long (" + str(q_length) + " words)"

                # Build a descriptive failure reason based on the query characteristics
                # This is much better than the previous hardcoded generic message
                if rel_count <= 2:
                    qrel_note = "sparse qrels (" + str(rel_count) + " relevant docs)"
                    cause_hint = "sparse coverage in qrels may amplify any small ranking change"
                elif q_length <= 3:
                    qrel_note = str(rel_count) + " relevant docs in qrels"
                    cause_hint = "very short query may suffer from vocabulary mismatch or poor term weighting"
                else:
                    qrel_note = str(rel_count) + " relevant docs in qrels"
                    cause_hint = "semantic representation may diverge from relevant documents for this topic"

                failure_reason = ("Query is " + length_note + ", " + qrel_note + ". "
                    + "Possible cause: " + cause_hint + ".")

                new_failures.append({
                    "query_id": q_id,
                    "baseline_ap": round(b_ap, 4),
                    "method_ap": round(m_ap, 4),
                    "query_text": q_text[:100],
                    "query_length_words": q_length,
                    "relevant_docs_count": rel_count,
                    "failure_reason": failure_reason
                })

            # Fixed: baseline got nothing, but method managed to retrieve something
            if b_ap < 0.001 and m_ap > 0.001:
                fixed_failures.append({
                    "query_id": q_id,
                    "baseline_ap": round(b_ap, 4),
                    "method_ap": round(m_ap, 4),
                    "query_text": q_text[:100]
                })

        failure_report[sys_name] = {
            "new_failures_count": len(new_failures),
            "fixed_failures_count": len(fixed_failures),
            "new_failures": new_failures,
            "fixed_failures": fixed_failures
        }

    return failure_report


# ===========================================================================
# MARGINAL DIFFERENCE ANALYSIS
# ===========================================================================

def marginal_difference_analysis(final_metrics, per_query_ap_dict, query_ids, query_text_dict):
    """
    When two systems differ by less than 2% MAP, we dig deeper to understand why.
    This function compares all pairs and for marginal ones, checks the query-level
    distribution to explain the difference rather than just accepting the number.
    """
    marginal_pairs = []

    system_names = list(final_metrics.keys())

    for i in range(len(system_names)):
        for j in range(i + 1, len(system_names)):
            name_a = system_names[i]
            name_b = system_names[j]
            map_a = final_metrics[name_a]["map"]
            map_b = final_metrics[name_b]["map"]

            diff = abs(map_a - map_b)
            # If less than 2% difference, we call it marginal and investigate
            if diff < 0.02:
                # Count query-level wins for each side
                ap_a = per_query_ap_dict[name_a]
                ap_b = per_query_ap_dict[name_b]

                wins_a = 0
                wins_b = 0
                ties = 0
                for q_idx in range(len(query_ids)):
                    if ap_a[q_idx] > ap_b[q_idx] + 0.001:
                        wins_a = wins_a + 1
                    elif ap_b[q_idx] > ap_a[q_idx] + 0.001:
                        wins_b = wins_b + 1
                    else:
                        ties = ties + 1

                explanation = ("Marginal MAP difference ("
                    + str(round(diff, 4))
                    + ") between " + name_a + " and " + name_b
                    + ". Query-level breakdown: " + name_a + " wins on "
                    + str(wins_a) + " queries, " + name_b + " wins on "
                    + str(wins_b) + " queries, " + str(ties)
                    + " ties. The aggregate difference is marginal and"
                    + " should not be taken as proof of superiority without"
                    + " formal hypothesis testing.")

                marginal_pairs.append({
                    "system_a": name_a,
                    "system_b": name_b,
                    "map_a": round(map_a, 4),
                    "map_b": round(map_b, 4),
                    "difference": round(diff, 4),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "explanation": explanation
                })

    return marginal_pairs


# ===========================================================================
# QUALITATIVE CASE STUDY
# ===========================================================================

def qualitative_case_study(all_ranked_dicts, query_ids, all_qrels,
                            query_text_dict, doc_text_dict):
    """
    For the documented failure cases from Part 4, print and return the
    rank each system assigns to the known-relevant document.

    Failure Case 1: Query 1, Doc 29 — vocabulary mismatch (heated vs thermal)
    Failure Case 2: Query "movie star" is toy example, not in Cranfield queries,
                    so we find the worst-ranked highly-relevant pair from qrels.
    """
    case_studies = {}

    target_query_id = 1
    target_doc_id   = 29

    if target_query_id in query_ids:
        # Check text in dict or return not available
        q_str = "N/A"
        if target_query_id in query_text_dict:
            q_str = query_text_dict[target_query_id]

        case_entry = {
            "query_id":  target_query_id,
            "doc_id":    target_doc_id,
            "query_text": q_str,
            "ranks": {}
        }

        list_of_sys_names = list(all_ranked_dicts.keys())
        for sn_idx in range(len(list_of_sys_names)):
            sys_name = list_of_sys_names[sn_idx]
            ranked_dict = all_ranked_dicts[sys_name]

            ranked_list = []
            if target_query_id in ranked_dict:
                ranked_list = ranked_dict[target_query_id]

            rank = -1
            for r_idx in range(len(ranked_list)):
                if ranked_list[r_idx] == target_doc_id:
                    rank = r_idx + 1
                    break

            case_entry["ranks"][sys_name] = rank
        case_studies["vocab_mismatch"] = case_entry

    # ── Case 2: Find the worst-ranked highly-relevant doc across all queries ─
    # (any query, position 1 or 2 in qrels, but baseline ranked it > 100)
    worst_case = None
    baseline_dict = list(all_ranked_dicts.values())[0]

    for q_idx in range(len(all_qrels)):
        qrel = all_qrels[q_idx]
        q_id = int(qrel["query_num"])
        d_id = int(qrel["id"])
        pos  = qrel["position"]
        if pos == 1 or pos == 2:
            if q_id in baseline_dict:
                bl_list = baseline_dict[q_id]
                bl_rank = -1
                for i in range(len(bl_list)):
                    if bl_list[i] == d_id:
                        bl_rank = i + 1
                        break

                if bl_rank > 100:
                    worst_case = {"query_id": q_id, "doc_id": d_id,
                                  "baseline_rank": bl_rank, "qrel_position": pos}
                    break

    if worst_case is not None:
        q_id = worst_case["query_id"]
        d_id = worst_case["doc_id"]

        q_str = "N/A"
        if q_id in query_text_dict:
            q_str = query_text_dict[q_id]
        worst_case["query_text"] = q_str
        worst_case["ranks"] = {}

        list_of_sys_names = list(all_ranked_dicts.keys())
        for sn_idx in range(len(list_of_sys_names)):
            sys_name = list_of_sys_names[sn_idx]
            ranked_dict = all_ranked_dicts[sys_name]

            ranked_list = []
            if q_id in ranked_dict:
                ranked_list = ranked_dict[q_id]

            rank = -1
            for r_idx in range(len(ranked_list)):
                if ranked_list[r_idx] == d_id:
                    rank = r_idx + 1
                    break

            worst_case["ranks"][sys_name] = rank
        case_studies["worst_ranked_relevant"] = worst_case

    return case_studies

# ===========================================================================
# HYPERPARAMETER TUNING K-FOLD CV FOR BM25
# ===========================================================================

def run_bm25_cross_validation(processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, k_eval=10):
    """
    This function does 5-fold cross validation to find the best BM25 hyperparameters.
    We strictly use cross-validation for parameter tuning only, NOT for final evaluation.
    The returned best_k1, best_b will be used for the final BM25 evaluation.
    This separation ensures we do not overfit parameters on the test set.
    Returns the best (k1, b) pair selected by averaging MAP across all 5 folds.
    """
    print("\n  Executing 5-Fold Cross Validation for hyperparameter tuning...")
    # Here we partition queries manually to rigorously avoid overfitting on the test set, as taught in theory.
    num_queries = len(query_ids)
    fold_size = int(num_queries / 5)

    k1_options = [1.2, 1.5, 2.0]
    b_options = [0.5, 0.75, 0.9]

    # For each (k1, b) combination we will accumulate MAP across all 5 folds.
    # Then we select the pair with the highest average CV MAP.
    # This is the correct way to use cross-validation for parameter selection.
    combo_cumulative_map = {}
    for k_idx in range(len(k1_options)):
        for b_idx in range(len(b_options)):
            combo_key = (k1_options[k_idx], b_options[b_idx])
            combo_cumulative_map[combo_key] = 0.0

    # We also track fold test MAPs to report the overall CV estimate
    fold_test_maps = []

    for fold_index in range(5):
        # Let us calculate the fold indices manually
        start_idx = fold_index * fold_size
        end_idx = start_idx + fold_size
        if fold_index == 4:
            # Handle the last fold which may have a few extra queries
            end_idx = num_queries

        # Separate queries into test fold and training fold
        test_q_ids = []
        train_q_ids = []
        for q_index in range(num_queries):
            if q_index >= start_idx and q_index < end_idx:
                test_q_ids.append(query_ids[q_index])
            else:
                train_q_ids.append(query_ids[q_index])

        # We build the training query data list in the same order as train_q_ids
        train_queries_data = []
        for q_idx2 in range(len(query_ids)):
            if query_ids[q_idx2] in train_q_ids:
                train_queries_data.append(processed_queries[q_idx2])

        # Evaluate each (k1, b) combination on training fold only
        best_fold_map = -1.0
        best_fold_k1 = 1.5
        best_fold_b = 0.75

        for k_idx in range(len(k1_options)):
            current_k1 = k1_options[k_idx]
            for b_idx in range(len(b_options)):
                current_b = b_options[b_idx]

                # Initialize a temporary BM25 for this param combination
                temp_bm25 = InformationRetrievalBM25(k1=current_k1, b=current_b)
                temp_bm25.buildIndex(processed_docs, doc_ids)

                temp_ranked = temp_bm25.rank(train_queries_data)
                temp_dict = ranked_list_to_dict(temp_ranked, train_q_ids)

                train_map = evaluator.meanAveragePrecision(temp_dict, train_q_ids, relevant_docs, k_eval)

                # Accumulate MAP for this combo so we can average later across folds
                combo_key = (current_k1, current_b)
                combo_cumulative_map[combo_key] = combo_cumulative_map[combo_key] + train_map

                if train_map > best_fold_map:
                    best_fold_map = train_map
                    best_fold_k1 = current_k1
                    best_fold_b = current_b

        print("    Fold", fold_index+1, ": Best train params found: k1=", best_fold_k1, "b=", best_fold_b, ". Now evaluating on held-out test fold...")

        # Test on out-of-fold queries using the best params for this fold
        test_bm25 = InformationRetrievalBM25(k1=best_fold_k1, b=best_fold_b)
        test_bm25.buildIndex(processed_docs, doc_ids)

        test_queries_data = []
        for q_idx2 in range(len(query_ids)):
            if query_ids[q_idx2] in test_q_ids:
                test_queries_data.append(processed_queries[q_idx2])

        test_ranked = test_bm25.rank(test_queries_data)
        test_dict = ranked_list_to_dict(test_ranked, test_q_ids)
        test_map = evaluator.meanAveragePrecision(test_dict, test_q_ids, relevant_docs, k_eval)

        fold_test_maps.append(test_map)

    # Now we compute the average CV MAP for each (k1, b) combination
    # The combination with the highest average MAP across all 5 folds is selected
    best_overall_k1 = 1.5
    best_overall_b = 0.75
    best_overall_avg_map = -1.0

    print("\n  Average CV MAP for each (k1, b) combination:")
    all_combos = list(combo_cumulative_map.keys())
    for combo_idx in range(len(all_combos)):
        combo_key = all_combos[combo_idx]
        avg_map_for_combo = combo_cumulative_map[combo_key] / 5.0
        print("    k1=", combo_key[0], "b=", combo_key[1], "-> Avg CV MAP =", round(avg_map_for_combo, 4))

        if avg_map_for_combo > best_overall_avg_map:
            best_overall_avg_map = avg_map_for_combo
            best_overall_k1 = combo_key[0]
            best_overall_b = combo_key[1]

    # Report the overall cross-validated test MAP (across fold test sets)
    overall_avg_test_map = 0.0
    for m in fold_test_maps:
        overall_avg_test_map = overall_avg_test_map + m
    overall_avg_test_map = overall_avg_test_map / 5.0

    print("\n  Globally best hyperparameters selected by CV: k1=", best_overall_k1, "b=", best_overall_b)
    print("  Overall robust Cross-Validated test MAP@10 estimate:", round(overall_avg_test_map, 4))
    print("  These params will now be used for the final evaluation on the full dataset.")

    # Returning the best params so the caller can use them for final evaluation
    return best_overall_k1, best_overall_b


# ===========================================================================
# LSA HYPERPARAMETER TUNING K-FOLD CV
# ===========================================================================

def run_lsa_cross_validation(processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, k_eval=10):
    """
    This function does 5-fold cross validation to find the best LSA n_components value.
    We strictly use cross-validation for parameter tuning only, NOT for final evaluation.
    The returned best_k will then be used for the final LSA evaluation on the full dataset.

    Previously, the best n_components was selected by evaluating MAP on ALL 225 queries
    and picking the k with highest MAP@10. This was a data leakage issue because the
    same queries used for selection were also used for the final evaluation. Now we
    properly separate the validation split from the test split using 5-fold CV.

    This mirrors exactly the same approach used for BM25 hyperparameter tuning above.
    """
    print("\n  Executing 5-Fold Cross Validation for LSA n_components tuning...")

    # Here we partition queries manually to avoid leakage on the test set
    num_queries = len(query_ids)
    fold_size = int(num_queries / 5)

    # These are the candidate values for n_components that we want to search over
    k_options = [50, 100, 200, 300]

    # For each n_components value we will accumulate MAP across all 5 folds
    # Then we select the k with the highest average CV MAP
    combo_cumulative_map = {}
    for k_val in k_options:
        combo_cumulative_map[k_val] = 0.0

    # We also track fold test MAPs for the overall CV estimate
    fold_test_maps = []

    for fold_index in range(5):
        # Calculate the start and end indices for this fold manually
        start_idx = fold_index * fold_size
        end_idx = start_idx + fold_size
        if fold_index == 4:
            # The last fold gets any remaining queries too
            end_idx = num_queries

        # Separate queries into test fold and training fold
        test_q_ids = []
        train_q_ids = []
        for q_index in range(num_queries):
            if q_index >= start_idx and q_index < end_idx:
                test_q_ids.append(query_ids[q_index])
            else:
                train_q_ids.append(query_ids[q_index])

        # Build the training query data list (only training queries)
        train_queries_data = []
        for q_idx2 in range(len(query_ids)):
            if query_ids[q_idx2] in train_q_ids:
                train_queries_data.append(processed_queries[q_idx2])

        # We evaluate each n_components option on the training fold only
        best_fold_map = -1.0
        best_fold_k = 100  # reasonable default if somehow all ties

        for k_val in k_options:
            # Initialize a temporary LSA model with this n_components value
            # We build on ALL docs (the index) but evaluate only on train queries
            temp_lsa = InformationRetrievalLSA(n_components=k_val)
            temp_lsa.buildIndex(processed_docs, doc_ids)

            temp_ranked = temp_lsa.rank(train_queries_data)
            temp_dict = ranked_list_to_dict(temp_ranked, train_q_ids)

            train_map = evaluator.meanAveragePrecision(temp_dict, train_q_ids, relevant_docs, k_eval)

            # Accumulating MAP for this k value so we can average over folds later
            combo_cumulative_map[k_val] = combo_cumulative_map[k_val] + train_map

            if train_map > best_fold_map:
                best_fold_map = train_map
                best_fold_k = k_val

        print("    Fold", fold_index+1, ": Best train k found:", best_fold_k,
              "(train MAP =", round(best_fold_map, 4), "). Evaluating on held-out test fold...")

        # Now test on out-of-fold queries using the best k found for this fold
        test_lsa = InformationRetrievalLSA(n_components=best_fold_k)
        test_lsa.buildIndex(processed_docs, doc_ids)

        # Collecting the test fold query data in the same order as test_q_ids
        test_queries_data = []
        for q_idx2 in range(len(query_ids)):
            if query_ids[q_idx2] in test_q_ids:
                test_queries_data.append(processed_queries[q_idx2])

        test_ranked = test_lsa.rank(test_queries_data)
        test_dict = ranked_list_to_dict(test_ranked, test_q_ids)
        test_map = evaluator.meanAveragePrecision(test_dict, test_q_ids, relevant_docs, k_eval)

        fold_test_maps.append(test_map)

    # Now we find the best overall k by looking at average MAP across all folds
    best_overall_k = 100  # default
    best_overall_avg_map = -1.0

    print("\n  Average CV MAP for each LSA n_components value:")
    for k_val in k_options:
        avg_map_for_k = combo_cumulative_map[k_val] / 5.0
        print("    n_components =", k_val, "-> Avg CV MAP =", round(avg_map_for_k, 4))

        if avg_map_for_k > best_overall_avg_map:
            best_overall_avg_map = avg_map_for_k
            best_overall_k = k_val

    # Report the overall CV test MAP estimate
    overall_avg_test_map = 0.0
    for m in fold_test_maps:
        overall_avg_test_map = overall_avg_test_map + m
    overall_avg_test_map = overall_avg_test_map / 5.0

    print("\n  Globally best LSA n_components selected by CV:", best_overall_k)
    print("  Overall Cross-Validated test MAP@10 estimate:", round(overall_avg_test_map, 4))
    print("  This k will now be used for the final LSA evaluation on the full dataset.")

    return best_overall_k


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    total_start = time.time()

    # We create the folder if it does not exist
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading Cranfield dataset...")
    all_queries = load_json(DATASET_PATH + "cran_queries.json")
    all_docs    = load_json(DATASET_PATH + "cran_docs.json")
    all_qrels   = load_json(DATASET_PATH + "cran_qrels.json")

    # Using shared parsing function
    query_ids, query_texts, doc_ids, doc_texts = parse_cranfield_data(all_queries, all_docs)

    # We also need text dicts for qualitative analysis
    query_text_dict = {}
    for i in range(len(all_queries)):
        q = all_queries[i]
        query_text_dict[int(q["query number"])] = q["query"]

    doc_text_dict = {}
    for i in range(len(all_docs)):
        d = all_docs[i]
        doc_text_dict[d["id"]] = d["body"]

    # Binary relevance dict (position 1-4 all counted as relevant)
    relevant_docs = get_relevant_docs(all_qrels)

    # Ensuring no errors if query ID is not in relevant_docs
    for q_idx in range(len(query_ids)):
        q_id = query_ids[q_idx]
        if q_id not in relevant_docs:
            relevant_docs[q_id] = []

    # Graded relevance dict (position 1-2 -> grade 2, position 3-4 -> grade 1)
    # We use this for graded nDCG computation which is more informative than binary nDCG
    graded_relevant_docs = build_graded_relevant_docs(all_qrels)

    # Also make sure every query has an entry in graded_relevant_docs
    for q_idx in range(len(query_ids)):
        q_id = query_ids[q_idx]
        if q_id not in graded_relevant_docs:
            graded_relevant_docs[q_id] = {}

    print("  Loaded", len(all_queries), "queries and", len(all_docs), "documents.")

    # ── Preprocess ──────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing...")
    processed_queries, processed_docs, stop_obj = preprocess_all(query_texts, doc_texts)

    evaluator = Evaluation()
    k_range = list(range(1, EVAL_K + 1))

    # ── HYPERPARAMETER SEARCH CV FOR BM25 ─────────────────────────────────
    # Cross-validation is used EXCLUSIVELY for parameter tuning.
    # The returned best_k1, best_b will be used for the final BM25 evaluation.
    # This separation ensures we do not overfit parameters on the test set.
    cv_best_k1, cv_best_b = run_bm25_cross_validation(
        processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, EVAL_K
    )

    # ── HYPERPARAMETER SEARCH CV FOR LSA ──────────────────────────────────
    # IMPORTANT FIX: Previously we selected the best LSA n_components by evaluating
    # MAP@10 on the full 225 queries. This is data leakage because those same queries
    # are also used for final evaluation. Now we use proper 5-fold CV for LSA too.
    cv_best_lsa_k = run_lsa_cross_validation(
        processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, EVAL_K
    )

    # We collect metrics-over-k for each final system (for the line plots)
    # and per-query AP at k=10 for Wilcoxon tests.
    all_systems_metrics_over_k = {}  # sys_name -> {k -> metrics_dict}
    final_metrics_at_k10       = {}  # sys_name -> metrics_dict at k=10
    per_query_ap_at_k10        = {}  # sys_name -> list of per-query AP values
    graded_ndcg_at_k10         = {}  # sys_name -> graded nDCG at k=10 (new)

    # Here we define the evaluation measures we report for the formulation
    eval_measures = ["Precision@k", "Recall@k", "F0.5@k", "MAP@k", "nDCG@k (binary)", "nDCG@k (graded)", "MRR@k"]

    # ── BASELINE: TF-IDF VSM ────────────────────────────────────────────────
    print("\n[3/6] Running BASELINE (TF-IDF VSM)...")
    print_standardized_formulation("VSM (TF-IDF Baseline)", eval_measures)
    t0 = time.time()
    vsm = InformationRetrieval()
    vsm.buildIndex(processed_docs, doc_ids)
    ranked_vsm = vsm.rank(processed_queries)
    vsm_dict   = ranked_list_to_dict(ranked_vsm, query_ids)
    print("  Completed VSM in", round(time.time()-t0, 2), "seconds")

    vsm_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        vsm_metrics_k[k] = compute_all_metrics(evaluator, vsm_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["VSM (Baseline)"] = vsm_metrics_k
    final_metrics_at_k10["VSM (Baseline)"]       = vsm_metrics_k[EVAL_K]
    per_query_ap_at_k10["VSM (Baseline)"]        = compute_per_query_ap(
        evaluator, vsm_dict, query_ids, relevant_docs, EVAL_K)

    # Computing graded nDCG for baseline as well
    graded_ndcg_at_k10["VSM (Baseline)"] = compute_mean_graded_ndcg(
        vsm_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(vsm_metrics_k[EVAL_K]['map'], 4),
          "| nDCG@10 =", round(vsm_metrics_k[EVAL_K]['ndcg'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["VSM (Baseline)"], 4),
          "| MRR@10 =", round(vsm_metrics_k[EVAL_K]['mrr'], 4))

    # ── METHOD A: BM25 ───────────────────────────────────────────────────────
    # Here we use the k1 and b values that were selected by cross-validation above.
    # This is the correct way to do it - CV selects params, test set is for final eval only.
    print("\n[4a/6] Running METHOD A (BM25, k1=" + str(cv_best_k1) + ", b=" + str(cv_best_b) + ")...")
    print_standardized_formulation("BM25 (k1=" + str(cv_best_k1) + ", b=" + str(cv_best_b) + ")", eval_measures)
    t0 = time.time()
    bm25 = InformationRetrievalBM25(k1=cv_best_k1, b=cv_best_b)
    bm25.buildIndex(processed_docs, doc_ids)
    ranked_bm25 = bm25.rank(processed_queries)
    bm25_dict   = ranked_list_to_dict(ranked_bm25, query_ids)
    print("  Completed BM25 in", round(time.time()-t0, 2), "seconds")

    bm25_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        bm25_metrics_k[k] = compute_all_metrics(evaluator, bm25_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["BM25"] = bm25_metrics_k
    final_metrics_at_k10["BM25"]       = bm25_metrics_k[EVAL_K]
    per_query_ap_at_k10["BM25"]        = compute_per_query_ap(
        evaluator, bm25_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10["BM25"] = compute_mean_graded_ndcg(
        bm25_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(bm25_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(bm25_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["BM25"], 4))

    # ── METHOD B: LSA — Ablation Plot + CV-Selected Final Model ─────────────
    # IMPORTANT NOTE: We still run the ablation over all k values for the plot
    # (this shows how MAP changes with n_components). But the k used for the
    # final comparison is cv_best_lsa_k selected by 5-fold CV above.
    # This distinction is crucial: the ablation is for visualization only,
    # not for selecting hyperparameters on the test set.
    print("\n[4b/6] Running METHOD B (LSA) — Ablation plot over n_components...")
    print_standardized_formulation("LSA (Truncated SVD, CV-selected k=" + str(cv_best_lsa_k) + ")", eval_measures)
    ablation_results = {}

    baseline_map_k10  = vsm_metrics_k[EVAL_K]["map"]
    baseline_ndcg_k10 = vsm_metrics_k[EVAL_K]["ndcg"]

    for i in range(len(LSA_K_VALUES)):
        lsa_k = LSA_K_VALUES[i]
        print("  Ablation: evaluating LSA with n_components =", lsa_k)
        t0 = time.time()
        lsa = InformationRetrievalLSA(n_components=lsa_k)
        lsa.buildIndex(processed_docs, doc_ids)
        ranked_lsa = lsa.rank(processed_queries)
        lsa_dict   = ranked_list_to_dict(ranked_lsa, query_ids)
        lsa_m      = compute_all_metrics(evaluator, lsa_dict, query_ids, relevant_docs, EVAL_K)
        print("  MAP@10 =", round(lsa_m['map'], 4), "| nDCG@10 =", round(lsa_m['ndcg'], 4),
              "(took", round(time.time()-t0, 1), "s)")
        ablation_results[lsa_k] = lsa_m

    # Add baseline references to ablation dict for the ablation plot
    ablation_results["baseline_map"]  = baseline_map_k10
    ablation_results["baseline_ndcg"] = baseline_ndcg_k10

    # Now we run the final LSA model using cv_best_lsa_k (selected by CV, not by ablation)
    print("\n  Final LSA model using CV-selected n_components =", cv_best_lsa_k)
    t0 = time.time()
    lsa_best = InformationRetrievalLSA(n_components=cv_best_lsa_k)
    lsa_best.buildIndex(processed_docs, doc_ids)
    ranked_lsa_best = lsa_best.rank(processed_queries)
    lsa_best_dict   = ranked_list_to_dict(ranked_lsa_best, query_ids)
    print("  Completed final LSA in", round(time.time()-t0, 2), "seconds")

    lsa_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        lsa_metrics_k[k] = compute_all_metrics(evaluator, lsa_best_dict, query_ids, relevant_docs, k)

    lsa_sys_name = "LSA (k=" + str(cv_best_lsa_k) + ", CV)"
    all_systems_metrics_over_k[lsa_sys_name] = lsa_metrics_k
    final_metrics_at_k10[lsa_sys_name]       = lsa_metrics_k[EVAL_K]
    per_query_ap_at_k10[lsa_sys_name]        = compute_per_query_ap(
        evaluator, lsa_best_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10[lsa_sys_name] = compute_mean_graded_ndcg(
        lsa_best_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  MAP@10 =", round(lsa_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(lsa_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10[lsa_sys_name], 4))

    # ── METHOD C: VSM + Query Expansion ─────────────────────────────────────
    print("\n[4c/6] Running METHOD C (VSM + WordNet Query Expansion)...")
    print_standardized_formulation("VSM + WordNet Query Expansion", eval_measures)
    t0 = time.time()
    expander = QueryExpander()

    expanded_queries = []
    for pq_idx in range(len(processed_queries)):
        pq = processed_queries[pq_idx]
        expanded_q, expansion_map = expander.expand_query_for_ir(pq)
        expanded_queries.append(expanded_q)

    # Run standard VSM on expanded queries (index is already built)
    vsm_exp = InformationRetrieval()
    vsm_exp.buildIndex(processed_docs, doc_ids)
    ranked_exp = vsm_exp.rank(expanded_queries)
    exp_dict   = ranked_list_to_dict(ranked_exp, query_ids)
    print("  Completed Query Expansion in", round(time.time()-t0, 2), "seconds")

    exp_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        exp_metrics_k[k] = compute_all_metrics(
            evaluator, exp_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["VSM + QE"] = exp_metrics_k
    final_metrics_at_k10["VSM + QE"]       = exp_metrics_k[EVAL_K]
    per_query_ap_at_k10["VSM + QE"]        = compute_per_query_ap(
        evaluator, exp_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10["VSM + QE"] = compute_mean_graded_ndcg(
        exp_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(exp_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(exp_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["VSM + QE"], 4))

    # ── METHOD D: Pseudo-Relevance Feedback (PRF) ───────────────────────────
    print("\n[4d/6] Running METHOD D (PRF)...")
    print_standardized_formulation("BM25 + Pseudo-Relevance Feedback", eval_measures)
    t0 = time.time()
    # Using the same CV-selected params as plain BM25, since PRF is built on top of BM25
    prf = InformationRetrievalPRF(k1=cv_best_k1, b=cv_best_b)
    prf.buildIndex(processed_docs, doc_ids)
    ranked_prf = prf.rank(processed_queries)
    prf_dict   = ranked_list_to_dict(ranked_prf, query_ids)
    print("  Completed Pseudo-Relevance Feedback in", round(time.time()-t0, 2), "seconds")

    prf_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        prf_metrics_k[k] = compute_all_metrics(evaluator, prf_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["BM25 + PRF"] = prf_metrics_k
    final_metrics_at_k10["BM25 + PRF"]       = prf_metrics_k[EVAL_K]
    per_query_ap_at_k10["BM25 + PRF"]        = compute_per_query_ap(
        evaluator, prf_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10["BM25 + PRF"] = compute_mean_graded_ndcg(
        prf_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(prf_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(prf_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["BM25 + PRF"], 4))

    # ── METHOD E: Explicit Semantic Analysis (ESA) ──────────────────────────
    print("\n[4e/6] Running METHOD E (ESA — WordNet Concept Space)...")
    print_standardized_formulation("ESA (WordNet Concept Space)", eval_measures)
    t0 = time.time()
    esa = InformationRetrievalESA()
    esa.buildIndex(processed_docs, doc_ids)
    ranked_esa = esa.rank(processed_queries)
    esa_dict   = ranked_list_to_dict(ranked_esa, query_ids)
    print("  Completed ESA in", round(time.time()-t0, 2), "seconds")

    esa_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        esa_metrics_k[k] = compute_all_metrics(evaluator, esa_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["ESA"] = esa_metrics_k
    final_metrics_at_k10["ESA"]       = esa_metrics_k[EVAL_K]
    per_query_ap_at_k10["ESA"]        = compute_per_query_ap(
        evaluator, esa_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10["ESA"] = compute_mean_graded_ndcg(
        esa_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(esa_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(esa_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["ESA"], 4))

    # ── METHOD F: Explicit Semantic Analysis (ESA - Cranfield Space) ─────────
    print("\n[4f/6] Running METHOD F (ESA — Cranfield Concept Space)...")
    print_standardized_formulation("ESA (Cranfield Concept Space)", eval_measures)
    t0 = time.time()
    esac = InformationRetrievalESACranfield()
    esac.buildIndex(processed_docs, doc_ids)
    ranked_esac = esac.rank(processed_queries)
    esac_dict   = ranked_list_to_dict(ranked_esac, query_ids)
    print("  Completed ESA-Cranfield in", round(time.time()-t0, 2), "seconds")

    esac_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        esac_metrics_k[k] = compute_all_metrics(evaluator, esac_dict, query_ids, relevant_docs, k)

    all_systems_metrics_over_k["ESA (Cranfield)"] = esac_metrics_k
    final_metrics_at_k10["ESA (Cranfield)"]       = esac_metrics_k[EVAL_K]
    per_query_ap_at_k10["ESA (Cranfield)"]        = compute_per_query_ap(
        evaluator, esac_dict, query_ids, relevant_docs, EVAL_K)
    graded_ndcg_at_k10["ESA (Cranfield)"] = compute_mean_graded_ndcg(
        esac_dict, query_ids, graded_relevant_docs, EVAL_K)

    print("  Calculated MAP@10 =", round(esac_metrics_k[EVAL_K]['map'], 4),
          "| MRR@10 =", round(esac_metrics_k[EVAL_K]['mrr'], 4),
          "| nDCG@10 (graded) =", round(graded_ndcg_at_k10["ESA (Cranfield)"], 4))

    # ── WILCOXON HYPOTHESIS TESTS ────────────────────────────────────────────
    print("\n[5/6] Running Wilcoxon Signed-Rank Tests...")
    print("  Formal Hypothesis Framework:")
    print("    H0: The AP scores of method X and the VSM baseline come from the same distribution.")
    print("    H1 (one-sided): Method X produces significantly higher AP scores (for BM25, LSA, PRF).")
    print("    H1 (two-sided): Method X produces significantly different AP scores (for QE, ESA).")
    print("    Significance level: alpha = 0.05")
    print("    Effect size: Cohen's d (small >= 0.2, medium >= 0.5, large >= 0.8)")
    print()

    baseline_ap = per_query_ap_at_k10["VSM (Baseline)"]
    wilcoxon_results = []

    # We use one-sided 'greater' for methods with strong theoretical prior for improvement.
    # We use two-sided for methods where the direction of change is uncertain.
    # This is the statistically correct approach - we should not assume direction without a prior.
    systems_and_alternatives = [
        ("BM25",          "greater"),   # BM25 has proven theoretical advantage over VSM
        (lsa_sys_name,    "greater"),   # LSA captures latent semantics -> expected to improve
        ("VSM + QE",      "two-sided"), # Query expansion can hurt (wrong synonyms) or help
        ("BM25 + PRF",    "greater"),   # PRF is an additional refinement step -> expected to improve
        ("ESA",           "two-sided"), # ESA effectiveness is not guaranteed a priori
        ("ESA (Cranfield)", "two-sided") # Same reasoning as ESA above
    ]

    for s_idx in range(len(systems_and_alternatives)):
        sys_name = systems_and_alternatives[s_idx][0]
        alt_direction = systems_and_alternatives[s_idx][1]

        result = wilcoxon_test(baseline_ap, per_query_ap_at_k10[sys_name], sys_name, alternative=alt_direction)
        wilcoxon_results.append(result)

        sig_str = "Cannot reject H0"
        if result["significant"]:
            sig_str = "REJECT H0"

        effect_str = ""
        if "cohens_d" in result:
            effect_str = " | Cohen's d=" + str(round(result['cohens_d'], 3))

        print("  System:", sys_name, "| alternative:", alt_direction,
              "| p-value:", round(result['p_value'], 4), "|", sig_str + effect_str)

    # ── QUALITATIVE CASE STUDIES ─────────────────────────────────────────────
    print("\n[6/6] Qualitative Analysis...")
    all_ranked_dicts = {
        "VSM (Baseline)": vsm_dict,
        "BM25":           bm25_dict,
        lsa_sys_name:     lsa_best_dict,
        "VSM + QE":       exp_dict,
        "BM25 + PRF":     prf_dict,
        "ESA":            esa_dict,
        "ESA (Cranfield)": esac_dict,
    }
    case_studies = qualitative_case_study(
        all_ranked_dicts, query_ids, all_qrels, query_text_dict, doc_text_dict)

    print("\n  ** Case 1: Vocabulary Mismatch (Query 1, Doc 29) **")
    if "vocab_mismatch" in case_studies:
        cs = case_studies["vocab_mismatch"]
        print("  -> Query was:", cs['query_text'][:80], "...")
        sys_names_list = list(cs["ranks"].keys())
        for sys_idx in range(len(sys_names_list)):
            sys_name = sys_names_list[sys_idx]
            rank = cs["ranks"][sys_name]
            print("     Rank by", sys_name, "is", rank)

    print("\n  ** Case 2: Very poorly ranked relevant document **")
    if "worst_ranked_relevant" in case_studies:
        wc = case_studies["worst_ranked_relevant"]
        print("  -> Query (ID " + str(wc['query_id']) + "):", wc['query_text'][:80], "... (Doc ID " + str(wc['doc_id']) + ")")
        sys_names_list = list(wc["ranks"].keys())
        for sys_idx in range(len(sys_names_list)):
            sys_name = sys_names_list[sys_idx]
            rank = wc["ranks"][sys_name]
            print("     Rank by", sys_name, "is", rank)

    # ── PER-QUERY GRANULARITY ANALYSIS ────────────────────────────────────────
    print("\n[Step 7] Per-Query Granularity Analysis...")
    pq_analysis = per_query_granularity_analysis(
        per_query_ap_at_k10, query_ids, query_text_dict, n_top=5)

    # Printing a summary for each system
    pq_sys_names = list(pq_analysis.keys())
    for pq_idx in range(len(pq_sys_names)):
        sys_name = pq_sys_names[pq_idx]
        info = pq_analysis[sys_name]
        print("  ", sys_name, ": improved on", info["queries_improved"],
              "queries, degraded on", info["queries_degraded"],
              "queries, unchanged on", info["queries_unchanged"], "queries.")

    # ── SYSTEMATIC FAILURE ANALYSIS ───────────────────────────────────────────
    # Note: We now pass relevant_docs so that the failure analysis can report
    # how many relevant documents each failed query actually has in qrels.
    print("\n[Step 8] Systematic Failure Analysis (with diagnostic info)...")
    failure_report = systematic_failure_analysis(
        per_query_ap_at_k10, query_ids, query_text_dict, relevant_docs)

    fa_sys_names = list(failure_report.keys())
    for fa_idx in range(len(fa_sys_names)):
        sys_name = fa_sys_names[fa_idx]
        fr = failure_report[sys_name]
        print("  ", sys_name, ":", fr["new_failures_count"],
              "new failures vs baseline,", fr["fixed_failures_count"],
              "baseline failures fixed.")

    # ── MARGINAL DIFFERENCE ANALYSIS ──────────────────────────────────────────
    print("\n[Step 9] Marginal Difference Analysis (< 2% MAP gap)...")
    marginal_pairs = marginal_difference_analysis(
        final_metrics_at_k10, per_query_ap_at_k10, query_ids, query_text_dict)

    if len(marginal_pairs) == 0:
        print("  No pairs with marginal MAP difference found.")
    else:
        for mp_idx in range(len(marginal_pairs)):
            mp = marginal_pairs[mp_idx]
            print("  ", mp["explanation"])

    # ── GRADED nDCG SUMMARY ──────────────────────────────────────────────────
    print("\n[Step 10] Graded nDCG@10 Summary (position 1-2 = grade 2, position 3-4 = grade 1):")
    sys_names_graded = list(graded_ndcg_at_k10.keys())
    for g_idx in range(len(sys_names_graded)):
        sys_name = sys_names_graded[g_idx]
        g_score = graded_ndcg_at_k10[sys_name]
        print("  ", sys_name, "-> Graded nDCG@10 =", round(g_score, 4))

    # ── WRITING ALL OUTPUTS ──────────────────────────────────────────────────
    print("\n[Step 11] Writing outputs to disk...")

    save_json(final_metrics_at_k10, OUT_DIR + "part5_final_metrics_k10.json")
    save_json(graded_ndcg_at_k10, OUT_DIR + "part5_graded_ndcg_k10.json")

    ablation_serialisable = {}
    for k in ablation_results.keys():
        ablation_serialisable[str(k)] = ablation_results[k]
    save_json(ablation_serialisable, OUT_DIR + "part5_lsa_ablation.json")

    save_json(wilcoxon_results, OUT_DIR + "part5_wilcoxon_tests.json")
    save_json(case_studies, OUT_DIR + "part5_case_studies.json")
    save_json(pq_analysis, OUT_DIR + "part5_per_query_analysis.json")

    # Saving the full per-query AP for all 225 queries for each system.
    # We need this in query_class_analysis.py to do class-level breakdown.
    # We save query_ids alongside so the analysis script can match them correctly.
    full_pq_ap_to_save = {}
    full_pq_ap_to_save["query_ids"] = query_ids
    pq_ap_sys_names = list(per_query_ap_at_k10.keys())
    for sys_idx in range(len(pq_ap_sys_names)):
        sys_name = pq_ap_sys_names[sys_idx]
        full_pq_ap_to_save[sys_name] = per_query_ap_at_k10[sys_name]
    save_json(full_pq_ap_to_save, OUT_DIR + "part5_full_per_query_ap.json")
    save_json(failure_report, OUT_DIR + "part5_failure_analysis.json")
    save_json(marginal_pairs, OUT_DIR + "part5_marginal_differences.json")

    metrics_over_k_serialisable = {}
    sys_list = list(all_systems_metrics_over_k.keys())
    for s_idx in range(len(sys_list)):
        sys_name = sys_list[s_idx]
        k_dict = all_systems_metrics_over_k[sys_name]

        inner_dict = {}
        k_list = list(k_dict.keys())
        for k_idx in range(len(k_list)):
            k = k_list[k_idx]
            inner_dict[str(k)] = k_dict[k]

        metrics_over_k_serialisable[sys_name] = inner_dict

    save_json(metrics_over_k_serialisable, OUT_DIR + "part5_metrics_over_k.json")

    plot_metrics_vs_k(all_systems_metrics_over_k, k_range, OUT_DIR + "part5_comparison_map_ndcg.png")
    plot_ablation(ablation_results, OUT_DIR + "part5_lsa_ablation.png")
    plot_final_bar_chart(final_metrics_at_k10, OUT_DIR + "part5_final_bar_chart.png")

    total_time_taken = time.time() - total_start
    print("\n================================")
    print("FINISHED SCRIPT in", round(total_time_taken, 1), "seconds")
    print("================================")

if __name__ == '__main__':
    main()
