"""
part5.py
=========
Part 5: Improving the IR System

This script runs a full comparative evaluation of four retrieval systems:
  - Baseline : TF-IDF VSM (from Part 2)
  - Method A : BM25 (Okapi BM25, manually implemented)
  - Method B : LSA  (TF-IDF + Truncated SVD, ablation over k)
  - Method C : VSM + WordNet Query Expansion

Evaluation is at k=10 on the Cranfield dataset using:
  Precision@10, Recall@10, F0.5@10, MAP@10, nDCG@10

Statistical significance testing:
  Wilcoxon Signed-Rank Test (scipy.stats.wilcoxon) on per-query AP scores.
  H0: method X and baseline have the same distribution of AP scores.
  We reject H0 (claim improvement is real) at p < 0.05.

Ablation study:
  LSA is evaluated for n_components in [50, 100, 200, 300].
  The best k (by MAP@10) is selected and compared against the other methods.

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
    """Compute all five metrics for a single system at cutoff k."""
    prec  = evaluator.meanPrecision(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    rec   = evaluator.meanRecall(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    fs    = evaluator.meanFscore(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    m_map = evaluator.meanAveragePrecision(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    ndcg  = evaluator.meanNDCG(doc_IDs_ordered_dict, query_ids, relevant_docs, k)
    return {"precision": prec, "recall": rec, "fscore": fs, "map": m_map, "ndcg": ndcg}



# ===========================================================================
# WILCOXON HYPOTHESIS TEST
# ===========================================================================

def wilcoxon_test(ap_baseline, ap_method, method_name):
    """
    Perform a two-sided Wilcoxon Signed-Rank test between two paired
    distributions of per-query AP scores.

    H0: The two systems have the same distribution of AP scores.
    We reject H0 at p < 0.05 and conclude the improvement is significant.

    Returns a dict with the test statistic, p-value, and interpretation.
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
            "statistic": None,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Both systems have identical performance."
        }

    stat, p_value = stats.wilcoxon(ap_method, ap_baseline, alternative='greater')

    # Find mean of delta
    sum_diff = 0.0
    for d in differences:
        sum_diff = sum_diff + d
    mean_delta = float(sum_diff) / float(len(differences))
    
    significant = False
    if p_value < 0.05:
        significant = True

    if significant:
        interpretation = str(method_name) + " IS significantly better than VSM baseline (p=" + str(round(p_value, 4)) + ", mean delta MAP=" + str(round(mean_delta, 4)) + ")."
    else:
        interpretation = str(method_name) + " is NOT significantly better than VSM baseline (p=" + str(round(p_value, 4)) + ", mean delta MAP=" + str(round(mean_delta, 4)) + ")."

    return {
        "method": method_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_delta_ap": mean_delta,
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

    line_styles = ['-o', '-s', '-^', '-D', '-x']
    
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
    Bar chart comparing Precision, Recall, MAP, nDCG at k=10
    for all final systems side by side.
    """
    systems = list(final_metrics.keys())
    metric_keys = ["precision", "recall", "map", "ndcg"]
    metric_labels = ["Precision@10", "Recall@10", "MAP@10", "nDCG@10"]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(systems)):
        sys_name = systems[i]
        vals = []
        for m_idx in range(len(metric_keys)):
            m = metric_keys[m_idx]
            vals.append(final_metrics[sys_name][m])
            
        offset = (i - len(systems) / 2.0 + 0.5) * width
        ax.bar(x + offset, vals, width, label=sys_name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Final calculated score")
    ax.set_title("Scores at cutoff k=10 across systems")
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved plot at:", out_path)

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
# HYPERPARAMETER TUNING K-FOLD CV
# ===========================================================================

def run_bm25_cross_validation(processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, k_eval=10):
    print("\n  Executing 5-Fold Cross Validation for hyperparameter tuning...")
    # Here we partition queries manually to rigorously avoid overfitting on the test set, as taught in theory.
    num_queries = len(query_ids)
    fold_size = int(num_queries / 5)
    
    k1_options = [1.2, 1.5, 2.0]
    b_options = [0.5, 0.75, 0.9]
    
    fold_maps = []
    
    for fold_index in range(5):
        # Let us calculate the indices manually
        start_idx = fold_index * fold_size
        end_idx = start_idx + fold_size
        if fold_index == 4:
            # Handle final remainder safely
            end_idx = num_queries
            
        test_q_ids = []
        train_q_ids = []
        for q_index in range(num_queries):
            if q_index >= start_idx and q_index < end_idx:
                test_q_ids.append(query_ids[q_index])
            else:
                train_q_ids.append(query_ids[q_index])
                
        # Now we find the best params on train split only
        best_train_map = -1.0
        best_k1 = 1.5
        best_b = 0.75
        
        for k_idx in range(len(k1_options)):
            current_k1 = k1_options[k_idx]
            for b_idx in range(len(b_options)):
                current_b = b_options[b_idx]
                
                # We initialize a temporary BM25 instance natively
                temp_bm25 = InformationRetrievalBM25(k1=current_k1, b=current_b)
                temp_bm25.buildIndex(processed_docs, doc_ids)
                
                # Only test on training data
                train_queries_data = []
                for q_idx2 in range(len(query_ids)):
                    if query_ids[q_idx2] in train_q_ids:
                        train_queries_data.append(processed_queries[q_idx2])
                        
                temp_ranked = temp_bm25.rank(train_queries_data)
                temp_dict = ranked_list_to_dict(temp_ranked, train_q_ids)
                
                temp_map = evaluator.meanAveragePrecision(temp_dict, train_q_ids, relevant_docs, k_eval)
                
                if temp_map > best_train_map:
                    best_train_map = temp_map
                    best_k1 = current_k1
                    best_b = current_b
                    
        print("    Fold", fold_index+1, ": Best Params (k1=", best_k1, "b=", best_b, "). Now blindly testing...")
        
        # Now testing properly exclusively on out-of-fold test queries
        final_bm25 = InformationRetrievalBM25(k1=best_k1, b=best_b)
        final_bm25.buildIndex(processed_docs, doc_ids)
        
        test_queries_data = []
        for q_idx2 in range(len(query_ids)):
            if query_ids[q_idx2] in test_q_ids:
                test_queries_data.append(processed_queries[q_idx2])
                
        test_ranked = final_bm25.rank(test_queries_data)
        test_dict = ranked_list_to_dict(test_ranked, test_q_ids)
        test_map = evaluator.meanAveragePrecision(test_dict, test_q_ids, relevant_docs, k_eval)
        
        fold_maps.append(test_map)
        
    overall_avg_map = 0.0
    for m in fold_maps:
        overall_avg_map = overall_avg_map + m
    overall_avg_map = overall_avg_map / 5.0
    
    print("  Overall robust Cross-Validated test theoretical MAP@10 is mathematically:", round(overall_avg_map, 4))


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

    relevant_docs = get_relevant_docs(all_qrels)
    
    # Ensuring no errors if query ID is not in relevant_docs
    for q_idx in range(len(query_ids)):
        q_id = query_ids[q_idx]
        if q_id not in relevant_docs:
            relevant_docs[q_id] = []

    print("  Loaded", len(all_queries), "queries and", len(all_docs), "documents.")

    # ── Preprocess ──────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing...")
    processed_queries, processed_docs, stop_obj = preprocess_all(query_texts, doc_texts)

    evaluator = Evaluation()
    k_range = list(range(1, EVAL_K + 1))

    # ── HYPERPARAMETER SEARCH CV ───────────────────────────────────────────
    run_bm25_cross_validation(processed_queries, processed_docs, doc_ids, query_ids, relevant_docs, evaluator, EVAL_K)

    # We collect metrics-over-k for each final system (for the line plots)
    # and per-query AP at k=10 for Wilcoxon tests.
    all_systems_metrics_over_k = {}  # sys_name -> {k -> metrics_dict}
    final_metrics_at_k10       = {}  # sys_name -> metrics_dict at k=10
    per_query_ap_at_k10        = {}  # sys_name -> list of per-query AP values

    # ── BASELINE: TF-IDF VSM ────────────────────────────────────────────────
    print("\n[3/6] Running BASELINE (TF-IDF VSM)...")
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

    print("  Calculated MAP@10 =", round(vsm_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(vsm_metrics_k[EVAL_K]['ndcg'], 4))

    # ── METHOD A: BM25 ───────────────────────────────────────────────────────
    print("\n[4a/6] Running METHOD A (BM25, k1=1.5, b=0.75)...")
    t0 = time.time()
    bm25 = InformationRetrievalBM25(k1=1.5, b=0.75)
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

    print("  Calculated MAP@10 =", round(bm25_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(bm25_metrics_k[EVAL_K]['ndcg'], 4))

    # ── METHOD B: LSA — Ablation Study ──────────────────────────────────────
    print("\n[4b/6] Running METHOD B (LSA) — Ablation over n_components...")
    ablation_results = {}
    best_lsa_k   = None
    best_lsa_map = -1
    best_lsa_dict = None

    baseline_map_k10  = vsm_metrics_k[EVAL_K]["map"]
    baseline_ndcg_k10 = vsm_metrics_k[EVAL_K]["ndcg"]

    for i in range(len(LSA_K_VALUES)):
        lsa_k = LSA_K_VALUES[i]
        print("  Evaluating LSA with k =", lsa_k)
        t0 = time.time()
        lsa = InformationRetrievalLSA(n_components=lsa_k)
        lsa.buildIndex(processed_docs, doc_ids)
        ranked_lsa = lsa.rank(processed_queries)
        lsa_dict   = ranked_list_to_dict(ranked_lsa, query_ids)
        lsa_m      = compute_all_metrics(evaluator, lsa_dict, query_ids, relevant_docs, EVAL_K)
        print("  MAP@10 =", round(lsa_m['map'], 4), "| nDCG@10 =", round(lsa_m['ndcg'], 4), "(took", round(time.time()-t0, 1), "s)")

        ablation_results[lsa_k] = lsa_m

        if lsa_m["map"] > best_lsa_map:
            best_lsa_map        = lsa_m["map"]
            best_lsa_k          = lsa_k
            best_lsa_dict       = lsa_dict
            best_lsa_metrics_k  = None  # recompute below for all k values

    # Recompute best LSA over all k for the line plot
    print(f"  Best LSA n_components = {best_lsa_k} (MAP@10 = {best_lsa_map:.4f})")
    print(f"  Re-running LSA k={best_lsa_k} over all k values for plots...")
    lsa_best = InformationRetrievalLSA(n_components=best_lsa_k)
    lsa_best.buildIndex(processed_docs, doc_ids)
    ranked_lsa_best = lsa_best.rank(processed_queries)
    lsa_best_dict   = ranked_list_to_dict(ranked_lsa_best, query_ids)

    lsa_metrics_k = {}
    for k_idx in range(len(k_range)):
        k = k_range[k_idx]
        lsa_metrics_k[k] = compute_all_metrics(evaluator, lsa_best_dict, query_ids, relevant_docs, k)

    lsa_sys_name = "LSA (k=" + str(best_lsa_k) + ")"
    all_systems_metrics_over_k[lsa_sys_name] = lsa_metrics_k
    final_metrics_at_k10[lsa_sys_name]       = lsa_metrics_k[EVAL_K]
    per_query_ap_at_k10[lsa_sys_name]        = compute_per_query_ap(
        evaluator, lsa_best_dict, query_ids, relevant_docs, EVAL_K)

    # Add baseline references to ablation dict for the ablation plot
    ablation_results["baseline_map"]  = baseline_map_k10
    ablation_results["baseline_ndcg"] = baseline_ndcg_k10

    # ── METHOD C: VSM + Query Expansion ─────────────────────────────────────
    print("\n[4c/6] Running METHOD C (VSM + WordNet Query Expansion)...")
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

    print("  Calculated MAP@10 =", round(exp_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(exp_metrics_k[EVAL_K]['ndcg'], 4))

    # ── METHOD D: Pseudo-Relevance Feedback (PRF) ───────────────────────────
    print("\n[4d/6] Running METHOD D (PRF)...")
    t0 = time.time()
    prf = InformationRetrievalPRF(k1=1.5, b=0.75)
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

    print("  Calculated MAP@10 =", round(prf_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(prf_metrics_k[EVAL_K]['ndcg'], 4))

    # ── METHOD E: Explicit Semantic Analysis (ESA) ──────────────────────────
    print("\n[4e/6] Running METHOD E (ESA — WordNet Concept Space)...")
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

    print("  Calculated MAP@10 =", round(esa_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(esa_metrics_k[EVAL_K]['ndcg'], 4))

    # ── METHOD F: Explicit Semantic Analysis (ESA - Cranfield Space) ─────────
    print("\n[4f/6] Running METHOD F (ESA — Cranfield Concept Space)...")
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

    print("  Calculated MAP@10 =", round(esac_metrics_k[EVAL_K]['map'], 4), "| nDCG@10 =", round(esac_metrics_k[EVAL_K]['ndcg'], 4))

    # ── WILCOXON HYPOTHESIS TESTS ────────────────────────────────────────────
    print("\n[5/6] Running Wilcoxon Signed-Rank Tests...")
    baseline_ap = per_query_ap_at_k10["VSM (Baseline)"]
    wilcoxon_results = []

    systems_to_test = ["BM25", lsa_sys_name, "VSM + QE", "BM25 + PRF", "ESA", "ESA (Cranfield)"]
    for s_idx in range(len(systems_to_test)):
        sys_name = systems_to_test[s_idx]
        result = wilcoxon_test(baseline_ap, per_query_ap_at_k10[sys_name], sys_name)
        wilcoxon_results.append(result)
        
        sig_str = "not significant"
        if result["significant"]:
            sig_str = "SIGNIFICANT"
            
        print("  System:", sys_name, "| p-value:", round(result['p_value'], 4), "|", sig_str)

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

    print("\n[Step 7] Writing outputs to disk...")

    save_json(final_metrics_at_k10, OUT_DIR + "part5_final_metrics_k10.json")

    ablation_serialisable = {}
    for k in ablation_results.keys():
        ablation_serialisable[str(k)] = ablation_results[k]
    save_json(ablation_serialisable, OUT_DIR + "part5_lsa_ablation.json")

    save_json(wilcoxon_results, OUT_DIR + "part5_wilcoxon_tests.json")
    save_json(case_studies, OUT_DIR + "part5_case_studies.json")

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