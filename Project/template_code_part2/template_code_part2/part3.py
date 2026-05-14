# part3.py
# I made this file to solve Question 3 of our NLP assignment.
# We need to find metrics like Precision, Recall, F-score, MAP, and nDCG for k=1 to 10.
# I am using cran_qrels.json to get the correct relevance ground truth.
# The assignment clearly said that if position is 1, 2, 3 or 4, it is relevant. So I have to check for that.
# In the end, I have to average all these metrics for every query and draw a plot showing metric against k.

import json
import os
import math
import time
import matplotlib
matplotlib.use('Agg')   # I had to add this so it does not open window and crash
import matplotlib.pyplot as plt

# These are the classes from other files of the project
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation


from util import load_json, get_relevant_docs, preprocess_text, parse_cranfield_data, ranked_list_to_dict


def main():
    # This is the main function for Question 3.
    # Basically it runs everything: loading data, building truth dictionary,
    # processing everything, building index, searching, computing metrics and then plotting.
    # I found this part very long to write but breaking it down helped me.

    # For Question 4, I need to report the run time of my IR system.
    # So I am starting the timer here using time module.
    start_time = time.time()

    print("I am loading Cranfield dataset...")

    my_dataset_path = 'cranfield/'
    my_out_dir = 'output/'

    # Checking if output folder is there, otherwise create it
    if not os.path.exists(my_out_dir):
        os.makedirs(my_out_dir)

    # Using my helper to load queries, docs, and the relevance ground truth
    all_queries = load_json(my_dataset_path + "cran_queries.json")
    all_docs = load_json(my_dataset_path + "cran_docs.json")
    all_qrels = load_json(my_dataset_path + "cran_qrels.json")

    print("I loaded", len(all_queries), "queries")
    print("I loaded", len(all_docs), "documents")
    print("I loaded", len(all_qrels), "qrels items")

    # Using my shared parsing function to load and separate data
    query_ids_list, query_text_list, doc_ids_list, doc_text_list = parse_cranfield_data(all_queries, all_docs)

    # Calling my function to build truth dict
    my_relevant_docs = get_relevant_docs(all_qrels)

    print("Setting up the NLP objects...")
    seg = SentenceSegmentation()
    tok = Tokenization()
    red = InflectionReduction()
    stop = StopwordRemoval()

    print("I am preprocessing queries now...")
    processed_queries = []
    for k in range(len(query_text_list)):
        res_q = preprocess_text(query_text_list[k], seg, tok, red, stop)
        processed_queries.append(res_q)


    print("Preprocessing documents... this takes a really long time!")
    processed_docs = []
    for k in range(len(doc_text_list)):
        res_d = preprocess_text(doc_text_list[k], seg, tok, red, stop)
        processed_docs.append(res_d)


    print("Building the index and calling rank function...")
    ir_system = InformationRetrieval()
    ir_system.buildIndex(processed_docs, doc_ids_list)
    doc_IDs_ordered_list = ir_system.rank(processed_queries)
    print("Done ranking!")

    # Using shared utility function for the list-to-dict conversion
    dict_doc_IDs_ordered = ranked_list_to_dict(doc_IDs_ordered_list, query_ids_list)

    # Also another issue I found: some queries don't have any relevant docs in json!
    # If I don't add them as empty list, I will get KeyError in evaluation.
    for qid in query_ids_list:
        if qid not in my_relevant_docs:
            my_relevant_docs[qid] = []

    # Creating evaluation object
    my_evaluator = Evaluation()

    # I will store the average metrics for each k in these lists
    list_prec = []
    list_rec = []
    list_fs = []
    list_map = []
    list_ndcg = []
    list_mrr = []

    print("\n-------------------------")
    print("FINAL RESULTS")
    print("-------------------------")

    # According to question, I need to check k from 1 to 10
    for my_k in range(1, 11):

        # Computing all metrics for current k
        m_prec = my_evaluator.meanPrecision(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)
        m_rec = my_evaluator.meanRecall(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)
        m_fs = my_evaluator.meanFscore(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)
        m_map = my_evaluator.meanAveragePrecision(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)
        m_ndcg = my_evaluator.meanNDCG(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)
        m_mrr = my_evaluator.meanReciprocalRank(dict_doc_IDs_ordered, query_ids_list, my_relevant_docs, my_k)

        # Adding to logic lists
        list_prec.append(m_prec)
        list_rec.append(m_rec)
        list_fs.append(m_fs)
        list_map.append(m_map)
        list_ndcg.append(m_ndcg)
        list_mrr.append(m_mrr)

        print("For k =", my_k)
        print("  Precision: ", round(m_prec, 4))
        print("  Recall:    ", round(m_rec, 4))
        print("  F-score:   ", round(m_fs, 4))
        print("  MAP:       ", round(m_map, 4))
        print("  nDCG:      ", round(m_ndcg, 4))
        print("  MRR:       ", round(m_mrr, 4))
        print("---")

    print("\nStarting to plot the graph...")
    
    # x axis will be 1 to 10
    k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Creating the plot
    my_fig, my_ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plotting each one with different lines so it's clear
    my_ax.plot(k_vals, list_prec, marker='o', label="Precision")
    my_ax.plot(k_vals, list_rec, marker='s', label="Recall")
    my_ax.plot(k_vals, list_fs, marker='^', label="F-score")
    my_ax.plot(k_vals, list_map, marker='D', label="MAP")
    my_ax.plot(k_vals, list_ndcg, marker='x', label="nDCG")
    my_ax.plot(k_vals, list_mrr, marker='*', label="MRR")

    # Adding labels
    my_ax.set_xlabel("k values")
    my_ax.set_ylabel("Scores")
    my_ax.set_title("Metrics")
    my_ax.set_xticks(k_vals)
    my_ax.legend(loc='best')
    my_ax.grid(True)

    # I will save the image so I can insert it in my report
    final_plot_path = my_out_dir + "eval_plot_part3.png"
    plt.tight_layout()
    plt.savefig(final_plot_path)
    print("I saved the plot to", final_plot_path)

    # Saving metrics array into json also just in case I need to copy numbers
    json_data = {}
    json_data["k"] = k_vals
    json_data["precision"] = list_prec
    json_data["recall"] = list_rec
    json_data["fscore"] = list_fs
    json_data["map"] = list_map
    json_data["ndcg"] = list_ndcg
    json_data["mrr"] = list_mrr

    f_json_out = open(my_out_dir + "metrics_part3.json", 'w')
    json.dump(json_data, f_json_out, indent=4)
    f_json_out.close()

    # Stopping the timer here to see how much time the system took.
    end_time = time.time()
    total_time = end_time - start_time
    print("The total run time of my IR system is:", total_time, "seconds")

if __name__ == '__main__':
    main()