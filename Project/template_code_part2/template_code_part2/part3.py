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


def load_json(file_path):
    # I wrote this small function just to load json files easily.
    # Basically it opens the file and reads data.
    f_in = open(file_path, 'r')
    read_data = json.load(f_in)
    f_in.close()
    return read_data


def get_relevant_docs(qrels):
    # This function goes through the qrels data to find relevant doc ids for each query.
    # The question mentions that position 1 up to 4 means the document is relevant.
    # So I have to put condition to check if position is in that range.
    # At first I was confused how to store them, but then decided dictionary is easiest.
    # key will be query id and value will be list of relevant doc ids.
    
    dict_relevant = {}

    # Going through all entries one by one in the qrels list
    for i in range(len(qrels)):
        curr_entry = qrels[i]
        
        # When I printed query_num I saw it was string! So I have to convert to int here
        # so it can match the query ids from cran_queries later on.
        q_id = int(curr_entry["query_num"])
        d_id = int(curr_entry["id"])
        pos = curr_entry["position"]

        # If query id is not present in my dictionary, create empty list
        if q_id not in dict_relevant:
            dict_relevant[q_id] = []

        # I have to check if position is 1, 2, 3 or 4.
        if pos == 1 or pos == 2 or pos == 3 or pos == 4:
            # I must check if document is already there so no duplicates get added.
            is_present = False
            for prev_doc in dict_relevant[q_id]:
                if prev_doc == d_id:
                    is_present = True
            
            if is_present == False:
                dict_relevant[q_id].append(d_id)

    # print("Count of queries that have true relevance:", len(dict_relevant))

    return dict_relevant


def preprocess_text(text, _segmenter, _tokenizer, _reducer, _stop_remover):
    # This function takes normal text and completely processes it.
    # I am going step by step: segmenting, tokenizing, reducing inflection, and removing stop words.
    # I am passing the objects so I don't have to create them again and again, which saves time.

    # 1. break text into sentences
    sent_list = _segmenter.punkt(text)

    # 2. break sentences into tokens
    tok_list = _tokenizer.pennTreeBank(sent_list)

    # 3. do inflection reduction like lemmatize
    red_list = _reducer.reduce(tok_list)

    # 4. remove all the stop words
    clean_list = _stop_remover.fromList(red_list)

    return clean_list


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

    # Now I need to separate out query ids and the query text into separate lists
    query_ids_list = []
    query_text_list = []
    for i in range(len(all_queries)):
        query_ids_list.append(all_queries[i]["query number"])
        query_text_list.append(all_queries[i]["query"])

    # Doing the exact same thing for documents
    doc_ids_list = []
    doc_text_list = []
    for i in range(len(all_docs)):
        doc_ids_list.append(all_docs[i]["id"])
        doc_text_list.append(all_docs[i]["body"])

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

    # At first I got a bug because rank returns a list of lists, but my evaluation
    # needs a dictionary with query_id as key. So I have to map them here manually.
    dict_doc_IDs_ordered = {}
    for i in range(len(query_ids_list)):
        curr_qid = query_ids_list[i]
        dict_doc_IDs_ordered[curr_qid] = doc_IDs_ordered_list[i]

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
        
        # Adding to logic lists
        list_prec.append(m_prec)
        list_rec.append(m_rec)
        list_fs.append(m_fs)
        list_map.append(m_map)
        list_ndcg.append(m_ndcg)

        print("For k =", my_k)
        print("  Precision: ", round(m_prec, 4))
        print("  Recall:    ", round(m_rec, 4))
        print("  F-score:   ", round(m_fs, 4))
        print("  MAP:       ", round(m_map, 4))
        print("  nDCG:      ", round(m_ndcg, 4))
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
    
    f_json_out = open(my_out_dir + "metrics_part3.json", 'w')
    json.dump(json_data, f_json_out, indent=4)
    f_json_out.close()

    # Stopping the timer here to see how much time the system took.
    end_time = time.time()
    total_time = end_time - start_time
    print("The total run time of my IR system is:", total_time, "seconds")

if __name__ == '__main__':
    main()