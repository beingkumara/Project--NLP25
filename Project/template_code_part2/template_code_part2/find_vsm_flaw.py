import json
import os
import math
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from util import load_json, preprocess_text


def main():
    # Let us append the results
    # Basically the main function to run the logic
    print("Loading data...")
    dataset_path = "cranfield/"
    all_queries = load_json(dataset_path + "cran_queries.json")
    all_docs = load_json(dataset_path + "cran_docs.json")
    all_qrels = load_json(dataset_path + "cran_qrels.json")
    query_ids_list = []
    query_text_list = []
    for q in all_queries:
        query_ids_list.append(int(q["query number"]))
        query_text_list.append(q["query"])
    doc_ids_list = []
    doc_text_list = []
    for d in all_docs:
        doc_ids_list.append(d["id"])
        doc_text_list.append(d["body"])
    doc_text_dict = {}
    for d in all_docs:
        doc_text_dict[d["id"]] = d["body"]
    query_text_dict = {}
    for q in all_queries:
        query_text_dict[int(q["query number"])] = q["query"]
    print("Preprocessing...")
    seg = SentenceSegmentation()
    tok = Tokenization()
    red = InflectionReduction()
    stop = StopwordRemoval()
    processed_queries = []
    for q in query_text_list:
        clean_q = preprocess_text(q, seg, tok, red, stop)
        processed_queries.append(clean_q)
    processed_docs = []
    for d in doc_text_list:
        clean_d = preprocess_text(d, seg, tok, red, stop)
        processed_docs.append(clean_d)
    print("Building indexing and ranking...")
    ir_system = InformationRetrieval()
    ir_system.buildIndex(processed_docs, doc_ids_list)
    doc_IDs_ordered_list = ir_system.rank(processed_queries)
    print("Finding VSM flaws...")
    for qrel in all_qrels:
        q_id = int(qrel["query_num"])
        d_id = int(qrel["id"])
        pos = qrel["position"]
        if pos in [1, 2]:
            q_idx = query_ids_list.index(q_id)
            ranks = doc_IDs_ordered_list[q_idx]
            if d_id in ranks:
                computed_rank = ranks.index(d_id) + 1
                if computed_rank > 100:
                    print(f"\nFound an example! Query {q_id}, Doc {d_id}")
                    print(f"Ground truth rank: {pos}, Our system rank: {computed_rank}")
                    print(f"--- Query {q_id} text ---")
                    print(query_text_dict[q_id])
                    print(f"--- Doc {d_id} text ---")
                    print(doc_text_dict[d_id])
                    break


if __name__ == "__main__":
    main()
