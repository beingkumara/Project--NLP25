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
    print("Loading data...")
    dataset_path = 'cranfield/'
    
    # load files required for the flaw finding
    all_queries = load_json(dataset_path + "cran_queries.json")
    all_docs = load_json(dataset_path + "cran_docs.json")
    all_qrels = load_json(dataset_path + "cran_qrels.json")
    
    # avoid list comprehensions
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
    
    # Store doc text and query text in dictionary for easy retrieval later
    # using simple loop instead of dict comprehension
    doc_text_dict = {}
    for d in all_docs:
        doc_text_dict[d["id"]] = d["body"]
        
    query_text_dict = {}
    for q in all_queries:
        query_text_dict[int(q["query number"])] = q["query"]

    print("Preprocessing...")
    # instantiate the required modules
    seg = SentenceSegmentation()
    tok = Tokenization()
    red = InflectionReduction()
    stop = StopwordRemoval()

    # pre-processing all queries
    processed_queries = []
    for q in query_text_list:
        clean_q = preprocess_text(q, seg, tok, red, stop)
        processed_queries.append(clean_q)

    # pre-processing all documents
    processed_docs = []
    for d in doc_text_list:
        clean_d = preprocess_text(d, seg, tok, red, stop)
        processed_docs.append(clean_d)

    print("Building indexing and ranking...")
    # creating index and finding ranks
    ir_system = InformationRetrieval()
    ir_system.buildIndex(processed_docs, doc_ids_list)
    
    # ranking all queries at once
    doc_IDs_ordered_list = ir_system.rank(processed_queries)

    print("Finding VSM flaws...")
    # Find a relevant doc ranked > 100 to show the flaw of our system
    for qrel in all_qrels:
        q_id = int(qrel["query_num"])
        d_id = int(qrel["id"])
        pos = qrel["position"]
        
        # position 1 or 2 means the document is highly relevant according to ground truth
        if pos in [1, 2]: 
            # Find rank of d_id for our calculated q_id
            q_idx = query_ids_list.index(q_id)
            ranks = doc_IDs_ordered_list[q_idx]
            
            # making sure the document is returned in our ranks
            if d_id in ranks:
                # index + 1 gives the rank (1-indexed)
                computed_rank = ranks.index(d_id) + 1
                
                # I am checking if our system gave it a very bad rank, like > 130
                if computed_rank > 100:
                    print(f"\nFound an example! Query {q_id}, Doc {d_id}")
                    print(f"Ground truth rank: {pos}, Our system rank: {computed_rank}")
                    print(f"--- Query {q_id} text ---")
                    print(query_text_dict[q_id])
                    print(f"--- Doc {d_id} text ---")
                    print(doc_text_dict[d_id])
                    break # We just need one good example to show the flaw, no need to loop all

if __name__ == '__main__':
    main()
