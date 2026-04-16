import json

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval

from util import load_json, preprocess_text

def main():
    print("Loading data for OOV test...")
    dataset_path = 'cranfield/'
    # reading the documents json file
    all_docs = load_json(dataset_path + "cran_docs.json")
    
    # I cannot use list comprehension, writing normal loop instead
    doc_ids_list = []
    for d in all_docs:
        doc_ids_list.append(d["id"])
        
    doc_text_list = []
    for d in all_docs:
        doc_text_list.append(d["body"])

    # creating objects for all preprocessing classes
    seg = SentenceSegmentation()
    tok = Tokenization()
    red = InflectionReduction()
    stop = StopwordRemoval()

    print("Preprocessing docs... this might take some time")
    processed_docs = []
    # applying preprocessing to each document body
    for d in doc_text_list:
        clean_doc = preprocess_text(d, seg, tok, red, stop)
        processed_docs.append(clean_doc)

    # initialize ir system and build index
    ir_system = InformationRetrieval()
    ir_system.buildIndex(processed_docs, doc_ids_list)

    # Test Query 1: Mixed OOV and In-Vocabulary
    # 'aerodynamics' might be there, but 'intergalactic starships' is definitely not
    query1 = "aerodynamics of intergalactic starships"
    
    # Test Query 2: Pure OOV
    # these words are completely random and alien to cranfield
    query2 = "intergalactic starships flippity floppity"

    print(f"\nEvaluating Query 1: '{query1}'")
    processed_q1 = preprocess_text(query1, seg, tok, red, stop)
    
    try:
        ranked_docs_1 = ir_system.rank([processed_q1])
        print("Query 1 successfully returned a ranking!")
        # I am just printing the top 5 to check
        # print("checking ranked docs size:", len(ranked_docs_1[0]))
        print(f"Top 5 doc IDs: {ranked_docs_1[0][:5]}")
    except Exception as e:
        print(f"Query 1 failed with error: {e}")

    print(f"\nEvaluating Query 2: '{query2}'")
    processed_q2 = preprocess_text(query2, seg, tok, red, stop)
    
    try:
        ranked_docs_2 = ir_system.rank([processed_q2])
        print("Query 2 successfully returned a ranking!")
        # If all weights are 0, documents might be returned randomly or not at all depending on ranking sort
        print(f"Top 5 doc IDs: {ranked_docs_2[0][:5]}") 
    except Exception as e:
        # if the system crashes due to dimension mismatch or empty vector it will be caught here
        print(f"Query 2 failed with error: {e}")

if __name__ == '__main__':
    main()
