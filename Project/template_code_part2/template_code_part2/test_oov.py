import json

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval

# helper function to load json files that i got from reference
def load_json(file_path):
    f_in = open(file_path, 'r')
    read_data = json.load(f_in)
    f_in.close()
    return read_data

# this function will apply all the nlp steps one after another
def preprocess_text(text, _segmenter, _tokenizer, _reducer, _stop_remover):
    # first we segment into sentences
    sent_list = _segmenter.punkt(text)
    # then tokenize
    tok_list = _tokenizer.pennTreeBank(sent_list)
    # apply inflection reduction like stemming
    red_list = _reducer.reduce(tok_list)
    # finally remove stop words so we have clean tokens
    return _stop_remover.fromList(red_list)

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
