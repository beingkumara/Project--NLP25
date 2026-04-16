# util.py
# ======
# Utility functions used across the project modules.
# This file contains helper functions that are needed by the preprocessing
# and other modules in the IR system.

from nltk.corpus import wordnet
import json
import math

def get_wordnet_pos(treebank_tag):
    """
    Convert a Penn Treebank POS tag to a WordNet POS tag.
    Useful for WordNet Lemmatizer (which needs wordnet.NOUN, VERB, ADJ, ADV).

    Parameters
    ----------
    treebank_tag : str
        A Penn Treebank POS tag, e.g. 'NN', 'VBZ', 'JJ', 'RB'.

    Returns
    -------
    str
        The corresponding WordNet POS constant.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun if tag is unrecognised
        return wordnet.NOUN

def load_json(file_path):
    """
    This function is responsible for loading JSON data from a given file path.
    As we learned in class, reading files efficiently is important.
    Here we will open the file, read the data using json module, and then return it.
    """
    file_in = open(file_path, 'r')
    read_data = json.load(file_in)
    file_in.close()
    return read_data

def save_json(data_to_save, target_path):
    """
    This function will save our dictionary or list data into a JSON file format.
    We iterate basically and save the file properly with indents so it is readable.
    """
    file_out = open(target_path, 'w')
    json.dump(data_to_save, file_out, indent=4)
    file_out.close()
    print("  Saved the file at:", target_path)

def get_relevant_docs(qrels_list):
    """
    This function goes through the qrels data to find relevant doc ids for each query.
    As per theoretical guidelines of our assignment, a document is considered relevant
    if its position is 1, 2, 3, or 4.
    Hence we verify the condition and return a dictionary mapping of queries to relevant docs.
    """
    dict_relevant = {}
    
    # We will iterate through all items in the qrels list
    for i in range(len(qrels_list)):
        curr_entry = qrels_list[i]
        
        # We must convert to integer because sometimes they are read as string
        query_id = int(curr_entry["query_num"])
        doc_id = int(curr_entry["id"])
        position = curr_entry["position"]

        # If we have not seen this query before, initialize an empty list
        if query_id not in dict_relevant:
            dict_relevant[query_id] = []

        # Checking the base condition for relevance as studied in theory
        if position == 1 or position == 2 or position == 3 or position == 4:
            # We must verify to avoid adding duplicates
            is_found = False
            for prev_doc in dict_relevant[query_id]:
                if prev_doc == doc_id:
                    is_found = True
            
            # If not found, then we add it to our relevance list
            if is_found == False:
                dict_relevant[query_id].append(doc_id)
                
    return dict_relevant

def preprocess_text(text_to_process, segmenter_obj, tokenizer_obj, reducer_obj, stop_remover_obj):
    """
    This function takes normal text and completely processes it for our Information Retrieval system.
    We process step by step based on standard NLP theory: 
    segmentation -> tokenization -> inflection reduction -> stop word removal.
    Therefore, we use this logic to pass objects so it avoids re-initializing classes.
    """
    # First, break text into sentences using the segmenter
    sentences_list = segmenter_obj.punkt(text_to_process)

    # Second, break sentences into tokens using the treebank tokenizer
    tokens_list = tokenizer_obj.pennTreeBank(sentences_list)

    # Third, do inflection reduction like lemmatization or stemming
    reduced_list = reducer_obj.reduce(tokens_list)

    # Fourth, remove all the stop words to keep only meaningful words
    clean_list = stop_remover_obj.fromList(reduced_list)

    return clean_list


# ============================================================================
# SHARED FUNCTIONS FOR IR MODULES
# ============================================================================
# These functions are commonly used across multiple IR files.
# Instead of writing the same code again and again, we keep them here
# so all the modules can just import and call them directly.
# ============================================================================


def build_inverted_index(docs, docIDs):
    """
    This function builds the inverted index from the documents.
    It goes through every document, every sentence, and every word
    and counts how many times each word appears in each document.
    We need this in almost every IR module, so keeping it here.

    Parameters
    ----------
    docs : list
        A 3D list where docs[i] is a document, docs[i][j] is a sentence, 
        and docs[i][j][k] is a token.
    docIDs : list
        A list of integers representing the IDs of the documents.

    Returns
    -------
    dict
        A dictionary (the inverted index) mapping words to another 
        dictionary of {docID: frequency}.
    """
    index = {}

    # We iterate through all documents one by one
    for doc_index in range(len(docs)):
        d_id = docIDs[doc_index]

        # Going inside the document sentence by sentence
        for sent_index in range(len(docs[doc_index])):
            sentence = docs[doc_index][sent_index]

            # Getting each word from the sentence
            for word_index in range(len(sentence)):
                word = sentence[word_index]

                # Creating a dictionary for the word if it is coming first time
                if word not in index:
                    index[word] = {}

                # Setting initial frequency of this word in this document to 0
                if d_id not in index[word]:
                    index[word][d_id] = 0

                # Increment the count
                index[word][d_id] = index[word][d_id] + 1
                # print("word:", word, "found in doc:", d_id)

    return index


def compute_idf_values(index, total_documents):
    """
    This function computes the IDF (Inverse Document Frequency) for all words.
    Formula from class: IDF = log10(N / df)
    where df is the number of documents containing that word.

    Parameters
    ----------
    index : dict
        The inverted index built using build_inverted_index.
    total_documents : int
        The total number of documents in the corpus.

    Returns
    -------
    dict
        A dictionary mapping each word to its float IDF value.
    """
    idf_values = {}

    for word in index:
        # df is document frequency, which is just the number of docs containing this word
        document_frequency = len(index[word])

        if document_frequency > 0:
            idf_values[word] = math.log10(float(total_documents) / float(document_frequency))
        else:
            idf_values[word] = 0.0

    return idf_values


def compute_max_tf_per_doc(index):
    """
    This function finds the maximum term frequency in every document.
    We need this to calculate augmented TF later: TF_aug = 0.5 + 0.5*(tf/max_tf)
    This prevents longer documents from dominating shorter ones unfairly.

    Parameters
    ----------
    index : dict
        The inverted index containing word frequencies per document.

    Returns
    -------
    dict
        A dictionary mapping each docID to its highest raw term frequency.
    """
    max_tf_per_doc = {}

    for word in index:
        for d_id in index[word]:
            raw_tf = index[word][d_id]

            if d_id not in max_tf_per_doc:
                max_tf_per_doc[d_id] = raw_tf
            else:
                if raw_tf > max_tf_per_doc[d_id]:
                    max_tf_per_doc[d_id] = raw_tf

    return max_tf_per_doc


def count_query_words(query):
    """
    This function flattens a query (which is a list of sentences, where each
    sentence is a list of words) and counts the frequency of each word.
    Also returns the maximum frequency found among query words.
    We need this in every rank() method.

    Parameters
    ----------
    query : list
        A list of lists (sentences) containing tokens.

    Returns
    -------
    tuple
        A tuple (word_counts, max_tf) where word_counts is a dict of word 
        frequencies and max_tf is the highest frequency found.
    """
    word_counts = {}

    # Iterating through each sentence in the query
    for sent_idx in range(len(query)):
        sentence = query[sent_idx]

        # Iterating through each word in the sentence
        for word_idx in range(len(sentence)):
            word = sentence[word_idx]

            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] = word_counts[word] + 1

    # Now finding the maximum term frequency in the query
    max_tf = 0
    for word in word_counts:
        if word_counts[word] > max_tf:
            max_tf = word_counts[word]

    # Avoiding division by zero later, setting to 1 if query was empty
    if max_tf == 0:
        max_tf = 1

    return word_counts, max_tf


def sort_and_complete_ranking(doc_scores, all_docIDs):
    """
    This function takes a dictionary of document scores and sorts them
    in descending order. Then it appends any documents that had
    no score at all (zero similarity) at the end.
    This pattern is used in almost every IR module's rank() method.

    Parameters
    ----------
    doc_scores : dict
        A dictionary mapping docIDs to their similarity scores.
    all_docIDs : list
        A list of all document IDs in the dataset.

    Returns
    -------
    list
        A list of docIDs sorted by relevance (highest score first).
    """
    # Preparing the list to sort documents by their accumulated scores
    score_list = []
    for d_id in doc_scores:
        score_val = doc_scores[d_id]
        # putting score first so sorting works on score directly
        score_list.append((score_val, d_id))

    # Descending order sorting
    score_list.sort(reverse=True)

    # Extracting just the doc IDs in sorted order
    ordered_doc_ids = []
    for item_idx in range(len(score_list)):
        ordered_doc_ids.append(score_list[item_idx][1])

    # Adding documents that did not match any query word at the end
    # because they have 0 similarity score
    for d_id in all_docIDs:
        if d_id not in doc_scores:
            ordered_doc_ids.append(d_id)

    return ordered_doc_ids


def compute_vector_norms(concept_vectors, docIDs):
    """
    This function computes the L2 norm (magnitude) of each document concept vector.
    We need this for cosine similarity computation in ESA modules.
    The concept_vectors is a dictionary: docID -> {concept_name: weight}

    Parameters
    ----------
    concept_vectors : dict
        A nested dictionary of document concept weights.
    docIDs : list
        The list of all document IDs.

    Returns
    -------
    dict
        A dictionary mapping each docID to its vector magnitude (norm).
    """
    norms = {}

    for d_id in docIDs:
        sum_of_squares = 0.0
        concept_vec = concept_vectors[d_id]

        for concept_name in concept_vec:
            weight = concept_vec[concept_name]
            sum_of_squares = sum_of_squares + (weight * weight)

        if sum_of_squares > 0:
            norms[d_id] = math.sqrt(sum_of_squares)
        else:
            norms[d_id] = 0.0

    return norms


def cosine_similarity_sparse(query_vec, doc_vec, query_norm, doc_norm):
    """
    This function computes the cosine similarity between two sparse vectors
    represented as dictionaries. We compute the dot product first and then
    divide by the product of their norms.

    Parameters
    ----------
    query_vec : dict
        Sparse vector for the query {item: weight}.
    doc_vec : dict
        Sparse vector for the document {item: weight}.
    query_norm : float
        The precomputed magnitude of the query vector.
    doc_norm : float
        The precomputed magnitude of the document vector.

    Returns
    -------
    float
        The calculated cosine similarity score.
    """
    # Computing dot product
    dot_product = 0.0
    for key in query_vec:
        if key in doc_vec:
            dot_product = dot_product + (query_vec[key] * doc_vec[key])

    # Preventing division by zero
    if query_norm > 0 and doc_norm > 0:
        cosine_sim = dot_product / (query_norm * doc_norm)
    else:
        cosine_sim = 0.0

    return cosine_sim


def parse_cranfield_data(queries_json, docs_json):
    """
    This function parses the Cranfield dataset JSON lists and separates
    the IDs and text content into individual lists.
    We use this in part3.py and part5.py to avoid writing the same parsing loop.

    Parameters
    ----------
    queries_json : list
        The list of dictionaries loaded from cran_queries.json.
    docs_json : list
        The list of dictionaries loaded from cran_docs.json.

    Returns
    -------
    tuple
        Four lists: (query_ids, query_texts, doc_ids, doc_texts).
    """
    query_ids = []
    query_texts = []
    for i in range(len(queries_json)):
        query_ids.append(queries_json[i]["query number"])
        query_texts.append(queries_json[i]["query"])

    doc_ids = []
    doc_texts = []
    for i in range(len(docs_json)):
        doc_ids.append(docs_json[i]["id"])
        doc_texts.append(docs_json[i]["body"])

    return query_ids, query_texts, doc_ids, doc_texts


def ranked_list_to_dict(ranked_list, query_ids):
    """
    This function converts a ranked results list (list of lists) into a
    dictionary keyed by query ID. This is needed because our evaluation
    module expects a dictionary format, but the rank() methods return lists.

    Parameters
    ----------
    ranked_list : list
        A list of lists containing the ranked docIDs.
    query_ids : list
        The list of corresponding query IDs.

    Returns
    -------
    dict
        A mapping of {query_id: [ranked_doc_ids]}.
    """
    result_dict = {}

    for i in range(len(query_ids)):
        result_dict[query_ids[i]] = ranked_list[i]

    return result_dict