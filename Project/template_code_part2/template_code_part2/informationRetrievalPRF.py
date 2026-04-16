"""
informationRetrievalPRF.py
===========================
Pseudo-Relevance Feedback (PRF) implementation for Part 5.

Pseudo-Relevance Feedback allows us to retrieve an initial baseline set of documents 
and we assume the top k of them are relevant. We then extract their dominant terms and
add them back to the original query to improve document recall.

However, a major disadvantage we must consider is Query Drift: if the very first retrieved
documents are irrelevant false positives, our algorithm will 
extract irrelevant vocabulary and our second-pass results will be much worse.
"""

from informationRetrievalBM25 import InformationRetrievalBM25
from util import count_query_words

class InformationRetrievalPRF:
    # Here we define our PRF system. We will simply wrap the Okapi BM25 model built previously.
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.base_bm25_model = InformationRetrievalBM25(k1=self.k1, b=self.b)
        
        # We also need a simple way to access document underlying words.
        # Since the assignment gives us preprocessed docs in buildIndex, we can safely save them locally.
        self.saved_docs_map = {}

    def buildIndex(self, docs, docIDs):
        """
        Build the index for PRF (uses a base IR system).

        Parameters
        ----------
        docs : list
            A 3D list of lists representing the documents.
        docIDs : list
            The corresponding list of document IDs.

        Returns
        -------
        None.
        """
        # We directly call the BM25 buildIndex function to avoid rewriting matrix logic.
        self.base_bm25_model.buildIndex(docs, docIDs)
        
        # Now we save the document contents for our feedback pass later.
        for index_counter in range(len(docs)):
            document_id = docIDs[index_counter]
            self.saved_docs_map[document_id] = docs[index_counter]

    def rank(self, queries):
        """
        Rank documents for each query using Pseudo-Relevance Feedback.

        Parameters
        ----------
        queries : list
            A 3D list of lists representing the queries.

        Returns
        -------
        list
            A list of lists containing ranked document IDs for each query.
        """
        # First we execute the standard baseline Pass 1 retrieval.
        first_pass_results = self.base_bm25_model.rank(queries)
        
        second_pass_queries = []
        
        # Now we iterate through every query to expand it using feedback.
        for q_index in range(len(queries)):
            current_query = queries[q_index]
            current_baseline_ranked_docs = first_pass_results[q_index]
            
            # Let us extract only the top 3 assumed documents from the initial first pass.
            # Base condition checking to entirely prevent any index out of bounds crashes.
            limit_docs = 3
            if len(current_baseline_ranked_docs) < 3:
                limit_docs = len(current_baseline_ranked_docs)
                
            top_pseudo_relevant_docs = []
            for i in range(limit_docs):
                top_pseudo_relevant_docs.append(current_baseline_ranked_docs[i])
                
            # Now we iterate and extract terms from these top identified documents
            term_frequencies = {}
            for doc_id in top_pseudo_relevant_docs:
                doc_content = self.saved_docs_map[doc_id]
                
                # Iterating over sentences and words manually
                for sent_idx in range(len(doc_content)):
                    sentence = doc_content[sent_idx]
                    for word_idx in range(len(sentence)):
                        word = sentence[word_idx]
                        if word not in term_frequencies:
                            term_frequencies[word] = 0
                        term_frequencies[word] = term_frequencies[word] + 1
                        
            # Using shared function to count words in the current query
            existing_query_word_counts, _ = count_query_words(current_query)
                    
            # Let us manually sort the extracted terms by their frequency in descending order
            list_of_terms = []
            for word in term_frequencies.keys():
                # Checking if word exists already in our query using the counted words
                if word not in existing_query_word_counts:
                    freq = term_frequencies[word]
                    list_of_terms.append((freq, word))
                    
            # Sorting tuples natively sorts basically based on the first element (frequency here).
            list_of_terms.sort(reverse=True)
            
            # We extract the top 3 new expansion terms. 
            extracted_new_terms = []
            limit_terms = 3
            if len(list_of_terms) < 3:
                limit_terms = len(list_of_terms)
                
            for k_term in range(limit_terms):
                extracted_new_terms.append(list_of_terms[k_term][1])
                
            # Now we construct the completely new expanded query.
            # We copy the query list to avoid modifying the original user inputs.
            expanded_query_list = current_query.copy()
                
            # We then append the PRF terms as a distinct final block at the end.
            if len(extracted_new_terms) > 0:
                expanded_query_list.append(extracted_new_terms)
                
            second_pass_queries.append(expanded_query_list)
            
        # Finally, we run the updated PRF-expanded queries through the ranking model again
        second_pass_results = self.base_bm25_model.rank(second_pass_queries)
        
        return second_pass_results
