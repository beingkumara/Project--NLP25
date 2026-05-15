from informationRetrievalBM25 import InformationRetrievalBM25
from util import count_query_words


class InformationRetrievalPRF:

    def __init__(self, k1=1.5, b=0.75):
        # Let us initialize the basic parameters for BM25 first
        self.k1 = k1
        self.b = b
        self.base_bm25_model = InformationRetrievalBM25(k1=self.k1, b=self.b)
        self.saved_docs_map = {}

    def buildIndex(self, docs, docIDs):
        # Basically building the index using the base model first
        self.base_bm25_model.buildIndex(docs, docIDs)

        # Here we will iterate to save docs in memory for fast access later
        for index_counter in range(len(docs)):
            document_id = docIDs[index_counter]
            self.saved_docs_map[document_id] = docs[index_counter]

    def rank(self, queries):
        # First we do a normal rank using bm25
        first_pass_results = self.base_bm25_model.rank(queries)
        second_pass_queries = []

        for q_index in range(len(queries)):
            current_query = queries[q_index]
            current_baseline_ranked_docs = first_pass_results[q_index]

            # Limiting pseudo relevant docs to top 3
            limit_docs = 3
            if len(current_baseline_ranked_docs) < 3:
                limit_docs = len(current_baseline_ranked_docs)

            top_pseudo_relevant_docs = []
            for i in range(limit_docs):
                top_pseudo_relevant_docs.append(current_baseline_ranked_docs[i])

            # Now we will collect term frequencies from these top docs
            term_frequencies = {}
            for doc_id in top_pseudo_relevant_docs:
                doc_content = self.saved_docs_map[doc_id]

                # Checking each word individually
                for sent_idx in range(len(doc_content)):
                    sentence = doc_content[sent_idx]
                    for word_idx in range(len(sentence)):
                        word = sentence[word_idx]
                        if word not in term_frequencies:
                            term_frequencies[word] = 0
                        term_frequencies[word] = term_frequencies[word] + 1

            # Removing words that are already present in the query
            existing_query_word_counts, _ = count_query_words(current_query)
            list_of_terms = []

            for word in term_frequencies.keys():
                if word not in existing_query_word_counts:
                    freq = term_frequencies[word]
                    list_of_terms.append((freq, word))

            # Sorting based on the highest frequency
            list_of_terms.sort(reverse=True)
            extracted_new_terms = []

            # Limiting expansion terms to 3 for efficiency
            limit_terms = 3
            if len(list_of_terms) < 3:
                limit_terms = len(list_of_terms)

            for k_term in range(limit_terms):
                extracted_new_terms.append(list_of_terms[k_term][1])

            # Finally creating the new expanded query
            expanded_query_list = current_query.copy()
            if len(extracted_new_terms) > 0:
                expanded_query_list.append(extracted_new_terms)
            second_pass_queries.append(expanded_query_list)

        # We will rank again using the expanded queries
        second_pass_results = self.base_bm25_model.rank(second_pass_queries)
        return second_pass_results
