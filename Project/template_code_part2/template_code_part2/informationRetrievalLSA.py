import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from util import (
    build_inverted_index,
    compute_idf_values,
    compute_max_tf_per_doc,
    count_query_words,
)


class InformationRetrievalLSA:

    def __init__(self, n_components=100):
        # Let us initialize the default values first
        # Initializing all our variables to none first for safety
        self.n_components = n_components
        self.docIDs = None
        self.vocab = None
        self.doc_matrix_lsa = None
        self.svd_model = None
        self.idf_values = None
        self.index = None
        self.max_tf_per_doc = None

    def buildIndex(self, docs, docIDs):
        # We need to handle this edge case condition
        # Basically building the inverted index and counting frequencies
        self.docIDs = docIDs
        N = len(docIDs)
        self.index = build_inverted_index(docs, docIDs)
        self.max_tf_per_doc = compute_max_tf_per_doc(self.index)
        all_unique_words = list(self.index.keys())
        all_unique_words.sort()
        self.vocab = {}
        for idx in range(len(all_unique_words)):
            self.vocab[all_unique_words[idx]] = idx
        V = len(all_unique_words)
        self.idf_values = compute_idf_values(self.index, N)
        tfidf_matrix = np.zeros((V, N), dtype=np.float32)
        doc_id_to_col_mapping = {}
        for col_idx in range(len(docIDs)):
            doc_id_to_col_mapping[docIDs[col_idx]] = col_idx
        list_of_vocab_items = list(self.vocab.keys())
        for v_idx in range(len(list_of_vocab_items)):
            word = list_of_vocab_items[v_idx]
            term_idx = self.vocab[word]
            idf_val = self.idf_values[word]
            list_of_doc_ids_for_word = list(self.index[word].keys())
            for d_idx in range(len(list_of_doc_ids_for_word)):
                d_id = list_of_doc_ids_for_word[d_idx]
                raw_tf_value = self.index[word][d_id]
                col_number = doc_id_to_col_mapping[d_id]
                max_tf_in_doc = self.max_tf_per_doc[d_id]
                augmented_tf = 0.5 + 0.5 * (float(raw_tf_value) / float(max_tf_in_doc))
                tfidf_matrix[term_idx, col_number] = augmented_tf * idf_val
        actual_k_value = min(self.n_components, V - 1, N - 1)
        self.n_components = actual_k_value
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        raw_lsa_matrix = self.svd_model.fit_transform(tfidf_matrix.T)
        self.doc_matrix_lsa = normalize(raw_lsa_matrix, norm="l2")

    def convert_query_to_vector(self, query):
        # Executing the main logic for convert_query_to_vector
        V = len(self.vocab)
        query_vector = np.zeros(V, dtype=np.float32)
        query_word_counts, max_tf_in_query = count_query_words(query)
        if len(query_word_counts) == 0:
            return None
        words_in_vocab_count = 0
        list_of_query_words = list(query_word_counts.keys())
        for qw_idx in range(len(list_of_query_words)):
            word = list_of_query_words[qw_idx]
            raw_tf_value = query_word_counts[word]
            if word in self.vocab:
                if word in self.idf_values:
                    augmented_tf = 0.5 + 0.5 * (
                        float(raw_tf_value) / float(max_tf_in_query)
                    )
                    final_weight = augmented_tf * self.idf_values[word]
                    query_vector[self.vocab[word]] = final_weight
                    words_in_vocab_count = words_in_vocab_count + 1
        if words_in_vocab_count == 0:
            return None
        single_row_matrix = query_vector.reshape(1, -1)
        query_lsa_vector = self.svd_model.transform(single_row_matrix)[0]
        vector_length = np.linalg.norm(query_lsa_vector)
        if vector_length > 0:
            query_lsa_vector = query_lsa_vector / vector_length
        return query_lsa_vector

    def rank(self, queries):
        # Now we are calculating the scores and ranking the documents
        if self.doc_matrix_lsa is None:
            ans = []
            for q_idx in range(len(queries)):
                ans.append([])
            return ans
        final_ranking = []
        for q_idx in range(len(queries)):
            query = queries[q_idx]
            query_lsa_vector = self.convert_query_to_vector(query)
            if query_lsa_vector is None:
                doc_list_copy = []
                for doc_index in range(len(self.docIDs)):
                    doc_list_copy.append(self.docIDs[doc_index])
                final_ranking.append(doc_list_copy)
                continue
            similarity_scores = self.doc_matrix_lsa.dot(query_lsa_vector)
            sorted_indices = np.argsort(-similarity_scores)
            ranked_doc_list = []
            for sorted_idx in range(len(sorted_indices)):
                col = sorted_indices[sorted_idx]
                ranked_doc_list.append(self.docIDs[col])
            final_ranking.append(ranked_doc_list)
        return final_ranking
