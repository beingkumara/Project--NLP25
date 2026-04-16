"""
informationRetrievalLSA.py
===========================
Latent Semantic Analysis (LSA) based IR system for Part 5.

LSA addresses two core VSM failures:
  1. Vocabulary mismatch: words like "thermal" and "heated" that co-occur
     in the same documents end up close in the reduced latent space, so a
     query containing "heated" can still match a doc that uses "thermal".
  2. Polysemy: "star" in astronomy contexts and "star" in entertainment
     contexts will project to slightly different regions of the latent
     space because their surrounding vocabulary is different.

Method:
  1. Build a TF-IDF term-document matrix  (shape: V x N, V=vocab, N=docs)
  2. Apply Truncated SVD to reduce to k latent dimensions
     - SVD: M = U * Sigma * V^T
     - Reduced doc matrix = Sigma_k * V_k^T  (shape: k x N)
     - To project a query into the same space:
         q_lsa = Sigma_k^{-1} * U_k^T * q_tfidf
  3. Rank documents by cosine similarity in the k-dimensional latent space.

The number of components k is configurable for ablation testing.
Typical values to test: 50, 100, 200, 300.
"""

import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from util import build_inverted_index, compute_idf_values, compute_max_tf_per_doc, count_query_words

class InformationRetrievalLSA:

    def __init__(self, n_components=100):
        """
        Parameters
        ----------
        n_components : int
            Number of latent dimensions to keep after SVD reduction.
            Higher k = more expressive but slower and potentially noisier.
            This is our ablation hyperparameter.
            
        Outputs / Returns
        -----------------
        None.
        We initialize the foundational objects and matrix pointers 
        (like svd_model and doc_matrix_lsa) natively. 
        """
        self.n_components = n_components

        # These will store our model details
        self.docIDs = None
        self.vocab = None
        self.doc_matrix_lsa = None
        self.svd_model = None
        self.idf_values = None
        self.index = None
        self.max_tf_per_doc = None

    def buildIndex(self, docs, docIDs):
        """
        Build the LSA index:
          - Construct TF-IDF term-document matrix
          - Apply TruncatedSVD to reduce dimensions
          - Store reduced document representations

        Parameters
        ----------
        docs   : list of list of list of str
        docIDs : list of int

        Outputs / Returns
        -----------------
        None.
        We calculate our full TF-IDF matrix, and then run SVD on it 
        to compress all that high-dimensional word data down into a smaller space.
        """
        self.docIDs = docIDs
        N = len(docIDs)

        # Using shared utility functions for common work
        self.index = build_inverted_index(docs, docIDs)
        self.max_tf_per_doc = compute_max_tf_per_doc(self.index)

        # Here we get all unique words and sort them so that the matrix dimension is correct
        all_unique_words = list(self.index.keys())
        all_unique_words.sort()
        
        self.vocab = {}
        for idx in range(len(all_unique_words)):
            self.vocab[all_unique_words[idx]] = idx
            
        V = len(all_unique_words)

        # Using shared IDF computation
        self.idf_values = compute_idf_values(self.index, N)

        # Now checking and creating the TF-IDF matrix
        tfidf_matrix = np.zeros((V, N), dtype=np.float32)

        doc_id_to_col_mapping = {}
        for col_idx in range(len(docIDs)):
            doc_id_to_col_mapping[docIDs[col_idx]] = col_idx

        # Iterate through vocabulary
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
                
                # Using augmented TF for preventing bias towards long documents
                augmented_tf = 0.5 + 0.5 * (float(raw_tf_value) / float(max_tf_in_doc))
                
                tfidf_matrix[term_idx, col_number] = augmented_tf * idf_val

        # Safety condition check as we don't want SVD to crash
        actual_k_value = min(self.n_components, V - 1, N - 1)
        self.n_components = actual_k_value

        # Hence we use TruncatedSVD to approximate our large matrix
        # Transposing because sklearn expects rows=samples
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        raw_lsa_matrix = self.svd_model.fit_transform(tfidf_matrix.T)

        # Normalization is important so we can just use dot product later
        self.doc_matrix_lsa = normalize(raw_lsa_matrix, norm='l2')

    def convert_query_to_vector(self, query):
        """
        Convert a preprocessed query into a TF-IDF vector in the original
        vocabulary space, then project it into the LSA latent space.

        Query projection formula (folding-in):
          q_lsa = U_k^T * q_tfidf  (shape: k)
        This is what sklearn's transform() does internally.

        Returns
        -------
        np.ndarray of shape (k,)  or  None if query is all OOV
        """
        V = len(self.vocab)
        query_vector = np.zeros(V, dtype=np.float32)

        # Using shared query word counting function
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
                    augmented_tf = 0.5 + 0.5 * (float(raw_tf_value) / float(max_tf_in_query))
                    final_weight = augmented_tf * self.idf_values[word]
                    query_vector[self.vocab[word]] = final_weight
                    words_in_vocab_count = words_in_vocab_count + 1

        if words_in_vocab_count == 0:
            return None # Out of vocabulary entirely

        # We will fold-in the query right now
        single_row_matrix = query_vector.reshape(1, -1)
        query_lsa_vector = self.svd_model.transform(single_row_matrix)[0]

        # Normalize the length using norm
        vector_length = np.linalg.norm(query_lsa_vector)
        if vector_length > 0:
            query_lsa_vector = query_lsa_vector / vector_length

        return query_lsa_vector

    def rank(self, queries):
        """
        Rank documents for each query using LSA.

        Parameters
        ----------
        queries : list
            A 3D list of lists representing the queries.

        Returns
        -------
        list
            A list of lists containing ranked document IDs for each query.
        """
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
                # If OOV, return docs in same order
                doc_list_copy = []
                for doc_index in range(len(self.docIDs)):
                    doc_list_copy.append(self.docIDs[doc_index])
                final_ranking.append(doc_list_copy)
                continue

            # Now checking similarity
            similarity_scores = self.doc_matrix_lsa.dot(query_lsa_vector)

            # Need to sort in descending order
            sorted_indices = np.argsort(-similarity_scores)
            
            ranked_doc_list = []
            for sorted_idx in range(len(sorted_indices)):
                col = sorted_indices[sorted_idx]
                ranked_doc_list.append(self.docIDs[col])

            final_ranking.append(ranked_doc_list)

        return final_ranking