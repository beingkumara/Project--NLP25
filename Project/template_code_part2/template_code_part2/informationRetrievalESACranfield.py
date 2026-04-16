"""
informationRetrievalESACranfield.py
====================================
Explicit Semantic Analysis (ESA) using Cranfield Documents as Concepts.

Since we do not have an external Wikipedia dump, here we will use the
Cranfield documents themselves as our Explicit Concepts.
Every document in the corpus is treated as a unique concept.

Mathematical Theory:
1. An explicit concept space is built where each dimension is a Cranfield Document.
2. The concept vector for a specific word shows its TF-IDF distribution across all documents.
3. We project Queries and Documents into this "Document-Concept Space".
   Essentially, a query's concept vector becomes its baseline similarity
   profile against all documents.
4. We rank documents by taking the cosine similarity in this high-level concept space
   rather than the raw word space.
"""

import math
from util import (build_inverted_index, compute_idf_values, compute_max_tf_per_doc,
                  count_query_words, sort_and_complete_ranking, compute_vector_norms,
                  cosine_similarity_sparse)

class InformationRetrievalESACranfield:

    def __init__(self):
        """
        Initialization of the ESA Cranfield class objects.
        """
        self.index = None
        self.docIDs = None
        
        # Word-to-Concept mapping (word -> {concept_doc_id: weight})
        self.word_concept_weights = None
        
        # Concept vectors for documents (doc_id -> {concept_doc_id: weight})
        self.doc_concept_vectors = None
        self.doc_concept_norms = None

    def buildIndex(self, docs, docIDs):
        """
        Build the ESA Index using the documents themselves as the concept dimensions.

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
        self.docIDs = docIDs
        total_documents = len(docs)
        
        # ----- Step 1: Standard TF Indexing using shared function -----
        self.index = build_inverted_index(docs, docIDs)

        # ----- Step 2: Compute IDF and Maximum TF using shared functions -----
        idf_values = compute_idf_values(self.index, total_documents)
        max_tf_per_doc = compute_max_tf_per_doc(self.index)

        # ----- Step 3: Build Word-To-Concept Space -----
        # Since every document is a concept, the weight of a word in a concept
        # is simply its TF-IDF score in that document.
        self.word_concept_weights = {}
        
        for word in self.index:
            self.word_concept_weights[word] = {}
            idf_w = idf_values[word]
            
            for concept_doc_id in self.index[word]:
                raw_tf = self.index[word][concept_doc_id]
                max_tf = max_tf_per_doc[concept_doc_id]
                
                # Augmented TF
                aug_tf = 0.5 + 0.5 * (float(raw_tf) / float(max_tf))
                weight = aug_tf * idf_w
                
                self.word_concept_weights[word][concept_doc_id] = weight

        # ----- Step 4: Map Documents into the Concept Space -----
        # This fundamentally calculates the similarity profile of each document
        # with every other document in the corpus.
        self.doc_concept_vectors = {}
        for d_id in docIDs:
            self.doc_concept_vectors[d_id] = {}
            
        for word in self.index:
            concept_dict = self.word_concept_weights[word]
            idf_w = idf_values[word]
            
            for d_id in self.index[word]:
                raw_tf = self.index[word][d_id]
                max_tf = max_tf_per_doc[d_id]
                
                aug_tf = 0.5 + 0.5 * (float(raw_tf) / float(max_tf))
                term_weight_in_source = aug_tf * idf_w
                
                for concept_idx in concept_dict.keys():
                    concept_val = concept_dict[concept_idx]
                    
                    if concept_idx not in self.doc_concept_vectors[d_id]:
                        self.doc_concept_vectors[d_id][concept_idx] = 0.0
                        
                    self.doc_concept_vectors[d_id][concept_idx] = (
                        self.doc_concept_vectors[d_id][concept_idx] + (term_weight_in_source * concept_val)
                    )

        # ----- Step 5: Normalize Document Concept Vectors using shared function -----
        self.doc_concept_norms = compute_vector_norms(self.doc_concept_vectors, docIDs)


    def rank(self, queries):
        """
        Rank the queries by projecting into the Cranfield concept space.

        Parameters
        ----------
        queries : list
            A 3D list of lists representing the queries.

        Returns
        -------
        list
            A list of lists containing ranked document IDs for each query.
        """
        if self.index is None or len(self.docIDs) == 0:
            res = []
            for i in range(len(queries)):
                res.append([])
            return res

        final_ranked_lists = []
        total_docs = len(self.docIDs)

        for q_idx in range(len(queries)):
            query = queries[q_idx]

            # Using shared function for query word counting
            query_word_tf, max_tf_query = count_query_words(query)

            # Project query into document concept space
            query_concept_vector = {}

            for word in query_word_tf.keys():
                if word not in self.index:
                    continue
                if word not in self.word_concept_weights:
                    continue

                raw_tf_q = query_word_tf[word]
                aug_tf_q = 0.5 + 0.5 * (float(raw_tf_q) / float(max_tf_query))

                # Use original IDF
                df = len(self.index[word])
                idf_w = math.log10(float(total_docs) / float(df))

                term_weight_in_query = aug_tf_q * idf_w

                concept_dict = self.word_concept_weights[word]
                for concept_idx in concept_dict.keys():
                    concept_val = concept_dict[concept_idx]
                    
                    if concept_idx not in query_concept_vector:
                        query_concept_vector[concept_idx] = 0.0
                        
                    query_concept_vector[concept_idx] = (
                        query_concept_vector[concept_idx] + (term_weight_in_query * concept_val)
                    )

            # Compute query vector norm in concept space
            query_norm = 0.0
            for concept_key in query_concept_vector.keys():
                w = query_concept_vector[concept_key]
                query_norm = query_norm + (w * w)
            query_norm = math.sqrt(query_norm)

            # Cosine calculation using shared function
            doc_scores = {}
            for d_id in self.docIDs:
                doc_vec = self.doc_concept_vectors[d_id]
                d_norm = self.doc_concept_norms[d_id]
                
                cos_sim = cosine_similarity_sparse(
                    query_concept_vector, doc_vec, query_norm, d_norm
                )
                doc_scores[d_id] = cos_sim

            # Sorting using shared function
            ordered_doc_list = sort_and_complete_ranking(doc_scores, self.docIDs)
            final_ranked_lists.append(ordered_doc_list)

        return final_ranked_lists
