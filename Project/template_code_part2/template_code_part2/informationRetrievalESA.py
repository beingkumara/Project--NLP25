# informationRetrievalESA.py
# ===========================
# ESA (Explicit Semantic Analysis) based retrieval for Part 5.
#
# The idea here is to represent documents and queries not as bags of words
# but as vectors over "concepts". We use WordNet synsets as our concepts.
# Each synset (like 'dog.n.01') acts as one dimension in a concept vector.
#
# For each word in a document, I look up its WordNet synsets and add the
# TF-IDF weight of that word to those concept dimensions.
# So two documents that use totally different words but describe the same
# concept (like 'velocity' vs 'speed') will hopefully end up with similar
# concept vectors because they activate overlapping synsets.
#
# I am using WordNet here because we do not have Wikipedia available.
# The original ESA paper used Wikipedia articles as concepts instead,
# which would probably work better for this technical domain.
#
# Main steps:
#   1. Build inverted index and compute TF-IDF
#   2. Map every vocabulary word to its WordNet synsets
#   3. Project each document into the concept space
#   4. For ranking, do the same projection for the query
#   5. Rank by cosine similarity in concept space

import math
from nltk.corpus import wordnet
from util import (build_inverted_index, compute_idf_values, compute_max_tf_per_doc,
                  count_query_words, sort_and_complete_ranking, compute_vector_norms,
                  cosine_similarity_sparse)

class InformationRetrievalESA:

    def __init__(self):
        """
        Here we initialize the ESA retrieval object.
        We will store all our computed data structures in these variables.
        """
        self.index = None
        self.docIDs = None

        # This will hold the mapping: word -> list of synset name strings
        self.word_to_concepts = None

        # This will hold document concept vectors: docID -> {concept_name: weight}
        self.doc_concept_vectors = None

        # This will hold the magnitude (norm) of each document concept vector
        self.doc_concept_norms = None

    # ------------------------------------------------------------------
    # Helper: get synset names for a given word
    # ------------------------------------------------------------------
    def get_concept_names(self, word):
        """
        For a given word, we look up all its WordNet synsets and return
        their canonical names as a list of strings.
        We limit to the first 5 synsets to avoid too much noise.

        Parameters
        ----------
        word : str
            The word token to look up in WordNet.

        Returns
        -------
        list
            A list of synset name strings (e.g., 'dog.n.01').
        """
        synset_list = wordnet.synsets(word)

        concept_names = []
        limit = 5
        count = 0
        for s_idx in range(len(synset_list)):
            if count >= limit:
                break
            concept_names.append(synset_list[s_idx].name())
            count = count + 1

        return concept_names

    # ------------------------------------------------------------------
    # buildIndex
    # ------------------------------------------------------------------
    def buildIndex(self, docs, docIDs):
        """
        Build the ESA index.

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

        # ----- Step 1: Build raw term frequency index using shared function -----
        self.index = build_inverted_index(docs, docIDs)

        # ----- Step 2: Compute IDF using shared function -----
        idf_values = compute_idf_values(self.index, total_documents)

        # ----- Step 3: Compute max TF per document using shared function -----
        max_tf_per_doc = compute_max_tf_per_doc(self.index)

        # ----- Step 4: Build word-to-concept mapping -----
        # For every unique word in our vocabulary, we find its WordNet synsets
        self.word_to_concepts = {}
        for word in self.index:
            concepts_for_word = self.get_concept_names(word)
            if len(concepts_for_word) > 0:
                self.word_to_concepts[word] = concepts_for_word
            # If a word has no synsets, we simply skip it (OOV for WordNet)

        # ----- Step 5: Project each document into concept space -----
        # For each document, we iterate through all words that appear in it.
        # For each word, we find its TF-IDF weight and distribute that weight
        # across all the synset concepts that the word maps to.
        self.doc_concept_vectors = {}

        for d_id in docIDs:
            self.doc_concept_vectors[d_id] = {}

        for word in self.index:
            # Skip words that have no WordNet concepts
            if word not in self.word_to_concepts:
                continue

            concept_list = self.word_to_concepts[word]
            idf_w = idf_values[word]

            for d_id in self.index[word]:
                raw_tf = self.index[word][d_id]
                max_tf = max_tf_per_doc[d_id]

                # Augmented TF calculation: 0.5 + 0.5 * (tf / max_tf)
                aug_tf = 0.5 + 0.5 * (float(raw_tf) / float(max_tf))
                tfidf_weight = aug_tf * idf_w

                # Now we distribute this weight to all concepts of this word
                for c_idx in range(len(concept_list)):
                    concept_name = concept_list[c_idx]

                    if concept_name not in self.doc_concept_vectors[d_id]:
                        self.doc_concept_vectors[d_id][concept_name] = 0.0

                    self.doc_concept_vectors[d_id][concept_name] = (
                        self.doc_concept_vectors[d_id][concept_name] + tfidf_weight
                    )

        # ----- Step 6: Pre-compute norms using shared function -----
        self.doc_concept_norms = compute_vector_norms(self.doc_concept_vectors, docIDs)

    # ------------------------------------------------------------------
    # rank
    # ------------------------------------------------------------------
    def rank(self, queries):
        """
        Rank documents for each query using ESA.

        Parameters
        ----------
        queries : list
            A 3D list of lists representing the queries.

        Returns
        -------
        list
            A list of lists containing ranked document IDs for each query.
        """
        # Base condition checking
        if self.index is None or len(self.docIDs) == 0:
            empty_result = []
            for q in queries:
                empty_result.append([])
            return empty_result

        final_ranked_lists = []

        for q_idx in range(len(queries)):
            query = queries[q_idx]

            # ----- Build raw TF for query words using shared function -----
            query_word_tf, max_tf_query = count_query_words(query)

            # ----- Project query into concept space -----
            query_concept_vector = {}

            for word in query_word_tf:
                # We need IDF from the index; skip unknown words
                if word not in self.index:
                    continue
                if word not in self.word_to_concepts:
                    continue

                raw_tf_q = query_word_tf[word]
                aug_tf_q = 0.5 + 0.5 * (float(raw_tf_q) / float(max_tf_query))

                # IDF from the document collection
                df = len(self.index[word])
                total_docs = len(self.docIDs)
                idf_w = math.log10(float(total_docs) / float(df))

                tfidf_q = aug_tf_q * idf_w

                concept_list = self.word_to_concepts[word]
                for c_idx in range(len(concept_list)):
                    concept_name = concept_list[c_idx]
                    if concept_name not in query_concept_vector:
                        query_concept_vector[concept_name] = 0.0
                    query_concept_vector[concept_name] = (
                        query_concept_vector[concept_name] + tfidf_q
                    )

            # ----- Compute query concept vector norm -----
            query_norm = 0.0
            for concept_name in query_concept_vector:
                w = query_concept_vector[concept_name]
                query_norm = query_norm + (w * w)
            query_norm = math.sqrt(query_norm)

            # ----- Compute cosine similarity with every document using shared function -----
            doc_scores = {}

            for d_id in self.docIDs:
                doc_vec = self.doc_concept_vectors[d_id]
                doc_norm = self.doc_concept_norms[d_id]

                cosine_sim = cosine_similarity_sparse(
                    query_concept_vector, doc_vec, query_norm, doc_norm
                )
                doc_scores[d_id] = cosine_sim

            # ----- Sort and complete ranking using shared function -----
            ordered_ids = sort_and_complete_ranking(doc_scores, self.docIDs)
            final_ranked_lists.append(ordered_ids)

        return final_ranked_lists
