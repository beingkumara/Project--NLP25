import math
from nltk.corpus import wordnet
from util import (
    build_inverted_index,
    compute_idf_values,
    compute_max_tf_per_doc,
    count_query_words,
    sort_and_complete_ranking,
    compute_vector_norms,
    cosine_similarity_sparse,
)


class InformationRetrievalESA:

    def __init__(self):
        # Let us set up our main variables first
        self.index = None
        self.docIDs = None
        self.word_to_concepts = None
        self.doc_concept_vectors = None
        self.doc_concept_norms = None

    def get_concept_names(self, word):
        # Basically finding synonyms using wordnet
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

    def buildIndex(self, docs, docIDs):
        # Building the inverted index first before anything else
        self.docIDs = docIDs
        total_documents = len(docs)
        self.index = build_inverted_index(docs, docIDs)
        idf_values = compute_idf_values(self.index, total_documents)
        max_tf_per_doc = compute_max_tf_per_doc(self.index)
        self.word_to_concepts = {}
        for word in self.index:
            concepts_for_word = self.get_concept_names(word)
            if len(concepts_for_word) > 0:
                self.word_to_concepts[word] = concepts_for_word
        self.doc_concept_vectors = {}
        for d_id in docIDs:
            self.doc_concept_vectors[d_id] = {}
        for word in self.index:
            if word not in self.word_to_concepts:
                continue
            concept_list = self.word_to_concepts[word]
            idf_w = idf_values[word]
            for d_id in self.index[word]:
                raw_tf = self.index[word][d_id]
                max_tf = max_tf_per_doc[d_id]
                aug_tf = 0.5 + 0.5 * (float(raw_tf) / float(max_tf))
                tfidf_weight = aug_tf * idf_w
                for c_idx in range(len(concept_list)):
                    concept_name = concept_list[c_idx]
                    if concept_name not in self.doc_concept_vectors[d_id]:
                        self.doc_concept_vectors[d_id][concept_name] = 0.0
                    self.doc_concept_vectors[d_id][concept_name] = (
                        self.doc_concept_vectors[d_id][concept_name] + tfidf_weight
                    )
        self.doc_concept_norms = compute_vector_norms(self.doc_concept_vectors, docIDs)

    def rank(self, queries):
        # Ranking documents based on the standard cosine similarity
        if self.index is None or len(self.docIDs) == 0:
            empty_result = []
            for q in queries:
                empty_result.append([])
            return empty_result
        final_ranked_lists = []
        for q_idx in range(len(queries)):
            query = queries[q_idx]
            query_word_tf, max_tf_query = count_query_words(query)
            query_concept_vector = {}
            for word in query_word_tf:
                if word not in self.index:
                    continue
                if word not in self.word_to_concepts:
                    continue
                raw_tf_q = query_word_tf[word]
                aug_tf_q = 0.5 + 0.5 * (float(raw_tf_q) / float(max_tf_query))
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
            query_norm = 0.0
            for concept_name in query_concept_vector:
                w = query_concept_vector[concept_name]
                query_norm = query_norm + w * w
            query_norm = math.sqrt(query_norm)
            doc_scores = {}
            for d_id in self.docIDs:
                doc_vec = self.doc_concept_vectors[d_id]
                doc_norm = self.doc_concept_norms[d_id]
                cosine_sim = cosine_similarity_sparse(
                    query_concept_vector, doc_vec, query_norm, doc_norm
                )
                doc_scores[d_id] = cosine_sim
            ordered_ids = sort_and_complete_ranking(doc_scores, self.docIDs)
            final_ranked_lists.append(ordered_ids)
        return final_ranked_lists
