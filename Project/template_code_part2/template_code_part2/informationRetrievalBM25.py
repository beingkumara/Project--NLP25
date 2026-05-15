import math
from util import build_inverted_index, count_query_words, sort_and_complete_ranking


class InformationRetrievalBM25:

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.index = None
        self.docIDs = None
        self.doc_lengths = None
        self.avg_doc_length = 0

    def buildIndex(self, docs, docIDs):
        self.docIDs = docIDs

        # Basically building the inverted index here
        self.index = build_inverted_index(docs, docIDs)

        self.doc_lengths = {}
        total_w = 0
        total_docs = len(docs)

        for d_idx in range(total_docs):
            d_id = docIDs[d_idx]
            w_count = 0

            # Here we will iterate to count words in the document
            for s_idx in range(len(docs[d_idx])):
                sentence = docs[d_idx][s_idx]
                for w_idx in range(len(sentence)):
                    w_count = w_count + 1

            self.doc_lengths[d_id] = w_count
            total_w = total_w + w_count

        if total_docs > 0:
            self.avg_doc_length = float(total_w) / float(total_docs)
        else:
            self.avg_doc_length = 1.0

    def calculate_robertson_idf(self, word):
        # Let us calculate this using the standard Robertson IDF formula
        N = len(self.docIDs)
        df = len(self.index[word])

        num = (N - df) + 0.5
        den = df + 0.5

        ans = math.log((num / den) + 1.0)
        return ans

    def calculate_bm25_score_for_term(self, word, doc_id, idf_val):
        tf_val = self.index[word][doc_id]
        doc_len = self.doc_lengths[doc_id]

        l_ratio = float(doc_len) / float(self.avg_doc_length)

        top = tf_val * (self.k1 + 1.0)
        bottom = tf_val + self.k1 * (1.0 - self.b + self.b * l_ratio)

        ans = idf_val * (top / bottom)
        return ans

    def rank(self, queries):
        # First we need to handle the edge case if index is None
        if self.index is None or len(self.docIDs) == 0:
            res = []
            for q in range(len(queries)):
                res.append([])
            return res

        final_ranked = []

        for q_idx in range(len(queries)):
            query = queries[q_idx]

            q_words, q_max_tf = count_query_words(query)

            doc_scores = {}

            for w in q_words.keys():
                if w in self.index:
                    idf_val = self.calculate_robertson_idf(w)

                    for d_id in self.index[w].keys():
                        score = self.calculate_bm25_score_for_term(w, d_id, idf_val)

                        if d_id not in doc_scores:
                            doc_scores[d_id] = 0.0

                        doc_scores[d_id] = doc_scores[d_id] + score

            ordered = sort_and_complete_ranking(doc_scores, self.docIDs)
            final_ranked.append(ordered)

        return final_ranked
