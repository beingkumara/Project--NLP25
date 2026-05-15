from util import *
import math


class InformationRetrieval:

    def __init__(self):
        # Let us initialize the basic variables first
        self.index = None
        self.docIDs = None

    def buildIndex(self, docs, docIDs):
        # Basically we use this to build the inverted index
        self.index = build_inverted_index(docs, docIDs)
        self.docIDs = docIDs

    def rank(self, queries):
        # Checking the base condition: if nothing is there, we must return empty list
        if self.index is None or len(self.docIDs) == 0:
            empty_res = []
            for q in queries:
                empty_res.append([])
            return empty_res

        all_ordered_docs = []
        total_docs = len(self.docIDs)

        # Now computing inverse document frequency values
        idf_dict = compute_idf_values(self.index, total_docs)
        # Also computing the maximum tf for the docs
        max_tf_dict = compute_max_tf_per_doc(self.index)

        # Let us calculate the document norms now
        doc_norms = dict()

        for w in self.index:
            curr_idf = idf_dict[w]
            for d_id in self.index[w]:
                tf_val = self.index[w][d_id]
                max_val = max_tf_dict[d_id]

                # Calculating augmented tf as studied
                aug_tf = 0.5 + 0.5 * (tf_val / max_val)
                weight = aug_tf * curr_idf

                # add square of weight
                if d_id in doc_norms:
                    doc_norms[d_id] = doc_norms[d_id] + (weight * weight)
                else:
                    doc_norms[d_id] = weight * weight

        # Taking the square root for the final norm
        for d_id in doc_norms:
            doc_norms[d_id] = math.sqrt(doc_norms[d_id])

        # Now we will iterate through the given queries
        for q in queries:
            q_tf, max_q_tf = count_query_words(q)

            q_vec = dict()
            q_norm = 0

            for w in q_tf:
                tf_val = q_tf[w]
                aug_tf = 0.5 + 0.5 * (tf_val / max_q_tf)

                if w in idf_dict:
                    w_weight = aug_tf * idf_dict[w]
                    q_vec[w] = w_weight
                    q_norm = q_norm + (w_weight * w_weight)

            # final query norm
            q_norm = math.sqrt(q_norm)

            doc_scores = dict()

            for w in q_vec:
                q_weight = q_vec[w]

                # Here we check which documents contain this particular word
                for d_id in self.index[w]:
                    d_tf = self.index[w][d_id]
                    d_max_tf = max_tf_dict[d_id]

                    d_aug_tf = 0.5 + 0.5 * (d_tf / d_max_tf)
                    d_weight = d_aug_tf * idf_dict[w]

                    # Finally multiplying the weights
                    score = q_weight * d_weight

                    if d_id in doc_scores:
                        doc_scores[d_id] = doc_scores[d_id] + score
                    else:
                        doc_scores[d_id] = score

            # Divide the dot product by norms to get cosine similarity
            for d_id in doc_scores:
                dot_prod = doc_scores[d_id]

                d_norm_val = doc_norms[d_id]

                # Standard check to avoid dividing by 0 error
                if q_norm > 0 and d_norm_val > 0:
                    cos_sim = dot_prod / (q_norm * d_norm_val)
                else:
                    cos_sim = 0

                doc_scores[d_id] = cos_sim

            # Using the sorting function to order our docs
            curr_ranks = sort_and_complete_ranking(doc_scores, self.docIDs)
            all_ordered_docs.append(curr_ranks)

        return all_ordered_docs
