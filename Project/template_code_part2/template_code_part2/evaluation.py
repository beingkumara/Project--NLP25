from util import *
import math


class Evaluation:

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        # Basically returning 0 if k is 0 as per logic
        if k == 0:
            return 0

        # Here I will just get the top k predicted ones
        top_predictions = []
        for i in range(len(query_doc_IDs_ordered)):
            if i < k:
                top_predictions.append(query_doc_IDs_ordered[i])

        # print("query:", query_id)

        # Let us check how many matched with actuals
        match_count = 0
        for d in top_predictions:
            if d in true_doc_IDs:
                match_count = match_count + 1

        # Precision is matches divided by k
        ans = match_count / k
        return ans

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        sum_prec = 0

        for q in query_ids:
            # Finding precision for each query and adding it up
            p = self.queryPrecision(doc_IDs_ordered[q], q, qrels[q], k)
            sum_prec = sum_prec + p

        # dividing by total queries
        mean_p = sum_prec / len(query_ids)
        return mean_p

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        total_actual = len(true_doc_IDs)

        # If nothing is relevant, recall is basically 0
        if total_actual == 0:
            return 0

        # Getting the top k elements
        top_k_list = []
        for i in range(len(query_doc_IDs_ordered)):
            if i < k:
                top_k_list.append(query_doc_IDs_ordered[i])

        match_count = 0
        for doc in top_k_list:
            if doc in true_doc_IDs:
                # found a match
                match_count = match_count + 1

        # Recall is matches divided by total actual relevant
        ans = match_count / total_actual
        return ans

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        sum_rec = 0

        for q in query_ids:
            # add all recalls
            r = self.queryRecall(doc_IDs_ordered[q], q, qrels[q], k)
            sum_rec = sum_rec + r

        # average it out
        mean_r = sum_rec / len(query_ids)
        return mean_r

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        # wait, making sure we don't divide by zero
        # Calculating precision and recall first to use in formula
        p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        # If either is 0 then fscore is naturally 0
        if p == 0 or r == 0:
            return 0
        else:
            # As studied, we need F0.5, so beta is 0.5 and beta square is 0.25
            b_sq = 0.25
            top_part = (1 + b_sq) * p * r
            bottom_part = (b_sq * p) + r
            f_ans = top_part / bottom_part
            # print("fscore is", f_ans)
            return f_ans

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        sum_f = 0

        for q in query_ids:
            # calculate fscore
            f = self.queryFscore(doc_IDs_ordered[q], q, qrels[q], k)
            sum_f = sum_f + f

        mean_f = sum_f / len(query_ids)
        return mean_f

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        # Finding the max value first
        dcg_val = 0
        idcg_val = 0

        # top k docs
        top_k_docs = []
        for i in range(len(query_doc_IDs_ordered)):
            if i < k:
                top_k_docs.append(query_doc_IDs_ordered[i])

        # Calculating DCG score here
        for i in range(len(top_k_docs)):
            pos = i + 1
            if top_k_docs[i] in true_doc_IDs:
                # Relevance is just 1 in this simple case
                dcg_val = dcg_val + (1.0 / math.log2(pos + 1))

        # Calculating IDCG for the ideal case
        ideal_count = min(k, len(true_doc_IDs))
        for i in range(ideal_count):
            pos = i + 1
            idcg_val = idcg_val + (1.0 / math.log2(pos + 1))

        if idcg_val == 0:
            return 0
        else:
            ans = dcg_val / idcg_val
            return ans

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        if len(query_ids) == 0:
            return 0

        sum_n = 0
        for q in query_ids:
            # Calculating ndcg for each one
            n = self.queryNDCG(doc_IDs_ordered[q], q, qrels[q], k)
            sum_n = sum_n + n

        mean_n = sum_n / len(query_ids)
        return mean_n

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        total_rel = len(true_doc_IDs)
        if total_rel == 0:
            return 0

        top_k_docs = []
        for i in range(len(query_doc_IDs_ordered)):
            if i < k:
                top_k_docs.append(query_doc_IDs_ordered[i])

        sum_precisions = 0
        for i in range(len(top_k_docs)):
            pos = i + 1
            if top_k_docs[i] in true_doc_IDs:
                # Getting precision at this exact position
                p_at_pos = self.queryPrecision(top_k_docs, query_id, true_doc_IDs, pos)
                sum_precisions = sum_precisions + p_at_pos

        ans = sum_precisions / total_rel
        return ans

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        if len(query_ids) == 0:
            return 0

        sum_ap = 0
        for q in query_ids:
            # Calculating Average Precision (AP)
            ap = self.queryAveragePrecision(doc_IDs_ordered[q], q, q_rels[q], k)
            sum_ap = sum_ap + ap

        mean_ap = sum_ap / len(query_ids)
        return mean_ap

    def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        # Adding to the final list for RR
        rr_val = 0

        top_k_docs = []
        for i in range(len(query_doc_IDs_ordered)):
            if i < k:
                top_k_docs.append(query_doc_IDs_ordered[i])

        # Checking for the first relevant doc
        for i in range(len(top_k_docs)):
            pos = i + 1
            if top_k_docs[i] in true_doc_IDs:
                rr_val = 1.0 / pos
                break  # stop at first one

        return rr_val

    def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
        if len(query_ids) == 0:
            return 0

        sum_rr = 0
        for q in query_ids:
            # calculating RR
            rr = self.queryReciprocalRank(doc_IDs_ordered[q], q, qrels[q], k)
            sum_rr = sum_rr + rr

        mean_rr = sum_rr / len(query_ids)
        return mean_rr
