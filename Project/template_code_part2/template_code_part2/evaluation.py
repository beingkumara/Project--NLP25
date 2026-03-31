from util import *

# Add your import statements here
import math

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		# initialize precision as 0 first
		precision = 0

		# If k is 0, precision is not defined so we return 0
		if k == 0:
			return 0

		# We only look at the top-k documents retrieved
		# As per theory: Precision@k = (relevant docs in top-k) / k
		# Here we will truncate the result list to k if it is longer
		top_k_docs = []
		for i in range(len(query_doc_IDs_ordered)):
			if i < k:
				top_k_docs.append(query_doc_IDs_ordered[i])

		# Now we count how many of the top-k retrieved docs are actually relevant
		count_relevant = 0
		for doc_id in top_k_docs:
			if doc_id in true_doc_IDs:
				count_relevant = count_relevant + 1

		# Precision@k = number of relevant docs retrieved in top-k / k
		precision = count_relevant / k

		# print("precision at k=", k, "is:", precision)   # kept for debugging

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""


		total_precision = 0

		for query_id in query_ids:
			# Get the precision for this particular query
			prec = self.queryPrecision(doc_IDs_ordered[query_id], query_id, qrels[query_id], k)
			total_precision = total_precision + prec

		# Now divide by number of queries to get the mean
		mean_precision = total_precision / len(query_ids)

		return mean_precision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""

		# Let us initialize recall as 0
		recall = 0

		# Total number of relevant documents in the ground truth
		total_relevant = len(true_doc_IDs)

		# If there are no relevant documents at all, recall is not meaningful
		# So we return 0 to avoid division by zero
		if total_relevant == 0:
			return 0

		# Truncate to top-k retrieved documents
		top_k_docs = []
		for i in range(len(query_doc_IDs_ordered)):
			if i < k:
				top_k_docs.append(query_doc_IDs_ordered[i])

		# Count how many relevant docs appear in our top-k results
		count_relevant = 0
		for doc_id in top_k_docs:
			if doc_id in true_doc_IDs:
				count_relevant = count_relevant + 1

		# Recall@k = (relevant docs retrieved in top-k) / (total relevant docs)
		recall = count_relevant / total_relevant

		# print("recall at k=", k, "is:", recall)  

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		total_recall = 0

		for query_id in query_ids:
			rec = self.queryRecall(doc_IDs_ordered[query_id], query_id, qrels[query_id], k)
			total_recall = total_recall + rec

		mean_recall = total_recall / len(query_ids)

		return mean_recall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query.

		NOTE: The assignment requires F0.5-score (beta = 0.5), NOT F1-score.
		The formula for F-beta score is:
		  F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
		For beta = 0.5:
		  F_0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
		       = 1.25 * P * R / (0.25 * P + R)
		This gives more weight to Precision than Recall.
		"""

		# First compute precision@k and recall@k for this query
		precision_at_k = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall_at_k = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		# If either is zero, F-score is also 0. 
		if precision_at_k == 0 or recall_at_k == 0:
			fscore = 0
		else:
			# Using the F0.5 formula as required by the assignment
			# beta = 0.5, beta_squared = 0.25
			beta_squared = 0.25
			numerator = (1 + beta_squared) * precision_at_k * recall_at_k
			denominator = (beta_squared * precision_at_k) + recall_at_k
			fscore = numerator / denominator

		# print("F0.5 score at k=", k, ":", fscore) 

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		total_fscore = 0

		for query_id in query_ids:
			fs = self.queryFscore(doc_IDs_ordered[query_id], query_id, qrels[query_id], k)
			total_fscore = total_fscore + fs

		mean_fscore = total_fscore / len(query_ids)

		return mean_fscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query.

		"""

		# Initialize all values
		DCG = 0
		IDCG = 0
		nDCG = 0

		# We first truncate the result list to top-k documents only
		top_k_docs = []
		for i in range(len(query_doc_IDs_ordered)):
			if i < k:
				top_k_docs.append(query_doc_IDs_ordered[i])

		# Now compute DCG by iterating through the top-k retrieved docs
		# The position is i+1 (1-indexed) and discount factor is log2(i+1+1) = log2(i+2)
		# Wait - standard formula: DCG = sum rel_i / log2(i+1) where i starts from 1
		# So for position i (1-indexed): discount = log2(i + 1)
		for i in range(len(top_k_docs)):
			# i is 0-indexed, so position = i+1 (1-indexed)
			position = i + 1
			if top_k_docs[i] in true_doc_IDs:
				# Relevance is 1 (binary), so gain = 1 / log2(position + 1)
				DCG = DCG + (1.0 / math.log2(position + 1))

		# Now compute IDCG - the ideal scenario
		# In the ideal ranking, we put all truly relevant documents first
		# We can place at most min(k, len(true_doc_IDs)) relevant docs in top-k
		num_ideal_relevant = min(k, len(true_doc_IDs))

		# IDCG assumes first num_ideal_relevant positions are all relevant
		for i in range(num_ideal_relevant):
			position = i + 1
			IDCG = IDCG + (1.0 / math.log2(position + 1))

		# print("DCG:", DCG, "IDCG:", IDCG)   # for debugging purposes

		# If IDCG is 0 (no relevant docs exist), nDCG is 0
		if IDCG == 0:
			nDCG = 0
		else:
			nDCG = DCG / IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		# Handle edge case where no queries are given
		if len(query_ids) == 0:
			return 0

		total_ndcg = 0

		for query_id in query_ids:
			nd = self.queryNDCG(doc_IDs_ordered[query_id], query_id, qrels[query_id], k)
			total_ndcg = total_ndcg + nd

		mean_ndcg = total_ndcg / len(query_ids)

		return mean_ndcg


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""

		# AP@k = (1 / |R|) * sum of Precision@i for each relevant doc at position i within top-k
		# where |R| is total number of relevant docs

		avg_precision = 0

		# If there are no relevant documents, AP is 0
		total_relevant = len(true_doc_IDs)
		if total_relevant == 0:
			return 0

		# Truncate to top-k
		top_k_docs = []
		for i in range(len(query_doc_IDs_ordered)):
			if i < k:
				top_k_docs.append(query_doc_IDs_ordered[i])

		# For each position i (1-indexed) in top-k, if the doc is relevant,
		# we add Precision@i to our running sum
		precision_sum = 0
		for i in range(len(top_k_docs)):
			position = i + 1   # 1-indexed
			if top_k_docs[i] in true_doc_IDs:
				# Compute precision at this position i
				prec_at_i = self.queryPrecision(top_k_docs, query_id, true_doc_IDs, position)
				precision_sum = precision_sum + prec_at_i

		# Divide by total number of relevant docs
		avg_precision = precision_sum / total_relevant

		# print("AP at k=", k, ":", avg_precision)   # debug

		return avg_precision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""

		# Handle edge case: if no queries, return 0
		if len(query_ids) == 0:
			return 0

		total_ap = 0

		# For each query, compute Average Precision and add it to total
		for query_id in query_ids:
			ap = self.queryAveragePrecision(doc_IDs_ordered[query_id], query_id, q_rels[query_id], k)
			total_ap = total_ap + ap

		# MAP = mean of all AP values
		mean_average_precision = total_ap / len(query_ids)

		return mean_average_precision



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		# Reciprocal Rank = 1 / rank of first relevant document in the result list
		# If no relevant doc is found in top-k, RR = 0

		reciprocal_rank = 0

		# Truncate the result list to top-k
		top_k_docs = []
		for i in range(len(query_doc_IDs_ordered)):
			if i < k:
				top_k_docs.append(query_doc_IDs_ordered[i])

		# Now iterate through top-k results and find the first relevant document
		for i in range(len(top_k_docs)):
			position = i + 1   # 1-indexed rank
			if top_k_docs[i] in true_doc_IDs:
				# Found the first relevant document at this rank
				reciprocal_rank = 1.0 / position
				break   # we only need the FIRST relevant doc, so we stop here

		# print("RR:", reciprocal_rank)   # debug check

		return reciprocal_rank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		if len(query_ids) == 0:
			return 0

		total_rr = 0
		#For each query, compute RR and add it to total
		for query_id in query_ids:
			rr = self.queryReciprocalRank(doc_IDs_ordered[query_id], query_id, qrels[query_id], k)
			total_rr = total_rr + rr

		mean_reciprocal_rank = total_rr / len(query_ids)

		return mean_reciprocal_rank
