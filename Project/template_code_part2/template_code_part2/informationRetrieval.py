from util import *
import math

class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable.
		We use our shared utility function to build the inverted index.

		Parameters
		----------
		docs : list
			A 3D list where docs[i] is a document, docs[i][j] is a sentence, 
			and docs[i][j][k] is a token.
		docIDs : list
			A list of integers representing the IDs of the documents.

		Returns
		-------
		None
		"""
		# Using the shared function from util.py to avoid writing same loop again
		self.index = build_inverted_index(docs, docIDs)
		self.docIDs = docIDs

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query.

		Parameters
		----------
		queries : list
			A 3D list where queries[i] is a query, queries[i][j] is a sentence, 
			and queries[i][j][k] is a token.

		Returns
		-------
		list
			A list of lists containing ranked document IDs for each query.
		"""
		# Base case: empty index or missing documents
		if self.index is None or len(self.docIDs) == 0:
			empty_result = []
			for q in queries:
				empty_result.append([])
			return empty_result

		doc_IDs_ordered = []

		total_documents = len(self.docIDs)

		# Using our shared IDF computation function
		idf_values = compute_idf_values(self.index, total_documents)

		# Using our shared max TF computation function
		max_tf_per_doc = compute_max_tf_per_doc(self.index)

		# We must compute norm (length) of every document vector to use in cosine similarity denominator
		doc_vector_norms = dict()
		
		for word in self.index:
			idf_w = idf_values[word]
			for docID in self.index[word]:
				raw_tf_d = self.index[word][docID]
				max_tf_d = max_tf_per_doc[docID]
				
				# Augmented TF calculation for document
				augmented_tf_d = 0.5 + 0.5 * (raw_tf_d / max_tf_d)
				
				# TF-IDF weight = augmented TF * IDF
				weight_d = augmented_tf_d * idf_w
				
				# Squaring the weight and adding to total
				if docID in doc_vector_norms:
					doc_vector_norms[docID] = doc_vector_norms[docID] + (weight_d * weight_d)
				else:
					doc_vector_norms[docID] = (weight_d * weight_d)
		
		# Finally doing square root to get the actual magnitude for each document
		for docID in doc_vector_norms:
			doc_vector_norms[docID] = math.sqrt(doc_vector_norms[docID])

		# Processing each query 
		for query in queries:
			
			# Using our shared query word counting function
			query_tf, max_tf_in_query = count_query_words(query)
			
			query_vector = dict()
			query_vector_norm = 0
			
			for word in query_tf:
				raw_tf_q = query_tf[word]
				augmented_tf_q = 0.5 + 0.5 * (raw_tf_q / max_tf_in_query)
				
				# Using the exact same IDF values that we pre-computed from documents
				if word in idf_values:
					weight_of_query_word = augmented_tf_q * idf_values[word]
					query_vector[word] = weight_of_query_word
					query_vector_norm = query_vector_norm + (weight_of_query_word * weight_of_query_word)
					
			# Magnitude of query
			query_vector_norm = math.sqrt(query_vector_norm)

			# Now calculate the numerator of cosine similarity: dot product of query and document
			document_scores = dict()
			
			for word in query_vector:
				weight_of_query_word = query_vector[word]
				
				# Finding documents that have this current word
				for docID in self.index[word]:
					tf_value_in_document = self.index[word][docID]
					max_tf_this_doc = max_tf_per_doc[docID]
					
					augmented_tf_doc = 0.5 + 0.5 * (tf_value_in_document / max_tf_this_doc)
					weight_of_document_word = augmented_tf_doc * idf_values[word]
					
					# Multiplying the matching weights
					score_contribution = weight_of_query_word * weight_of_document_word
					
					if docID in document_scores:
						document_scores[docID] = document_scores[docID] + score_contribution
					else:
						document_scores[docID] = score_contribution

			# Final step is to divide by norms (Cosine Similarity formula)
			for docID in document_scores:
				dot_product = document_scores[docID]
				
				# Q and D magnitudes
				query_norm_value = query_vector_norm
				doc_norm_value = doc_vector_norms[docID]
				
				# Preventing zero division error
				if query_norm_value > 0 and doc_norm_value > 0:
					cosine_similarity = dot_product / (query_norm_value * doc_norm_value)
				else:
					cosine_similarity = 0
					
				document_scores[docID] = cosine_similarity

			# Using our shared sort and complete ranking function
			current_query_ranked_ids = sort_and_complete_ranking(document_scores, self.docIDs)
			
			# Append the final ranked document IDs for this query
			doc_IDs_ordered.append(current_query_ranked_ids)
	
		return doc_IDs_ordered
