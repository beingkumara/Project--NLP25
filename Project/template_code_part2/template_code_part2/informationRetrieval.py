from util import *
import math

class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable
		"""
		number_of_documents = len(docs)
		index = dict()
		
		# Iterate through all the documents first
		for document_index in range(number_of_documents):
			docID = docIDs[document_index]
			
			# Going inside the document sentence by sentence
			for sentence in docs[document_index]:
				
				# Getting each word from the sentence list
				for word in sentence:
					
					# Creating a dictionary for the word if it is coming for the first time
					if word not in index:
						index[word] = dict()
					
					# Setting initial frequency of this word in this document to 0
					if docID not in index[word]:
						index[word][docID] = 0
					
					# Increment the count of the word for this particular docID
					index[word][docID] = index[word][docID] + 1
					# print("word:", word, "found in doc:", docID)

		# Saving the built index and document IDs for ranking phase
		self.index = index
		self.docIDs = docIDs

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query
		"""
		# Base case: empty index or missing documents
		if self.index is None or len(self.docIDs) == 0:
			empty_result = []
			for q in queries:
				empty_result.append([])
			return empty_result

		doc_IDs_ordered = []

		total_documents = len(self.docIDs)
		idf_values = dict()
		
		# Now calculating IDF (Inverse Document Frequency) for all words in our vocabulary
		# Formula from class is IDF = log10(N / df)
		for word in self.index:
			# df is document frequency, which is just the number of keys in the inner dictionary
			document_frequency = len(self.index[word])
			
			# Storing idf values so we do not have to compute them again and again
			idf_values[word] = math.log10(total_documents / document_frequency)

		# Computing the maximum Term Frequency (TF) for each document
		# This is required for finding the augmented TF later: TF_aug = 0.5 + 0.5*(tf/max_tf)
		max_tf_per_doc = dict()
		for word in self.index:
			for docID in self.index[word]:
				raw_tf = self.index[word][docID]
				if docID not in max_tf_per_doc:
					max_tf_per_doc[docID] = raw_tf
				else:
					if raw_tf > max_tf_per_doc[docID]:
						max_tf_per_doc[docID] = raw_tf

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
			
			# Finding raw word counts for the query
			query_tf = dict()
			for sentence in query:
				for word in sentence:
					if word in query_tf:
						query_tf[word] = query_tf[word] + 1
					else:
						query_tf[word] = 1
			
			query_vector = dict()
			query_vector_norm = 0
			
			# To apply augmented TF on the query, we find its max TF first
			max_tf_in_query = 0
			for word in query_tf:
				if query_tf[word] > max_tf_in_query:
					max_tf_in_query = query_tf[word]
			if max_tf_in_query == 0:
				max_tf_in_query = 1
			
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

			# Preparing the list to sort documents by their accumulated similarity scores
			ranked_documents_list = []
			for docID in document_scores:
				final_score = document_scores[docID]
				# putting score first to sort directly 
				ranked_documents_list.append([final_score, docID])
			
			# descending order sorting based on scores
			ranked_documents_list.sort(reverse=True)
			
			current_query_ranked_ids = []
			for item in ranked_documents_list:
				doc_id_only = item[1]
				current_query_ranked_ids.append(doc_id_only)
			
			# Adding documents that did not match any word in the query at the end
			# because they have 0 similarity score
			for docID in self.docIDs:
				if docID not in document_scores:
					current_query_ranked_ids.append(docID)
			
			# Append the final ranked document IDs for this query
			doc_IDs_ordered.append(current_query_ranked_ids)
	
		return doc_IDs_ordered
