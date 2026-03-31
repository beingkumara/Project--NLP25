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

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
			Example: [[['word1', 'word2'], ['word3']], [['word4']]]
		arg2 : list
			A list of integers denoting IDs of the documents
			Example: [1, 2, 3]

		Returns
		-------
		None
		"""

		# Let us initialize the dictionary for our index
		number_of_documents = len(docs)
		index = dict()
		
		# We will iterate through each document one by one as taught in class
		for document_index in range(number_of_documents):
			docID = docIDs[document_index]
			
			# Now we must go through each sentence in the current document
			for sentence in docs[document_index]:
				
				# Next, we check every single word in the sentence
				for word in sentence:
					
					# If the word is not already in our index dictionary, we add it
					if word not in index:
						index[word] = dict()
					
					# If the document ID is not recorded for this word, we start the count at 0
					if docID not in index[word]:
						index[word][docID] = 0
					
					# Finally, we increment the frequency of the word for this document
					index[word][docID] = index[word][docID] + 1
					# print("Added word:", word, "to doc:", docID)

		# Assigning the index to the class variable so we can use it later
		self.index = index
		self.docIDs = docIDs


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# If buildIndex was never called or if there are no documents, we cannot do anything useful.
		if self.index is None or len(self.docIDs) == 0:
			# Returning an empty list for each query so the caller doesn't get a crash
			empty_result = []
			for q in queries:
				empty_result.append([])
			return empty_result

		doc_IDs_ordered = []

		# IDF stands for Inverse Document Frequency and the idea is:
		# Words that appear in almost every document are not very useful for ranking
		# so we penalize them by giving a low IDF. The formula is log10(N / df).
		# N = total number of documents, df = in how many documents this word is present.
		# I am storing IDF separately in a dict so I can reuse it for every query.
		
		total_documents = len(self.docIDs)
		# print("total number of documents:", total_documents)  # checking this value once
		idf_values = dict()
		
		# Going through each word one by one in my inverted index
		for word in self.index:
			# df is just how many documents have this word
			# Since self.index[word] is a dict of {docID: count}, the length gives df
			document_frequency = len(self.index[word])
			
			# Now applying the IDF formula: log base 10 of (N / df)
			idf_values[word] = math.log10(total_documents / document_frequency)

		# Before computing norms, I need to find the maximum TF in each document.
		# This is needed for Augmented TF, whose formula is: 0.5 + 0.5 * (tf / max_tf_in_doc)
		# it prevents very long documents from dominating the scoring.
		# So a word appearing 4 times in a 4-word doc gets the same normalized weight as
		# a word appearing 100 times in a 100-word doc.
		
		# find the max TF for each document across all words
		max_tf_per_doc = dict()
		for word in self.index:
			for docID in self.index[word]:
				raw_tf = self.index[word][docID]
				if docID not in max_tf_per_doc:
					max_tf_per_doc[docID] = raw_tf
				else:
					if raw_tf > max_tf_per_doc[docID]:
						max_tf_per_doc[docID] = raw_tf

		# For cosine similarity, we need to divide the dot product by the magnitudes (norms)
		# of both the query vector and the document vector.
		# Now using augmented TF in the weight: weight = (0.5 + 0.5 * tf/max_tf) * idf
		# I will pre-compute the document norms here so I don't redo it for every query.
		
		doc_vector_norms = dict()
		
		for word in self.index:
			idf_w = idf_values[word]
			# Going inside each document that contains this word
			for docID in self.index[word]:
				raw_tf_d = self.index[word][docID]
				max_tf_d = max_tf_per_doc[docID]
				
				# Augmented TF formula: 0.5 + 0.5 * (raw_tf / max_tf)
				# This is to handle the length normalization problem discussed in class
				augmented_tf_d = 0.5 + 0.5 * (raw_tf_d / max_tf_d)
				
				# The weight of this word in this document using augmented TF
				weight_d = augmented_tf_d * idf_w
				
				# Adding squared weight for norm computation
				if docID in doc_vector_norms:
					doc_vector_norms[docID] = doc_vector_norms[docID] + (weight_d * weight_d)
				else:
					doc_vector_norms[docID] = (weight_d * weight_d)
		
		# After accumulating the sum of squares, we take the square root to get the actual norm length
		# This is basically the magnitude of the document vector in the tf-idf space
		for docID in doc_vector_norms:
			doc_vector_norms[docID] = math.sqrt(doc_vector_norms[docID])
			# print("doc:", docID, "norm:", doc_vector_norms[docID])

		# Now we will process each query one by one to find the best ranking documents
		for query in queries:
			
			# We need to find the term frequency (TF) for words in the query
			query_tf = dict()
			for sentence in query:
				for word in sentence:
					# Checking if word is already recorded in our temporary dictionary
					if word in query_tf:
						query_tf[word] = query_tf[word] + 1
					else:
						query_tf[word] = 1
			
			# Now we create the TF-IDF vector for the query
			query_vector = dict()
			query_vector_norm = 0
			
			# For the query, I also need augmented TF. So first I find max TF in this query.
			max_tf_in_query = 0
			for word in query_tf:
				if query_tf[word] > max_tf_in_query:
					max_tf_in_query = query_tf[word]
			# just in case the query is somehow empty, to avoid divide by zero
			if max_tf_in_query == 0:
				max_tf_in_query = 1
			
			for word in query_tf:
				raw_tf_q = query_tf[word]
				
				# Augmented TF for the query word - same formula as for documents
				augmented_tf_q = 0.5 + 0.5 * (raw_tf_q / max_tf_in_query)
				
				# If the query word exists in our index, we use its IDF
				if word in idf_values:
					weight_of_query_word = augmented_tf_q * idf_values[word]
					query_vector[word] = weight_of_query_word
					
					# Squaring it for norm computation (we need query magnitude for cosine similarity)
					query_vector_norm = query_vector_norm + (weight_of_query_word * weight_of_query_word)
					
			# Taking the square root to get the final norm = ||q|| 
			query_vector_norm = math.sqrt(query_vector_norm)

			# Next step is to calculate the similarity scores for each document
			# We will use simple dot product for the basic Vector Space Model
			document_scores = dict()
			
			for word in query_vector:
				weight_of_query_word = query_vector[word]
				
				# We check every document that contains this query word
				for docID in self.index[word]:
					tf_value_in_document = self.index[word][docID]
					
					# the document word weight is augmented_tf * IDF.
					# computing augmented TF of the document word here using the max_tf we found earlier.
					max_tf_this_doc = max_tf_per_doc[docID]
					augmented_tf_doc = 0.5 + 0.5 * (tf_value_in_document / max_tf_this_doc)
					weight_of_document_word = augmented_tf_doc * idf_values[word]
					
					# This gives us one term of the dot product: q_weight * d_weight for this word
					score_contribution = weight_of_query_word * weight_of_document_word
					# print("word:", word, "docID:", docID, "contribution:", score_contribution)
					
					# Adding this contribution to the docID's running total score
					if docID in document_scores:
						document_scores[docID] = document_scores[docID] + score_contribution
					else:
						document_scores[docID] = score_contribution

			# We already calculated the dot product (numerator) above.
			# Now we must normalize the scores to get Cosine Similarity using the denominators.
			# Cosine Similarity = Dot Product / (Query Norm * Document Norm)
			for docID in document_scores:
				dot_product = document_scores[docID]
				
				# Denominator = |Q| * |D|
				# Using the query norm we calculated earlier and document norm we calculated at the start
				query_norm_value = query_vector_norm
				doc_norm_value = doc_vector_norms[docID]
				
				# We only divide if both norms are greater than 0 to avoid Division by Zero error
				if query_norm_value > 0 and doc_norm_value > 0:
					cosine_similarity = dot_product / (query_norm_value * doc_norm_value)
				else:
					cosine_similarity = 0
					
				# Updating the score to be the normalized cosine similarity
				document_scores[docID] = cosine_similarity

			# Now we must rank the documents based on their accumulated scores
			ranked_documents_list = []
			for docID in document_scores:
				final_score = document_scores[docID]
				# We store final_score at index 0 and docID at index 1 so we can sort easily without lambda
				ranked_documents_list.append([final_score, docID])
			
			# ordering the documents by score in descending order
			ranked_documents_list.sort(reverse=True)
			
			# Extract just the document IDs in their ranked order (top scoring ones first)
			current_query_ranked_ids = []
			for item in ranked_documents_list:
				doc_id_only = item[1]
				current_query_ranked_ids.append(doc_id_only)
			
			# Now I need to append documents that had zero overlap with the query.
			# These documents are not in document_scores at all because they share no words with the query.
			# But for MAP@k evaluation, the output list must have all len(self.docIDs) entries.
			# So I just push these zero-score docs at the end in their original order.
			for docID in self.docIDs:
				if docID not in document_scores:
					# This doc had no matching query word, so its effective score is 0
					# We add it at the tail end since it shouldn't rank above actual matches
					# print("appending unranked doc:", docID)
					current_query_ranked_ids.append(docID)
			
			# Adding this query's complete ranked IDs to our main list
			doc_IDs_ordered.append(current_query_ranked_ids)
	
		return doc_IDs_ordered




