
# Add your import statements here
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math


class StopwordRemoval():

	def __init__(self):
		# We initialize this variable to store our customized data-driven stopwords
		self.data_driven_stopwords = None

	def get_data_driven_stopwords(self, docs, docIDs):
		"""
		Calculates the stopwords dynamically from the dataset using Augmented TF-IDF.
		As discussed in the lectures, words with very low TF-IDF scores across the entire corpus
		are highly frequent and do not help in distinguishing between documents.

		Parameters
		----------
		docs : list
			A 3D list of lists representing the documents.
		docIDs : list
			The corresponding list of document IDs.

		Returns
		-------
		list
			A list of strings containing the calculated stopwords.
		"""
		# We use standard dictionary to count frequencies
		document_frequencies = dict()
		list_of_term_frequencies = []
		
		# We need to know total number of documents for IDF calculation
		total_documents = len(dataset_docs)
		
		for document in dataset_docs:
			# Tokenize the document
			words_in_doc = word_tokenize(document.lower())
			
			# We only want to keep alphabetic words to ignore numbers and punctuation
			clean_words = []
			for word in words_in_doc:
				if word.isalpha():
					clean_words.append(word)
			
			# Count word occurrences in this specific document
			term_frequencies = dict()
			for word in clean_words:
				if word in term_frequencies:
					term_frequencies[word] = term_frequencies[word] + 1
				else:
					term_frequencies[word] = 1
					
			list_of_term_frequencies.append(term_frequencies)
			
			# Update document frequency. If a word appears in this document, we increment its DF by 1.
			for word in term_frequencies.keys():
				if word in document_frequencies:
					document_frequencies[word] = document_frequencies[word] + 1
				else:
					document_frequencies[word] = 1
					
		# Now we calculate the Augmented TF and the final TF-IDF scores
		sum_of_augmented_tf = dict()
		
		for term_freq_dict in list_of_term_frequencies:
			if len(term_freq_dict) == 0:
				continue
				
			# Find the maximum frequency in this document for normalization
			max_frequency = 0
			for word in term_freq_dict:
				if term_freq_dict[word] > max_frequency:
					max_frequency = term_freq_dict[word]
					
			# Calculate Augmented TF (0.5 + 0.5 * (f / max_f))
			for word in term_freq_dict:
				frequency = term_freq_dict[word]
				augmented_tf = 0.5 + 0.5 * (frequency / max_frequency)
				
				if word in sum_of_augmented_tf:
					sum_of_augmented_tf[word] = sum_of_augmented_tf[word] + augmented_tf
				else:
					sum_of_augmented_tf[word] = augmented_tf
					
		# Calculate final TF-IDF for each word
		tfidf_scores_dictionary = dict()
		
		for word in document_frequencies:
			df = document_frequencies[word]
			idf = math.log10(total_documents / df)
			
			# Average Augmented TF across the corpus
			average_augmented_tf = sum_of_augmented_tf[word] / df
			
			final_tfidf_score = average_augmented_tf * idf
			tfidf_scores_dictionary[word] = final_tfidf_score
			
		# We will sort the dictionary to find the lowest TF-IDF scores.
		# We use a simple helper function to get the score for sorting.
		def get_score_value(item):
			return item[1]
			
		# Convert dictionary to list of tuples and sort
		sorted_tfidf_list = sorted(tfidf_scores_dictionary.items(), key=get_score_value)
		
		# We want to match the number of stopwords in NLTK (which is 198) for a fair comparison.
		# I think we can use a threshold value to remove the stopwords.
		
		number_of_nltk_stopwords = len(stopwords.words('english'))
		
		# Extract the top words with the lowest scores
		data_driven_stopwords_list = []
		for i in range(number_of_nltk_stopwords):
			word, score = sorted_tfidf_list[i]
			data_driven_stopwords_list.append(word)
			
		# Convert to a set for O(1) time complexity lookups later
		self.data_driven_stopwords = set(data_driven_stopwords_list)
		
		# I am also saving the sorted list internally because I need it for plotting graphs later.
		self.sorted_tfidf_scores = sorted_tfidf_list
		
		return self.data_driven_stopwords

	def fromDataDriven(self, text):
		"""
		Stopword Removal using the Data-Driven (TF-IDF) stopword list

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		# If the stopwords have not been generated yet, we fallback to fromList as a safety measure.
		if self.data_driven_stopwords == None:
			print("Warning: Data-driven stopwords not calculated. Please call get_data_driven_stopwords() first.")
			print("Falling back to NLTK stopwords.")
			return self.fromList(text)
			
		# We initialize the master list of filtered sentences
		result_sentences = []
		
		# We iterate over every single sentence given in the input
		for sentence in text:
			# List to store words that are NOT stopwords
			filtered_sentence = []
			
			# We iterate word by word
			for word in sentence:
				# We convert the word to lower case to ensure case-insensitive matching.
				lowercased_word = word.lower()
				
				# Checking if the word is a stopword based on our data-driven set
				if lowercased_word not in self.data_driven_stopwords:
					filtered_sentence.append(word)
					
			# Append the successfully filtered sentence to the final document
			result_sentences.append(filtered_sentence)

		return result_sentences

	def fromList(self, text):
		"""
		Stopword Removal using the NLTK stopwords list

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		# Stopwords are words like "the", "is", "and" etc that appear everywhere
		# but don't really help us tell one document apart from another.
		# Getting rid of them makes our vocabulary much smaller and the whole search faster.
		
		# Fetching the NLTK English stopwords and putting them in a set.
		# I use a set here because checking "is this word a stopword?" is O(1) in a set
		# whereas in a list it would be O(N) which is slower.
		stop_words_set = set(stopwords.words('english'))
		
		# We initialize the master list of filtered sentences
		result_sentences = []
		
		# We iterate over every single sentence given in the input
		for sentence in text:
			# List to store words that are NOT stopwords
			filtered_sentence = []
			
			# We iterate word by word
			for word in sentence:
				# We convert the word to lower case to ensure case-insensitive matching.
				# E.g., 'The' and 'the' should both be removed.
				lowercased_word = word.lower()
				
				# Checking if the word is a stopword based on our set
				if lowercased_word not in stop_words_set:
					# If it is an important word, we append it to our filtered list
					# We append the original word, not the lowercased one, to maintain original casing
					filtered_sentence.append(word)
					
			# Append the successfully filtered sentence to the final document
			result_sentences.append(filtered_sentence)

		return result_sentences