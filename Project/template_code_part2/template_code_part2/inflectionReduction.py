from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


class InflectionReduction:

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""
		
		# Porter Stemmer uses a set of 5 sequential algorithmic rules to strip suffixes.
		# It is a rule-based approach and does not look at the actual English dictionary context.
		
		# First we initialize the stemmer object
		stemmer = PorterStemmer()
		
		# We need a list to hold the final processed sentences
		result_sentences = []
		
		# We must iterate over each sentence in our provided document text
		for sentence in text:
			# Here we store the stemmed tokens for the current sentence
			stemmed_sentence = []
			
			# We loop through each individual token/word
			for word in sentence:
				# We apply the stemming rule to the word
				stemmed_word = stemmer.stem(word)
				# Then we add the stemmed word back to our sentence list
				stemmed_sentence.append(stemmed_word)
				
			# Finally, we append the fully reconstructed sentence back
			result_sentences.append(stemmed_sentence)
			
		return result_sentences



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""
		
		# Lemmatization is different from stemming. Instead of just chopping off word endings,
		# it actually looks up the word in a dictionary (like WordNet) and finds the proper base form.
		# It is slower because of this lookup, but the results are actual English words which is better.
		
		# Initializing the WordNet lemmatizer from NLTK
		lemmatizer = WordNetLemmatizer()
		
		# List to hold the output sentences
		result_sentences = []
		
		# Iterating through every sentence in the text matrix
		for sentence in text:
			# Temporary list for the words in this specific sentence
			lemmatized_sentence = []
			
			# Iterating over each token
			for word in sentence:
				# Using the default lemmatize function. 
				# In theory, passing the exact Parts of Speech (POS) tag gives better results,
				# but for a basic implementation, we use the default noun context.
				lemmatized_word = lemmatizer.lemmatize(word)
				
				# Appending the lemmatized word into our sentence container
				lemmatized_sentence.append(lemmatized_word)
				
			# Adding the completed sentence to the overall document list
			result_sentences.append(lemmatized_sentence)
			
		return result_sentences



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""
		
		# Earlier I had tried to return both the lemmatized and stemmed forms 
		# inside a single list so that I could evaluate both later.
		# However, this broke the pipeline(in the main.py) for the next module which only expects a single list.
		# So I am just using the Lemmatizer because it preserves actual dictionary words better.
		# We can modify the function args to choose between lemmatizer and stemmer
		# For now, I am just using the Lemmatizer
		lemmatized_output = self.wordnetLemmatizer(text)
		# stemmed_output = self.porterStemmer(text)
		
		# Returning only one so the stopword remover does not crash
		return lemmatized_output
