from util import *

# Add your import statements here
# All core NLP imports are available via: from util import *
# Explicit imports for clarity:
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
import re
import spacy


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# We will store the full list of tokenized sentences here
		tokenizedText = []

		# Let us iterate through each sentence in the input list
		for sentence in text:
			# This list will hold words for the current sentence
			words_in_sentence = []
			
			# We first replace punctuation with spaces so we don't attach them to words
			# As studied, naive tokenizers just strip punctuation and split by whitespace
			clean_sentence = ""
			for char in sentence:
				if char == '.' or char == ',' or char == '?' or char == '!' or char == ';' or char == ':' or char == '-' or char == '(' or char == ')' or char == '"' or char == "'":
					clean_sentence = clean_sentence + " "
				else:
					clean_sentence = clean_sentence + char
			
			# Now we split the cleaned sentence by spaces using the built-in split command
			# Split without arguments splits by any whitespace
			split_words = clean_sentence.split()
			
			# We will append the words to our list
			for word in split_words:
				words_in_sentence.append(word)
				
			# Finally, we add the list of tokens for this sentence to the main output list
			tokenizedText.append(words_in_sentence)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# We initialize the Penn Treebank Tokenizer from NLTK
		# This uses regular expressions to tokenize text based on Penn Treebank rules
		tokenizer = TreebankWordTokenizer()
		
		# Now we process each sentence
		for sentence in text:
			# The tokenize method splits the sentence into a list of tokens
			tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		
		# spaCy uses a hybrid approach combining rule-based and statistical methods.
		# Unlike the naive approach, spaCy understands word boundaries much better
		# because it also considers prefixes, suffixes, infixes, and special exception lists.
		# We load the small English model which is enough for tokenization.
		nlp = spacy.load("en_core_web_sm")

		# We loop through each sentence in the input list
		for sentence in text:
			# Processing the sentence through the full spaCy NLP pipeline
			# This is slower than the other two methods because spaCy builds
			# an entire dependency parse tree in the background
			doc = nlp(sentence)
			
			# We will store the extracted tokens for this sentence here
			sentence_tokens = []
			
			# The doc object is an iterator of Token objects
			# Each Token has a .text attribute which gives us the raw string
			for token in doc:
				# We extract just the text and add it to our list
				sentence_tokens.append(token.text)
				
			# Appending the list of tokens for this sentence to our main output
			tokenizedText.append(sentence_tokens)

		return tokenizedText
