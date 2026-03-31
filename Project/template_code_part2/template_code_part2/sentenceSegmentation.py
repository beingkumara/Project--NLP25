from util import *

# Add your import statements here
# All core NLP imports are available via: from util import *
import re
import spacy
from nltk.tokenize import sent_tokenize

# spacy gets loaded down here because of some error if done at the top
class SentenceSegmentation():


	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		# A very naive approach is to just split whenever we see a period, question mark, or exclamation.
		current_sentence = ""
		
		# We will iterate through each character in the string
		for char in text:
			current_sentence = current_sentence + char
			
			# Checking for the base condition of sentence-ending punctuation
			if char == '.' or char == '?' or char == '!':
				# If we find a punctuation, we append the accumulated sentence to our list
				# We also strip any leading/trailing whitespace to keep it clean
				segmentedText.append(current_sentence.strip())
				# Reset the current sentence for the next sentence
				current_sentence = ""
		
		# If there is any remaining text that didn't end with a punctuation, we add it too
		if len(current_sentence.strip()) > 0:
			segmentedText.append(current_sentence.strip())

		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# Here we use the NLTK Punkt segmenter.
		# This uses an unsupervised machine learning model to understand abbreviations, collocations, and words that start sentences.
		
		# sent_tokenize is the wrapper function for the pre-trained unsupervised ML model
		segmentedText = sent_tokenize(text)

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		# Spacy works differently from Punkt. It tries to understand the grammar of the text
		# and figures out sentence boundaries from the structure instead of just looking at periods.
		# This makes it slower but much more accurate on tricky sentences.
		nlp = spacy.load("en_core_web_sm")
		spacy_doc = nlp(text)
		
		# spacy_doc.sents is an iterator, so we will use a for loop to extract each sentence
		for sentence in spacy_doc.sents:
			# We convert the spacy Span object back into a normal Python string
			# We also strip any leading/trailing whitespace to keep it clean
			sentence_string = str(sentence)
			segmentedText.append(sentence_string.strip())

		return segmentedText
