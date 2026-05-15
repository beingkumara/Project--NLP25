from util import *

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


class InflectionReduction:

    def porterStemmer(self, text):
        # Let us append the results here
        # Here we initialize the standard porter stemmer
        stemmer = PorterStemmer()

        # Basically to store our final processed sentences
        ans_sentences = []

        for sentence in text:
            curr_sentence = []

            for word in sentence:
                # We stem the word here
                stemmed_word = stemmer.stem(word)
                curr_sentence.append(stemmed_word)

            ans_sentences.append(curr_sentence)

        return ans_sentences

    def wordnetLemmatizer(self, text):
        # From what we studied, wordnet lemmatizer is better for actual english words
        lemmatizer = WordNetLemmatizer()

        ans_sentences = []

        for sentence in text:
            curr_sentence = []

            for word in sentence:
                # Lemmatizing the word properly
                lemma_word = lemmatizer.lemmatize(word)
                curr_sentence.append(lemma_word)

            ans_sentences.append(curr_sentence)

        return ans_sentences

    def reduce(self, text):
        # Finally returning the lemmatizer output
        ans = self.wordnetLemmatizer(text)
        return ans
