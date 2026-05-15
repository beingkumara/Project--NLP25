from util import *
import re
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation:

    def naive(self, text):

        ans_text = []

        curr_sentence = ""

        for char in text:
            curr_sentence = curr_sentence + char

            # Here we check for punctuation correctly
            if char == "." or char == "?" or char == "!":
                ans_text.append(curr_sentence.strip())
                curr_sentence = ""

        if len(curr_sentence.strip()) > 0:
            ans_text.append(curr_sentence.strip())

        return ans_text

    def punkt(self, text):
        # Here we use the standard punkt tokenizer
        ans_text = sent_tokenize(text)
        return ans_text

    def spacySegmenter(self, text):
        ans_text = []

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        for sentence in doc.sents:
            ans_text.append(str(sentence).strip())

        return ans_text
