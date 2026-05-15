from util import *
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
import re
import spacy


class Tokenization:

    def naive(self, text):
        ans_text = []

        for sentence in text:
            words = []

            clean_s = ""
            for char in sentence:
                if (
                    char == "."
                    or char == ","
                    or char == "?"
                    or char == "!"
                    or char == ";"
                    or char == ":"
                    or char == "-"
                    or char == "("
                    or char == ")"
                    or char == '"'
                    or char == "'"
                ):
                    clean_s = clean_s + " "
                else:
                    clean_s = clean_s + char

            split_w = clean_s.split()

            for w in split_w:
                words.append(w)

            ans_text.append(words)

        return ans_text

    def pennTreeBank(self, text):
        ans_text = []

        tokenizer = TreebankWordTokenizer()

        for sentence in text:
            tokens = tokenizer.tokenize(sentence)
            ans_text.append(tokens)

        return ans_text

    def spacyTokenizer(self, text):
        # Let us implement the basic tokenization logic
        ans_text = []

        # We are loading the spacy model here
        nlp = spacy.load("en_core_web_sm")

        for sentence in text:
            doc = nlp(sentence)

            curr_tokens = []

            for t in doc:
                curr_tokens.append(t.text)

            ans_text.append(curr_tokens)

        return ans_text
