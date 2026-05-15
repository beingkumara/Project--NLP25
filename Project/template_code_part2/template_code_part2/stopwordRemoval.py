from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math


class StopwordRemoval:

    def __init__(self):
        # Let us set up our main variables first
        self.data_driven_stopwords = None

    def get_data_driven_stopwords(self, docs, docIDs):
        # Basically we use the standard formula logic here
        doc_freqs = dict()
        list_tf = []

        total_docs = len(docs)

        for doc in docs:
            words = word_tokenize(doc.lower())

            # We only want to keep the valid alphabets
            clean_words = []
            for w in words:
                if w.isalpha():
                    clean_words.append(w)

            tf_dict = dict()
            for w in clean_words:
                if w in tf_dict:
                    tf_dict[w] = tf_dict[w] + 1
                else:
                    tf_dict[w] = 1

            list_tf.append(tf_dict)

            for w in tf_dict.keys():
                if w in doc_freqs:
                    doc_freqs[w] = doc_freqs[w] + 1
                else:
                    doc_freqs[w] = 1

        sum_aug_tf = dict()

        for tf_dict in list_tf:
            if len(tf_dict) == 0:
                continue

            max_f = 0
            for w in tf_dict:
                if tf_dict[w] > max_f:
                    max_f = tf_dict[w]

            for w in tf_dict:
                freq = tf_dict[w]
                aug_tf = 0.5 + 0.5 * (freq / max_f)

                if w in sum_aug_tf:
                    sum_aug_tf[w] = sum_aug_tf[w] + aug_tf
                else:
                    sum_aug_tf[w] = aug_tf

        tfidf_dict = dict()

        for w in doc_freqs:
            df = doc_freqs[w]
            idf = math.log10(total_docs / df)

            avg_aug_tf = sum_aug_tf[w] / df

            final_tfidf = avg_aug_tf * idf
            tfidf_dict[w] = final_tfidf

        def get_val(item):
            return item[1]

        sorted_list = sorted(tfidf_dict.items(), key=get_val)

        nltk_count = len(stopwords.words("english"))

        data_stopwords = []
        for i in range(nltk_count):
            w, s = sorted_list[i]
            data_stopwords.append(w)

        self.data_driven_stopwords = set(data_stopwords)
        self.sorted_tfidf_scores = sorted_list

        return self.data_driven_stopwords

    def fromDataDriven(self, text):
        # Here we calculate the main metric
        if self.data_driven_stopwords == None:
            print("calculating stopwords first fallback")
            return self.fromList(text)

        ans_sentences = []

        for sentence in text:
            filtered = []

            for word in sentence:
                lower_w = word.lower()

                if lower_w not in self.data_driven_stopwords:
                    filtered.append(word)

            ans_sentences.append(filtered)

        return ans_sentences

    def fromList(self, text):

        # Let us just use the standard nltk stopwords
        stop_set = set(stopwords.words("english"))

        ans_sentences = []

        for sentence in text:
            filtered = []

            for word in sentence:
                lower_w = word.lower()

                if lower_w not in stop_set:
                    filtered.append(word)

            ans_sentences.append(filtered)

        return ans_sentences
