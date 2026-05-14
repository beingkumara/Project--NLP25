# informationRetrievalBM25.py
# ============================
# BM25 (Okapi BM25) based retrieval for Part 5.
#
# The main problem with our basic TF-IDF approach was that it treated every
# extra occurrence of a word as equally important, and it also did not handle
# document length differences properly.
#
# BM25 fixes this with two things:
#   - It limits how much a word's count can contribute (so 100 occurrences
#     is NOT treated as 100x more important than 1 occurrence).
#   - It divides by a length factor so short focused documents are not
#     beaten by long documents just because they have more words.
#
# The scoring formula I am using is:
#   score = sum of [ IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl)) ]
# where IDF uses the Robertson variant: log((N - df + 0.5)/(df + 0.5) + 1)
# This keeps all IDF values positive even for very common words.

import math
from util import build_inverted_index, count_query_words, sort_and_complete_ranking

class InformationRetrievalBM25:

    def __init__(self, k1=1.5, b=0.75):
        """
        Parameters
        ----------
        k1 : float
            Term frequency saturation parameter. Controls how much a term's
            frequency contributes beyond a single occurrence.
        b : float
            Document length normalisation parameter. b=0 disables length
            normalisation, b=1 fully normalises by document length.
            
        Outputs / Returns
        -----------------
        None.
        We initialize the BM25 scoring parameters 
        so we can dynamically adjust the penalization weights for document lengths.
        """
        self.k1 = k1
        self.b = b

        # We use these dictionaries to securely store our word index and document metrics
        self.index = None
        self.docIDs = None
        self.doc_lengths = None
        self.avg_doc_length = 0

    def buildIndex(self, docs, docIDs):
        """
        Build an inverted index and precompute document lengths.

        Parameters
        ----------
        docs : list
            A 3D list of lists representing the documents.
        docIDs : list
            The corresponding list of document IDs.

        Returns
        -------
        None.
        """
        self.docIDs = docIDs

        # Using the shared utility function for the core inverted index
        self.index = build_inverted_index(docs, docIDs)

        # Now we compute document lengths separately since BM25 needs them
        # This is specific to BM25, so we keep it here
        self.doc_lengths = {}
        total_words = 0
        total_documents = len(docs)

        for doc_index in range(total_documents):
            d_id = docIDs[doc_index]
            word_count = 0

            # Counting total words in this document
            for sent_index in range(len(docs[doc_index])):
                sentence = docs[doc_index][sent_index]
                for word_index in range(len(sentence)):
                    word_count = word_count + 1

            self.doc_lengths[d_id] = word_count
            total_words = total_words + word_count

        # Now calculating the average document length across the corpus for the BM25 formula
        if total_documents > 0:
            self.avg_doc_length = float(total_words) / float(total_documents)
        else:
            self.avg_doc_length = 1.0 # avoiding division by zero just in case

    def calculate_robertson_idf(self, word):
        """
        Robertson-Sparck Jones IDF variant used in BM25.
        
        The standard IDF can mathematically drop below 0 if a term appears in more than 
        half the corpus. To solve this, we add +1 inside the log to force 
        all our IDF values to be strictly positive.
        """
        N = len(self.docIDs)
        df = len(self.index[word])
        
        numerator = (N - df) + 0.5
        denominator = df + 0.5
        
        ans = math.log((numerator / denominator) + 1.0)
        return ans

    def calculate_bm25_score_for_term(self, word, doc_id, idf_val):
        """
        Compute the BM25 contribution of a single term for a single document.
        
        We apply the formal BM25 equation here. This fraction balances term frequency 
        saturation (k1) with relative document length normalization (b).
        """
        tf_value = self.index[word][doc_id]
        length_of_doc = self.doc_lengths[doc_id]
        
        length_ratio = float(length_of_doc) / float(self.avg_doc_length)
        
        # Numerator calculation
        upper_part = tf_value * (self.k1 + 1.0)
        
        # Denominator calculation
        lower_part = tf_value + self.k1 * (1.0 - self.b + self.b * length_ratio)
        
        final_term_score = idf_val * (upper_part / lower_part)
        return final_term_score

    def rank(self, queries):
        """
        Rank all documents for each query using BM25.

        Parameters
        ----------
        queries : list
            A 3D list of lists representing the queries.

        Returns
        -------
        list
            A list of lists containing ranked document IDs for each query.
        """
        # Base condition checking
        if self.index is None or len(self.docIDs) == 0:
            result_list = []
            for q in range(len(queries)):
                result_list.append([])
            return result_list

        final_ranked_lists = []

        # Iterate through all the user queries
        for q_index in range(len(queries)):
            query = queries[q_index]
            
            # Using our shared query word counting function
            words_in_current_query, max_tf_q = count_query_words(query)
            
            doc_scores_for_this_query = {}
            
            # Now checking terms and adding scores based on BM25 formula
            for word in words_in_current_query.keys():
                if word in self.index:
                    idf_val = self.calculate_robertson_idf(word)
                    
                    # Only calculate scores for documents that actually have this word
                    for d_id in self.index[word].keys():
                        score_for_this_word = self.calculate_bm25_score_for_term(word, d_id, idf_val)
                        
                        if d_id not in doc_scores_for_this_query:
                            doc_scores_for_this_query[d_id] = 0.0
                            
                        doc_scores_for_this_query[d_id] = doc_scores_for_this_query[d_id] + score_for_this_word
            
            # Using our shared sort and complete ranking function
            ordered_doc_ids = sort_and_complete_ranking(doc_scores_for_this_query, self.docIDs)
            final_ranked_lists.append(ordered_doc_ids)

        return final_ranked_lists
