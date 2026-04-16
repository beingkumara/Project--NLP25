"""
This file handles adding more words (synonyms) to the user's queries to improve recall.
As studied in theory class, standard vector space models fail when there is a vocabulary mismatch (e.g., thermal vs heated).
To solve this, I am using the WordNet library from NLTK.
"""

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Setting this to 3 so we do not get too many unrelated words and face query drift
LIMIT_MAX_SYNONYMS = 3 
# We give less weight to the expanded words so the original meaning is not lost
WEIGHT_FOR_EXPANDED_WORD = 0.5 

class QueryExpander:
    # Here we define the class targeting Query Expansion via semantic thesauruses.

    def __init__(self):
        # Let us initialize the lemmatizer object here natively
        self.lemmatizer = WordNetLemmatizer()

    def get_synonyms_from_wordnet(self, current_word):
        """
        This function is responsible for finding the synonyms.

        Parameters
        ----------
        current_word : str
            The word for which we need to find synonyms.

        Returns
        -------
        list
            A list of strings containing the found synonyms.
        """
        all_synsets = wordnet.synsets(current_word)
        
        # Base condition checking
        if len(all_synsets) == 0:
            return []

        # Here we only take the primary meaning
        primary_meaning_synset = all_synsets[0]

        found_synonyms = []
        all_lemmas_found = primary_meaning_synset.lemmas()
        
        for lemma_index in range(len(all_lemmas_found)):
            lemma_obj = all_lemmas_found[lemma_index]
            synonym_string = lemma_obj.name().lower()

            # We will ignore synonyms with multiple words having underscores
            if '_' in synonym_string:
                continue

            # We should not add the exact same word back again
            if synonym_string == current_word.lower():
                continue

            # Checking the limit constraint for max synonyms
            if len(found_synonyms) >= LIMIT_MAX_SYNONYMS:
                break
            
            found_synonyms.append(synonym_string)

        return found_synonyms

    def find_all_synonyms_for_query(self, query):
        """
        Here we manually iterate through the query and find synonyms for each individual token.

        Parameters
        ----------
        query : list
            A 3D list (sentences) containing tokens.

        Returns
        -------
        tuple
            (list_of_original_words, list_of_new_synonyms, map_of_expansion).
        """
        list_of_original_words = []
        
        for sent_idx in range(len(query)):
            sentence = query[sent_idx]
            for word_idx in range(len(sentence)):
                list_of_original_words.append(sentence[word_idx])

        list_of_new_synonyms = []
        map_of_expansion = {}

        for word_index in range(len(list_of_original_words)):
            original_word = list_of_original_words[word_index]
            
            # Fetch synonyms using theoretical method
            curr_synonyms = self.get_synonyms_from_wordnet(original_word)
            
            if len(curr_synonyms) > 0:
                map_of_expansion[original_word] = curr_synonyms
                for s_index in range(len(curr_synonyms)):
                    list_of_new_synonyms.append(curr_synonyms[s_index])

        return list_of_original_words, list_of_new_synonyms, map_of_expansion

    def expand_query_for_ir(self, query):
        """
        This function returns the expanded query in the correct 3D list format.

        Parameters
        ----------
        query : list
            A 3D list (sentences) containing tokens.

        Returns
        -------
        tuple
            (expanded_query_list, map_of_expansion).
        """
        original_words, new_synonyms, map_of_expansion = self.find_all_synonyms_for_query(query)

        # If we did not find any synonyms, we return original directly
        if len(new_synonyms) == 0:
            return query, map_of_expansion

        expanded_query_list = []
        
        # Start by strictly copying the original query
        for sent_idx in range(len(query)):
            curr_sent = []
            for word_idx in range(len(query[sent_idx])):
                curr_sent.append(query[sent_idx][word_idx])
            expanded_query_list.append(curr_sent)

        # Appending the new synonyms as an extra physical sentence completely
        if len(new_synonyms) > 0:
            expanded_query_list.append(new_synonyms)
            
        return expanded_query_list, map_of_expansion
