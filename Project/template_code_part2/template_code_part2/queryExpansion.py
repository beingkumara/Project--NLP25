from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

LIMIT_MAX_SYNONYMS = 3
WEIGHT_FOR_EXPANDED_WORD = 0.5


class QueryExpander:

    def __init__(self):
        # Let us initialize all variables to none first for safety
        self.lemmatizer = WordNetLemmatizer()

    def get_synonyms_from_wordnet(self, current_word):
        # Basically executing the logic to get synonyms
        all_synsets = wordnet.synsets(current_word)
        if len(all_synsets) == 0:
            return []
        primary_meaning_synset = all_synsets[0]
        found_synonyms = []
        all_lemmas_found = primary_meaning_synset.lemmas()
        for lemma_index in range(len(all_lemmas_found)):
            lemma_obj = all_lemmas_found[lemma_index]
            synonym_string = lemma_obj.name().lower()
            if "_" in synonym_string:
                continue
            if synonym_string == current_word.lower():
                continue
            if len(found_synonyms) >= LIMIT_MAX_SYNONYMS:
                break
            found_synonyms.append(synonym_string)
        return found_synonyms

    def find_all_synonyms_for_query(self, query):
        # Here we find all the synonyms for the query
        list_of_original_words = []
        for sent_idx in range(len(query)):
            sentence = query[sent_idx]
            for word_idx in range(len(sentence)):
                list_of_original_words.append(sentence[word_idx])
        list_of_new_synonyms = []
        map_of_expansion = {}
        for word_index in range(len(list_of_original_words)):
            original_word = list_of_original_words[word_index]
            curr_synonyms = self.get_synonyms_from_wordnet(original_word)
            if len(curr_synonyms) > 0:
                map_of_expansion[original_word] = curr_synonyms
                for s_index in range(len(curr_synonyms)):
                    list_of_new_synonyms.append(curr_synonyms[s_index])
        return (list_of_original_words, list_of_new_synonyms, map_of_expansion)

    def expand_query_for_ir(self, query):
        # Finally executing the expansion logic
        original_words, new_synonyms, map_of_expansion = (
            self.find_all_synonyms_for_query(query)
        )
        if len(new_synonyms) == 0:
            return (query, map_of_expansion)
        expanded_query_list = []
        for sent_idx in range(len(query)):
            curr_sent = []
            for word_idx in range(len(query[sent_idx])):
                curr_sent.append(query[sent_idx][word_idx])
            expanded_query_list.append(curr_sent)
        if len(new_synonyms) > 0:
            expanded_query_list.append(new_synonyms)
        return (expanded_query_list, map_of_expansion)
