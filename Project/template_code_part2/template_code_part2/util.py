# util.py
# ======
# Utility functions used across the project modules.
# This file contains helper functions that are needed by the preprocessing
# and other modules in the IR system.

from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    """
    Convert a Penn Treebank POS tag to a WordNet POS tag.
    Useful for WordNet Lemmatizer (which needs wordnet.NOUN, VERB, ADJ, ADV).

    Parameters
    ----------
    treebank_tag : str
        A Penn Treebank POS tag, e.g. 'NN', 'VBZ', 'JJ', 'RB'.

    Returns
    -------
    str
        The corresponding WordNet POS constant.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun if tag is unrecognised
        return wordnet.NOUN