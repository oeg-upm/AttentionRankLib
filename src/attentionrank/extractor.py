# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

"""Contain method that return list of candidate"""

import re
import nltk

GRAMMAR_EN = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR_DE = """
NBAR:
        {<JJ|CARD>*<NN.*>+}  # [Adjective(s) or Article(s) or Posessive pronoun](optional) + Noun(s)
        {<NN>+<PPOSAT><JJ|CARD>*<NN.*>+}

NP:
{<NBAR><APPR|APPRART><ART>*<NBAR>}# Above, connected with APPR and APPART (beim vom)
{<NBAR>+}
"""

GRAMMAR_FR = """  NP:
        {<NN.*|JJ>*<NN.*>+<JJ>*}  # Adjective(s)(optional) + Noun(s) + Adjective(s)(optional)"""


def get_grammar(lang):
    if lang == 'en':
        grammar = GRAMMAR_EN
    elif lang == 'de':
        grammar = GRAMMAR_DE
    elif lang == 'fr':
        grammar = GRAMMAR_FR
    else:
        raise ValueError('Language not handled')
    return grammar


def extract_candidates(text_obj, no_subset=False, repeat=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :param repeat: return candidate whenever it shows, not just show once
    :param lang: language (currently en, fr and de are supported)
    :return: list of candidate phrases (string)
    """
    if repeat:
        keyphrase_candidate = []
    else:
        keyphrase_candidate = set()

    np_parser = nltk.RegexpParser(get_grammar(text_obj.lang))  # Noun phrase parser
    print(text_obj.pos_tagged)
    trees = np_parser.parse_sents(text_obj.pos_tagged)  # Generator with one tree per sentence
    print(trees)
    print("NO DEBERÍA ESTAR AQUÍ")

    for p, tree in enumerate(trees):
        # print(p, tree)
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            if repeat:
                keyphrase_candidate.append(' '.join(word for word, tag in subtree.leaves()))
            else:
                keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

    if repeat:
        keyphrase_candidate = [kp for kp in keyphrase_candidate if len(kp.split()) <= 5]
    else:
        keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}

    if no_subset:
        keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
    else:
        keyphrase_candidate = list(keyphrase_candidate)

    return keyphrase_candidate


def extract_sent_candidates(text_obj):
    """

    :param text_obj: input Text Representation see @InputTextObj
    :return: list of tokenized sentence (string) , each token is separated by a space in the string
    """
    return [(' '.join(word for word, tag in sent)) for sent in text_obj.pos_tagged]


def unique_ngram_candidates(strings):
    """
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    :param strings: List of string
    :return: List of string where no string is fully contained inside another string
    """
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results):
            results.append(s)
    return results
