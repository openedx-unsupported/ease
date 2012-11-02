"""
Extracts features from training set and test set essays
"""

import numpy
import re
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from itertools import chain
import copy

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
from essay_set import EssaySet
import util_functions

if not base_path.endswith("/"):
    base_path=base_path+"/"


class FeatureExtractor(object):
    def __init__(self):
        self._good_pos_ngrams = self.get_good_pos_ngrams()
        self.dict_initialized = False

    def initialize_dictionaries(self, e_set):
        """
        Initializes dictionaries from an essay set object
        Dictionaries must be initialized prior to using this to extract features
        e_set is an input essay set
        returns a confirmation of initialization
        """
        if(hasattr(e_set, '_type')):
            if(e_set._type == "train"):
                nvocab = util_functions.get_vocab(e_set._text, e_set._score)
                svocab = util_functions.get_vocab(e_set._clean_stem_text, e_set._score)
                self._normal_dict = CountVectorizer(ngram_range=(1,2), vocabulary=nvocab)
                self._stem_dict = CountVectorizer(ngram_range=(1,2), vocabulary=svocab)
                self.dict_initialized = True
                self._mean_spelling_errors=sum(e_set._spelling_errors)/float(len(e_set._spelling_errors))
                self._spell_errors_per_character=sum(e_set._spelling_errors)/float(sum([len(t) for t in e_set._text]))
                self._grammar_errors_per_character=1-(sum(self._get_grammar_errors
                    (e_set._pos,e_set._text,e_set._tokens))/float(sum([len(t) for t in e_set._text])))
                ret = "ok"
            else:
                raise util_functions.InputError(e_set, "needs to be an essay set of the train type.")
        else:
            raise util_functions.InputError(e_set, "wrong input. need an essay set object")
        return ret

    def get_good_pos_ngrams(self):
        """
        Gets a list of gramatically correct part of speech sequences from an input file called essaycorpus.txt
        Returns the list and caches the file
        """
        if(os.path.isfile(base_path + "good_pos_ngrams.p")):
            good_pos_ngrams = pickle.load(open(base_path + 'good_pos_ngrams.p', 'rb'))
        else:
            essay_corpus = open(base_path + "essaycorpus.txt").read()
            essay_corpus = util_functions.sub_chars(essay_corpus)
            good_pos_ngrams = util_functions.regenerate_good_tokens(essay_corpus)
            pickle.dump(good_pos_ngrams, open(base_path + 'good_pos_ngrams.p', 'wb'))
        return good_pos_ngrams

    def _get_grammar_errors(self,pos,text,tokens):
        word_counts = [max(len(t),1) for t in tokens]
        good_pos_tags = []
        for i in xrange(0, len(text)):
            pos_seq = [tag[1] for tag in pos[i]]
            pos_ngrams = util_functions.ngrams(pos_seq, 2, 4)
            overlap_ngrams = [i for i in pos_ngrams if i in self._good_pos_ngrams]
            good_pos_tags.append(len(overlap_ngrams))
        return good_pos_tags

    def gen_length_feats(self, e_set):
        """
        Generates length based features from an essay set
        Generally an internal function called by gen_feats
        Returns an array of length features
        """
        text = e_set._text
        lengths = [len(e) for e in text]
        word_counts = [max(len(t),1) for t in e_set._tokens]
        comma_count = [e.count(",") for e in text]
        ap_count = [e.count("'") for e in text]
        punc_count = [e.count(".") + e.count("?") + e.count("!") for e in text]
        chars_per_word = [lengths[m] / float(word_counts[m]) for m in xrange(0, len(text))]

        good_pos_tags= self._get_grammar_errors(e_set._pos,e_set._text,e_set._tokens)
        good_pos_tag_prop = [good_pos_tags[m] / float(word_counts[m]) for m in xrange(0, len(text))]

        length_arr = numpy.array((
        lengths, word_counts, comma_count, ap_count, punc_count, chars_per_word, good_pos_tags,
        good_pos_tag_prop)).transpose()

        return length_arr.copy()

    def gen_bag_feats(self, e_set):
        """
        Generates bag of words features from an input essay set and trained FeatureExtractor
        Generally called by gen_feats
        Returns an array of features
        """
        if(hasattr(self, '_stem_dict')):
            sfeats = self._stem_dict.transform(e_set._clean_stem_text)
            nfeats = self._normal_dict.transform(e_set._text)
            bag_feats = numpy.concatenate((sfeats.toarray(), nfeats.toarray()), axis=1)
        else:
            raise util_functions.InputError(self, "Dictionaries must be initialized prior to generating bag features.")
        return bag_feats.copy()

    def gen_feats(self, e_set):
        """
        Generates bag of words, length, and prompt features from an essay set object
        returns an array of features
        """
        bag_feats = self.gen_bag_feats(e_set)
        length_feats = self.gen_length_feats(e_set)
        prompt_feats = self.gen_prompt_feats(e_set)
        overall_feats = numpy.concatenate((length_feats, prompt_feats, bag_feats), axis=1)
        overall_feats = overall_feats.copy()

        return overall_feats

    def gen_prompt_feats(self, e_set):
        """
        Generates prompt based features from an essay set object and internal prompt variable.
        Generally called internally by gen_feats
        Returns an array of prompt features
        """
        prompt_toks = nltk.word_tokenize(e_set._prompt)
        expand_syns = []
        for word in prompt_toks:
            synonyms = util_functions.get_wordnet_syns(word)
            expand_syns.append(synonyms)
        expand_syns = list(chain.from_iterable(expand_syns))
        prompt_overlap = []
        prompt_overlap_prop = []
        for j in e_set._tokens:
            tok_length=len(j)
            if(tok_length==0):
                tok_length=1
            prompt_overlap.append(len([i for i in j if i in prompt_toks]))
            prompt_overlap_prop.append(prompt_overlap[len(prompt_overlap) - 1] / float(tok_length))
        expand_overlap = []
        expand_overlap_prop = []
        for j in e_set._tokens:
            tok_length=len(j)
            if(tok_length==0):
                tok_length=1
            expand_overlap.append(len([i for i in j if i in expand_syns]))
            expand_overlap_prop.append(expand_overlap[len(expand_overlap) - 1] / float(tok_length))

        prompt_arr = numpy.array((prompt_overlap, prompt_overlap_prop, expand_overlap, expand_overlap_prop)).transpose()

        return prompt_arr.copy()

    def gen_feedback(self, e_set, features=None):
        set_grammar=self._get_grammar_errors(e_set._pos,e_set._text,e_set._tokens)
        set_grammar_per_character=[set_grammar[m]/float(len(e_set._text[m])) for m in xrange(0,len(e_set._text))]
        set_spell_errors_per_character=[e_set._spelling_errors[m]/float(len(e_set._text[m])) for m in xrange(0,len(e_set._text))]
        all_feedback=[]
        for m in xrange(0,len(e_set._text)):
            individual_feedback=[]
            if set_grammar_per_character[m]>self._grammar_errors_per_character:
                individual_feedback.append("Potential grammatical errors.")
            if set_spell_errors_per_character[m]>self._spell_errors_per_character:
                individual_feedback.append("Potential spelling errors.")
            if features is not None:
                f_row_sum=numpy.sum(features[m,12:])
                f_row_prop=f_row_sum/len(e_set._text[m])
                if f_row_prop<.05:
                    individual_feedback.append("Essay may be off topic.")
            all_feedback.append(individual_feedback)

        return all_feedback
