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
import operator
import logging

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
from essay_set import EssaySet
import util_functions

if not base_path.endswith("/"):
    base_path=base_path+"/"

log = logging.getLogger(__name__)

#Paths to needed data files
NGRAM_PATH = base_path + "data/good_pos_ngrams.p"
ESSAY_CORPUS_PATH = util_functions.ESSAY_CORPUS_PATH

class FeatureExtractor(object):
    def __init__(self):
        self._good_pos_ngrams = self.get_good_pos_ngrams()
        self.dict_initialized = False
        self._spell_errors_per_character=0
        self._grammar_errors_per_character=0

    def initialize_dictionaries(self, e_set, max_feats2 = 200):
        """
        Initializes dictionaries from an essay set object
        Dictionaries must be initialized prior to using this to extract features
        e_set is an input essay set
        returns a confirmation of initialization
        """
        if(hasattr(e_set, '_type')):
            if(e_set._type == "train"):
                #normal text (unstemmed) useful words/bigrams
                nvocab = util_functions.get_vocab(e_set._text, e_set._score, max_feats2 = max_feats2)
                #stemmed and spell corrected vocab useful words/ngrams
                svocab = util_functions.get_vocab(e_set._clean_stem_text, e_set._score, max_feats2 = max_feats2)
                #dictionary trained on proper vocab
                self._normal_dict = CountVectorizer(ngram_range=(1,2), vocabulary=nvocab)
                #dictionary trained on proper vocab
                self._stem_dict = CountVectorizer(ngram_range=(1,2), vocabulary=svocab)
                self.dict_initialized = True
                #Average spelling errors in set. needed later for spelling detection
                self._mean_spelling_errors=sum(e_set._spelling_errors)/float(len(e_set._spelling_errors))
                self._spell_errors_per_character=sum(e_set._spelling_errors)/float(sum([len(t) for t in e_set._text]))
                #Gets the number and positions of grammar errors
                good_pos_tags,bad_pos_positions=self._get_grammar_errors(e_set._pos,e_set._text,e_set._tokens)
                self._grammar_errors_per_character=(sum(good_pos_tags)/float(sum([len(t) for t in e_set._text])))
                #Generate bag of words features
                bag_feats=self.gen_bag_feats(e_set)
                #Sum of a row of bag of words features (topical words in an essay)
                f_row_sum=numpy.sum(bag_feats[:,:])
                #Average index of how "topical" essays are
                self._mean_f_prop=f_row_sum/float(sum([len(t) for t in e_set._text]))
                ret = "ok"
            else:
                raise util_functions.InputError(e_set, "needs to be an essay set of the train type.")
        else:
            raise util_functions.InputError(e_set, "wrong input. need an essay set object")
        return ret

    def get_good_pos_ngrams(self):
        """
        Gets a set of gramatically correct part of speech sequences from an input file called essaycorpus.txt
        Returns the set and caches the file
        """
        if(os.path.isfile(NGRAM_PATH)):
            good_pos_ngrams = pickle.load(open(NGRAM_PATH, 'rb'))
        elif os.path.isfile(ESSAY_CORPUS_PATH):
            essay_corpus = open(ESSAY_CORPUS_PATH).read()
            essay_corpus = util_functions.sub_chars(essay_corpus)
            good_pos_ngrams = util_functions.regenerate_good_tokens(essay_corpus)
            pickle.dump(good_pos_ngrams, open(NGRAM_PATH, 'wb'))
        else:
            #Hard coded list in case the needed files cannot be found
            good_pos_ngrams=['NN PRP', 'NN PRP .', 'NN PRP . DT', 'PRP .', 'PRP . DT', 'PRP . DT NNP', '. DT',
             '. DT NNP', '. DT NNP NNP', 'DT NNP', 'DT NNP NNP', 'DT NNP NNP NNP', 'NNP NNP',
             'NNP NNP NNP', 'NNP NNP NNP NNP', 'NNP NNP NNP .', 'NNP NNP .', 'NNP NNP . TO',
             'NNP .', 'NNP . TO', 'NNP . TO NNP', '. TO', '. TO NNP', '. TO NNP NNP',
             'TO NNP', 'TO NNP NNP']

        return set(good_pos_ngrams)

    def _get_grammar_errors(self,pos,text,tokens):
        """
        Internal function to get the number of grammar errors in given text
        pos - part of speech tagged text (list)
        text - normal text (list)
        tokens - list of lists of tokenized text
        """
        word_counts = [max(len(t),1) for t in tokens]
        good_pos_tags = []
        min_pos_seq=2
        max_pos_seq=4
        bad_pos_positions=[]
        for i in xrange(0, len(text)):
            pos_seq = [tag[1] for tag in pos[i]]
            pos_ngrams = util_functions.ngrams(pos_seq, min_pos_seq, max_pos_seq)
            long_pos_ngrams=[z for z in pos_ngrams if z.count(' ')==(max_pos_seq-1)]
            bad_pos_tuples=[[z,z+max_pos_seq] for z in xrange(0,len(long_pos_ngrams)) if long_pos_ngrams[z] not in self._good_pos_ngrams]
            bad_pos_tuples.sort(key=operator.itemgetter(1))
            to_delete=[]
            for m in reversed(xrange(len(bad_pos_tuples)-1)):
                start, end = bad_pos_tuples[m]
                for j in xrange(m+1, len(bad_pos_tuples)):
                    lstart, lend = bad_pos_tuples[j]
                    if lstart >= start and lstart <= end:
                        bad_pos_tuples[m][1]=bad_pos_tuples[j][1]
                        to_delete.append(j)

            fixed_bad_pos_tuples=[bad_pos_tuples[z] for z in xrange(0,len(bad_pos_tuples)) if z not in to_delete]
            bad_pos_positions.append(fixed_bad_pos_tuples)
            overlap_ngrams = [z for z in pos_ngrams if z in self._good_pos_ngrams]
            if (len(pos_ngrams)-len(overlap_ngrams))>0:
                divisor=len(pos_ngrams)/len(pos_seq)
            else:
                divisor=1
            if divisor == 0:
                divisor=1
            good_grammar_ratio = (len(pos_ngrams)-len(overlap_ngrams))/divisor
            good_pos_tags.append(good_grammar_ratio)
        return good_pos_tags,bad_pos_positions

    def gen_length_feats(self, e_set):
        """
        Generates length based features from an essay set
        Generally an internal function called by gen_feats
        Returns an array of length features
        e_set - EssaySet object
        """
        text = e_set._text
        lengths = [len(e) for e in text]
        word_counts = [max(len(t),1) for t in e_set._tokens]
        comma_count = [e.count(",") for e in text]
        ap_count = [e.count("'") for e in text]
        punc_count = [e.count(".") + e.count("?") + e.count("!") for e in text]
        chars_per_word = [lengths[m] / float(word_counts[m]) for m in xrange(0, len(text))]

        good_pos_tags,bad_pos_positions= self._get_grammar_errors(e_set._pos,e_set._text,e_set._tokens)
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
        e_set - EssaySet object
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
        e_set - EssaySet object
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
        e_set - EssaySet object
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
        """
        Generate feedback for a given set of essays
        e_set - EssaySet object
        features - optionally, pass in a matrix of features extracted from e_set using FeatureExtractor
        in order to get off topic feedback.
        Returns a list of lists (one list per essay in e_set)
        e_set - EssaySet object
        """

        #Set ratio to modify thresholds for grammar/spelling errors
        modifier_ratio=1.05

        #Calc number of grammar and spelling errors per character
        set_grammar,bad_pos_positions=self._get_grammar_errors(e_set._pos,e_set._text,e_set._tokens)
        set_grammar_per_character=[set_grammar[m]/float(len(e_set._text[m])+.1) for m in xrange(0,len(e_set._text))]
        set_spell_errors_per_character=[e_set._spelling_errors[m]/float(len(e_set._text[m])+.1) for m in xrange(0,len(e_set._text))]

        #Iterate through essays and create a feedback dict for each
        all_feedback=[]
        for m in xrange(0,len(e_set._text)):
            #Be very careful about changing these messages!
            individual_feedback={'grammar' : "Grammar: Ok.",
                                 'spelling' : "Spelling: Ok.",
                                 'markup_text' : "",
                                 'grammar_per_char' : set_grammar_per_character[m],
                                 'spelling_per_char' : set_spell_errors_per_character[m],
                                 'too_similar_to_prompt' : False,
                                 }
            markup_tokens=e_set._markup_text[m].split(" ")

            #This loop ensures that sequences of bad grammar get put together into one sequence instead of staying
            #disjointed
            bad_pos_starts=[z[0] for z in bad_pos_positions[m]]
            bad_pos_ends=[z[1]-1 for z in bad_pos_positions[m]]
            for z in xrange(0,len(markup_tokens)):
                if z in bad_pos_starts:
                    markup_tokens[z]='<bg>' + markup_tokens[z]
                elif z in bad_pos_ends:
                    markup_tokens[z]=markup_tokens[z] + "</bg>"
            if(len(bad_pos_ends)>0 and len(bad_pos_starts)>0 and len(markup_tokens)>1):
                if max(bad_pos_ends)>(len(markup_tokens)-1) and max(bad_pos_starts)<(len(markup_tokens)-1):
                    markup_tokens[len(markup_tokens)-1]+="</bg>"

            #Display messages if grammar/spelling errors greater than average in training set
            if set_grammar_per_character[m]>(self._grammar_errors_per_character*modifier_ratio):
                individual_feedback['grammar']="Grammar: More grammar errors than average."
            if set_spell_errors_per_character[m]>(self._spell_errors_per_character*modifier_ratio):
                individual_feedback['spelling']="Spelling: More spelling errors than average."

            #Test topicality by calculating # of on topic words per character and comparing to the training set
            #mean.  Requires features to be passed in
            if features is not None:
                f_row_sum=numpy.sum(features[m,12:])
                f_row_prop=f_row_sum/len(e_set._text[m])
                if f_row_prop<(self._mean_f_prop/1.5) or len(e_set._text[m])<20:
                    individual_feedback['topicality']="Topicality: Essay may be off topic."

                if(features[m,9]>.6):
                    individual_feedback['prompt_overlap']="Prompt Overlap: Too much overlap with prompt."
                    individual_feedback['too_similar_to_prompt']=True
                    log.debug(features[m,9])

            #Create string representation of markup text
            markup_string=" ".join(markup_tokens)
            individual_feedback['markup_text']=markup_string
            all_feedback.append(individual_feedback)

        return all_feedback
