"""
Extracts features from training set and test set essays
"""

import numpy
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from itertools import chain
import operator
import logging

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
from essay_set import EssaySet
import util_functions

if not base_path.endswith("/"):
    base_path = base_path + "/"

log = logging.getLogger(__name__)

# Paths to needed data files
NGRAM_PATH = base_path + "data/good_pos_ngrams.p"
ESSAY_CORPUS_PATH = util_functions.ESSAY_CORPUS_PATH


class FeatureExtractor(object):
    """
    An object which serves as a feature extractor, using NLTK and some statistics to derive an object which will extract
    features from an object which will allow object classification.
    """

    def __init__(self, essay_set, max_features_pass_2=200):
        """
        Initializes requisite dictionaries/statistics before the feature extraction can occur.

        Was originally separated between an __init__ and an instantiate_dictionaries method, but they were never
        called separately, so I combined them, as the logical similarity is striking.

        Args:
            essay_set: an input set of essays that the feature extractor extracts from and is based upon

        Kwargs:
            max_features_pass_2: The maximum number of features we consider on the second pass of vocabulary grooming

        """
        if hasattr(essay_set, '_type'):
            if essay_set._type == "train":
                # Finds vocabulary which differentiates good/high scoring essays from bad/low scoring essays.
                normal_vocab = util_functions.get_vocab(
                    essay_set._cleaned_spelled_essays, essay_set._scores, max_features_pass_2=max_features_pass_2
                )

                # Finds vocab (same criteria as above), but with essays that have been porter stemmed
                stemmed_vocab = util_functions.get_vocab(
                    essay_set._cleaned_stemmed_essays, essay_set._scores, max_features_pass_2=max_features_pass_2
                )

                # Constructs dictionaries trained based on the important vocabularies
                self._normal_dict = CountVectorizer(ngram_range=(1, 2), vocabulary=normal_vocab)
                self._stem_dict = CountVectorizer(ngram_range=(1, 2), vocabulary=stemmed_vocab)

                # Sets the flag to show that this instance is now ready for training
                self.dict_initialized = True

                # Average the number of spelling errors in the set. This is needed later for spelling detection.
                spelling_errors = essay_set._spelling_errors
                self._mean_spelling_errors = sum(spelling_errors) / float(len(spelling_errors))
                self._spell_errors_per_character = sum(spelling_errors) / float(
                    sum([len(essay) for essay in essay_set._cleaned_essays]))

                # Gets the number and positions of grammar errors
                good_pos_tags, bad_pos_positions = self._get_grammar_errors(
                    essay_set._pos_tags, essay_set._cleaned_essays, essay_set._tokens
                )
                # NOTE!!! Here, I changed the definition from utilizing good grammar ratios to using the counts of
                # grammatical errors.  Though this was not what the original author used, it is clearly what his code
                # implies, as if this is intended to be a true "grammar errors per character", we should have that
                # exact number.  The replaced call is included for posterity.
                # self._grammar_errors_per_character =
                # (sum(good_pos_tags) / float(sum([len(t) for t in essay_set._text])))
                total_grammar_errors = sum(len(l) for l in bad_pos_positions)
                total_characters = float(sum([len(t) for t in essay_set._cleaned_essays]))
                self._grammar_errors_per_character = total_grammar_errors / total_characters

                # Generates a bag of vocabulary features
                vocabulary_features = self._generate_vocabulary_features(essay_set)

                # Sum of a row of bag of words features (topical words in an essay)
                feature_row_sum = numpy.sum(vocabulary_features[:, :])

                # Average index of how "topical" essays are
                self._mean_topical_index = feature_row_sum / float(sum([len(t) for t in essay_set._cleaned_essays]))
            else:
                raise util_functions.InputError(essay_set, "needs to be an essay set of the train type.")
        else:
            raise util_functions.InputError(essay_set, "wrong input. need an essay set object.")

        self._good_pos_ngrams = self._get_good_pos_ngrams()
        self._spell_errors_per_character = 0
        self._grammar_errors_per_character = 0

    def generate_features(self, essay_set):
        """
        Generates bag of words, length, and prompt features from an essay set object

        Args:
            essay_set (EssaySet): the essay set to extract features for

        Returns:
            Array of features with the following included:
                - Length Features
                - Vocabulary Features (both Normal and Stemmed Vocabulary)
                - Prompt Features
        """
        vocabulary_features = self._generate_vocabulary_features(essay_set)
        length_features = self._generate_length_features(essay_set)
        prompt_features = self._generate_prompt_features(essay_set)

        # Lumps them all together, copies to solidify, and returns
        overall_features = numpy.concatenate((length_features, prompt_features, vocabulary_features), axis=1)
        overall_features = overall_features.copy()
        return overall_features

    def _generate_length_features(self, essay_set):
        """
        Generates length based features from an essay set

        An exclusively internal function, called by generate_features

        Args:
            essay_set (EssaySet): the essay set to extract length features from

        Returns:
            An array of features that have been extracted based on length
        """
        essays = essay_set._cleaned_essays
        lengths = [len(e) for e in essays]
        word_counts = [max(len(t), 1) for t in essay_set._tokens]
        comma_count = [e.count(",") for e in essays]
        apostrophe_count = [e.count("'") for e in essays]
        punctuation_count = [e.count(".") + e.count("?") + e.count("!") for e in essays]
        chars_per_word = [lengths[m] / float(word_counts[m]) for m in xrange(0, len(essays))]

        # SEE COMMENT AROUND LINE 85
        good_grammar_ratios, bad_pos_positions = self._get_grammar_errors(essay_set._pos_tags,
                                                                          essay_set._cleaned_essays, essay_set._tokens)
        good_pos_tag_proportion = [len(bad_pos_positions[m]) / float(word_counts[m]) for m in xrange(0, len(essays))]

        length_array = numpy.array((
            lengths, word_counts, comma_count, apostrophe_count, punctuation_count, chars_per_word, good_grammar_ratios,
            good_pos_tag_proportion)).transpose()

        return length_array.copy()

    def _generate_vocabulary_features(self, essay_set):
        """
        Generates a bag of words features from an essay set and a trained FeatureExtractor (self)

        Args:
            self: The trained Feature Extractor (trained by the init_method)
            essay_set: the EssaySet Object to generate the bag of word features from.

        Returns:
            An array of features to be used for extraction
        """
        # Calculates Stem and Normal features
        stem_features = self._stem_dict.transform(essay_set._cleaned_stemmed_essays)
        normal_features = self._normal_dict.transform(essay_set._cleaned_essays)

        # Mushes them together and returns
        bag_features = numpy.concatenate((stem_features.toarray(), normal_features.toarray()), axis=1)
        return bag_features.copy()

    def _generate_prompt_features(self, essay_set):
        """
        Generates prompt based features from an essay set object and internal prompt variable.

        Called internally by generate_features

        Args:
            essay_set (EssaySet): an essay set object that is manipulated to generate prompt features

        Returns:
            an array of prompt features
        """
        prompt_toks = nltk.word_tokenize(essay_set._prompt)
        expand_syns = []
        for word in prompt_toks:
            synonyms = util_functions.get_wordnet_syns(word)
            expand_syns.append(synonyms)
        expand_syns = list(chain.from_iterable(expand_syns))
        prompt_overlap = []
        prompt_overlap_prop = []
        for j in essay_set._tokens:
            tok_length = len(j)
            if (tok_length == 0):
                tok_length = 1
            prompt_overlap.append(len([i for i in j if i in prompt_toks]))
            prompt_overlap_prop.append(prompt_overlap[len(prompt_overlap) - 1] / float(tok_length))
        expand_overlap = []
        expand_overlap_prop = []
        for j in essay_set._tokens:
            tok_length = len(j)
            if (tok_length == 0):
                tok_length = 1
            expand_overlap.append(len([i for i in j if i in expand_syns]))
            expand_overlap_prop.append(expand_overlap[len(expand_overlap) - 1] / float(tok_length))

        prompt_arr = numpy.array((prompt_overlap, prompt_overlap_prop, expand_overlap, expand_overlap_prop)).transpose()

        return prompt_arr.copy()

    def _get_grammar_errors(self, pos, essays, tokens):
        """
        Internal function to get the number of grammar errors in given text

        Args:
            pos: list of pos values for an essay set
            essays: list of essay texts
            tokens: list of the lists of the tokens in each essay

        Returns:
            Tuple of the form (good_grammar_ratios, bad_pos_positions)
                The former is a list of each essay's "good grammar ratio", which is not very well defined
                The latter is a list of lists of each essay's grammatical mistakes as a location in its tokens
        """
        good_grammar_ratios = []
        min_pos_seq = 2
        max_pos_seq = 4
        bad_pos_positions = []
        for i in xrange(0, len(essays)):
            pos_seq = [tag[1] for tag in pos[i]]
            pos_ngrams = util_functions.ngrams(pos_seq, min_pos_seq, max_pos_seq)
            long_pos_ngrams = [z for z in pos_ngrams if z.count(' ') == (max_pos_seq - 1)]
            bad_pos_tuples = [[z, z + max_pos_seq] for z in xrange(0, len(long_pos_ngrams)) if
                              long_pos_ngrams[z] not in self._good_pos_ngrams]
            bad_pos_tuples.sort(key=operator.itemgetter(1))
            to_delete = []
            for m in reversed(xrange(len(bad_pos_tuples) - 1)):
                start, end = bad_pos_tuples[m]
                for j in xrange(m + 1, len(bad_pos_tuples)):
                    lstart, lend = bad_pos_tuples[j]
                    if lstart >= start and lstart <= end:
                        bad_pos_tuples[m][1] = bad_pos_tuples[j][1]
                        to_delete.append(j)

            fixed_bad_pos_tuples = [bad_pos_tuples[z] for z in xrange(0, len(bad_pos_tuples)) if z not in to_delete]
            bad_pos_positions.append(fixed_bad_pos_tuples)
            overlap_ngrams = [z for z in pos_ngrams if z in self._good_pos_ngrams]
            if (len(pos_ngrams) - len(overlap_ngrams)) > 0:
                divisor = len(pos_ngrams) / len(pos_seq)
            else:
                divisor = 1
            if divisor == 0:
                divisor = 1
            good_grammar_ratio = (len(pos_ngrams) - len(overlap_ngrams)) / divisor
            good_grammar_ratios.append(good_grammar_ratio)
        return good_grammar_ratios, bad_pos_positions

    def _get_good_pos_ngrams(self):
        """
        Gets a list of grammatically correct part of speech sequences from an input file called essaycorpus.txt
        Returns the list and caches the file

        Returns:
            A list of all grammatically correct parts of speech.
        """
        if os.path.isfile(NGRAM_PATH):
            good_pos_ngrams = pickle.load(open(NGRAM_PATH, 'rb'))
        else:
            # Hard coded an incomplete list in case the needed files cannot be found
            good_pos_ngrams = ['NN PRP', 'NN PRP .', 'NN PRP . DT', 'PRP .', 'PRP . DT', 'PRP . DT NNP', '. DT',
                               '. DT NNP', '. DT NNP NNP', 'DT NNP', 'DT NNP NNP', 'DT NNP NNP NNP', 'NNP NNP',
                               'NNP NNP NNP', 'NNP NNP NNP NNP', 'NNP NNP NNP .', 'NNP NNP .', 'NNP NNP . TO',
                               'NNP .', 'NNP . TO', 'NNP . TO NNP', '. TO', '. TO NNP', '. TO NNP NNP',
                               'TO NNP', 'TO NNP NNP']
        return good_pos_ngrams
