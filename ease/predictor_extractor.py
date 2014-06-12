"""
Extracts features for an arbitrary set of textual and numeric inputs
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
import math
from feature_extractor import FeatureExtractor

# Append to path and then import things that depend on path
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
from essay_set import EssaySet
import util_functions

if not base_path.endswith("/"):
    base_path = base_path + "/"

log = logging.getLogger(__name__)


class PredictorExtractor(object):
    """
    Provides an interface for extracting features from a predictor set (as opposed to an essay set), and uses the
    methods of the essay set feature extractor in order to maintain cohesion between the two different methods.
    """

    def __init__(self, predictor_set):
        """
        Initializes dictionaries with the textual inputs in the PredictorSet object
        Uses a predictor_set in the definition of the PredictorExtractor to train the extractor.

        Args:
            predictor_set (PredictorSet): PredictorSet object that has had data fed to it
        """

        if not (hasattr(predictor_set, '_type')):
            error_message = "needs to be an essay set of the train type."
            log.exception(error_message)
            raise util_functions.InputError(predictor_set, error_message)

        if not (predictor_set._type == "train"):
            error_message = "needs to be an essay set of the train type."
            log.exception(error_message)
            raise util_functions.InputError(predictor_set, error_message)

        div_length = len(predictor_set._essay_sets)
        if div_length == 0:
            div_length = 1

        self._extractors = []
        # Ensures that even with a large amount of input textual features, training time will stay reasonable
        max_features_pass_2 = int(math.floor(200 / div_length))
        for i in xrange(0, len(predictor_set._essay_sets)):
            self._extractors.append(FeatureExtractor(predictor_set._essay_sets[i]))
            self._initialized = True

    def generate_features(self, predictor_set):
        """
        Generates features given a predictor set containing the essays/data we want to extract from

        Args:
            predictor_set (PredictorSet): the wrapper which contains the prediction data we want to extract from

        Returns:
            an array of features

        """
        if self._initialized != True:
            error_message = "Dictionaries have not been initialized."
            log.exception(error_message)
            raise util_functions.InputError(predictor_set, error_message)

        textual_features = []
        # Generates features by using the generate_features method from the essay set class
        for i in xrange(0, len(predictor_set._essay_sets)):
            textual_features.append(
                self._extractors[i].generate_features(predictor_set._essay_sets[i])
            )

        textual_matrix = numpy.concatenate(textual_features, axis=1)
        predictor_matrix = numpy.array(predictor_set._numeric_features)

        # Originally there were two calls here to print the shape of the feature matricies.  GBW didn't think this was
        # appropriate, and deleted them.

        overall_matrix = numpy.concatenate((textual_matrix, predictor_matrix), axis=1)

        return overall_matrix.copy()
