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

#Append to path and then import things that depend on path
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
from essay_set import EssaySet
import util_functions

if not base_path.endswith("/"):
    base_path=base_path+"/"

log = logging.getLogger(__name__)

class PredictorExtractor(object):
    def __init__(self):
        self._extractors = []
        self._initialized = False

    def initialize_dictionaries(self, p_set):
        """
        Initialize dictionaries with the textual inputs in the PredictorSet object
        p_set - PredictorSet object that has had data fed in
        """
        success = False
        if not (hasattr(p_set, '_type')):
            error_message = "needs to be an essay set of the train type."
            log.exception(error_message)
            raise util_functions.InputError(p_set, error_message)

        if not (p_set._type == "train"):
            error_message = "needs to be an essay set of the train type."
            log.exception(error_message)
            raise util_functions.InputError(p_set, error_message)

        div_length=len(p_set._essay_sets)
        if div_length==0:
            div_length=1

        #Ensures that even with a large amount of input textual features, training time stays reasonable
        max_feats2 = int(math.floor(200/div_length))
        for i in xrange(0,len(p_set._essay_sets)):
            self._extractors.append(FeatureExtractor())
            self._extractors[i].initialize_dictionaries(p_set._essay_sets[i], max_feats2=max_feats2)
            self._initialized = True
            success = True
        return success

    def gen_feats(self, p_set):
        """
        Generates features based on an iput p_set
        p_set - PredictorSet
        """
        if self._initialized!=True:
            error_message = "Dictionaries have not been initialized."
            log.exception(error_message)
            raise util_functions.InputError(p_set, error_message)

        textual_features = []
        for i in xrange(0,len(p_set._essay_sets)):
            textual_features.append(self._extractors[i].gen_feats(p_set._essay_sets[i]))

        textual_matrix = numpy.concatenate(textual_features, axis=1)
        predictor_matrix = numpy.array(p_set._numeric_features)

        print textual_matrix.shape
        print predictor_matrix.shape

        overall_matrix = numpy.concatenate((textual_matrix, predictor_matrix), axis=1)

        return overall_matrix.copy()
