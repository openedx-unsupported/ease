import numpy
import nltk
import sys
import random
import os
import logging

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions

if not base_path.endswith("/"):
    base_path=base_path+"/"

log=logging.getLogger(__name__)


class AlgorithmTypes(object):
    regression = "regression"
    classification = "classifiction"


class PredictorSet(object):
    def __init__(self, type="train", algorithm = AlgorithmTypes.regression):
        """
        Initialize variables and check essay set type
        """
        if(type != "train" and type != "test"):
            type = "train"

        if(algorithm not in [AlgorithmTypes.regression, AlgorithmTypes.classification]):
            algorithm = AlgorithmTypes.regression

        self._type = type
        self._target=[]
        self._textual_features=[]
        self._numeric_features=[]
        self.essay_sets=[]

    def add_row(self, numeric_features, textual_features, target):
        pass
