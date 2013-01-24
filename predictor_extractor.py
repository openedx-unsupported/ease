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


class PredictorExtractor(object):
    def __init__(self):
        pass

    
