#Provides interface functions to create and save models

import numpy
import re
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import sklearn.ensemble
from itertools import chain

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

from essay_set import EssaySet
import util_functions
import feature_extractor

def read_in_test_data(filename):
    """
    Reads in test data file found at filename.
    filename must be a tab delimited file with columns id, dummy number column, score, dummy score, text
    returns the score and the text
    """
    id, e_set, score, score2, text = [], [], [], [], []
    combined_raw = open(filename).read()
    raw_lines = combined_raw.splitlines()
    for row in xrange(1, len(raw_lines)):
        id1, set1, score1, score12, text1 = raw_lines[row].strip().split("\t")
        id.append(int(id1))
        text.append(text1)
        e_set.append(int(set1))
        score.append(int(score1))
        score2.append(int(score12))

    return score, text


def read_in_test_prompt(filename):
    """
    Reads in the prompt from a text file
    Returns string
    """
    prompt_string = open(filename).read()
    return prompt_string


def create_essay_set(text, score, prompt_string, generate_additional=True):
    """
    Creates an essay set from given data.
    Text should be a list of strings corresponding to essay text.
    Score should be a list of scores where score[n] corresponds to text[n]
    Prompt string is just a string containing the essay prompt.
    Generate_additional indicates whether to generate additional essays at the minimum score point or not.
    """
    x = EssaySet()
    for i in xrange(0, len(text)):
        x.add_essay(text[i], score[i])
        if score[i] == min(score) and generate_additional == True:
            x.generate_additional_essays(x._clean_text[len(x._clean_text) - 1], score[i])

    x.update_prompt(prompt_string)

    return x

def extract_features_and_generate_model(essays):
    """
    Feed in an essay set to get feature vector and classifier
    essays must be an essay set object
    returns a trained FeatureExtractor object and a trained classifier
    """
    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(essays)

    train_feats = f.gen_feats(essays)

    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learn_rate=.05,
        max_depth=4, random_state=1,
        min_samples_leaf=3)

    model = util_functions.gen_model(clf, train_feats, essays._score)

    return f, clf

def dump_model_to_file(prompt_string, feature_ext, classifier, model_path):
    """
    Writes out a model to a file.
    prompt string is a string containing the prompt
    feature_ext is a trained FeatureExtractor object
    classifier is a trained classifier
    model_path is the path of write out the model file to
    """
    model_file = {'prompt': prompt_string, 'extractor': feature_ext, 'model': classifier}
    pickle.dump(model_file, file=open(model_path, "w"))


