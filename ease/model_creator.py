# Provides interface functions to create and save models

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
import logging
import create

log = logging.getLogger()


def read_in_test_data(filename):
    """
    Reads in tab delimited test data file found at filename for training purposes.

    filename must be a tab delimited file with columns id, dummy number column, score, dummy score, text

    Args:
        filename (str): The path to the data

    Return:
        Tuple of the form (score, text), where:
            The former is the list of scores assigned to the essays in the file (int)
            The latter is the list of essays in the file

    """
    tid, e_set, score, score2, text = [], [], [], [], []
    combined_raw = open(filename).read()
    raw_lines = combined_raw.splitlines()
    for row in xrange(1, len(raw_lines)):
        tid1, set1, score1, score12, text1 = raw_lines[row].strip().split("\t")
        tid.append(int(tid1))
        text.append(text1)
        e_set.append(int(set1))
        score.append(int(score1))
        score2.append(int(score12))

    return score, text


def read_in_test_prompt(filename):
    """
    Reads in the prompt from a file.

    Args:
        filename (str): the name of the file

    Returns:
        (str): the prompt as a string.
    """
    prompt_string = open(filename).read()
    return prompt_string


def read_in_test_data_twocolumn(filename, sep=","):
    """
    Reads in a two column version of the test data.

    In filename, the first column should be integer score data.
    The second column should be string text data.
    Sep specifies the type of separator between fields.

    Return:
        Tuple of the form (score, text), where:
            The former is the list of scores assigned to the essays in the file (int)
            The latter is the list of essays in the file

    """
    score, text = [], []
    combined_raw = open(filename).read()
    raw_lines = combined_raw.splitlines()
    for row in xrange(1, len(raw_lines)):
        score1, text1 = raw_lines[row].strip().split("\t")
        text.append(text1)
        score.append(int(score1))

    return score, text


def create_essay_set(text, score, prompt_string, generate_additional=True):
    """
    Creates an essay set from given data.
    Text should be a list of strings corresponding to essay text.
    Score should be a list of scores where score[n] corresponds to text[n]
    Prompt string is just a string containing the essay prompt.
    Generate_additional indicates whether to generate additional essays at the minimum score point or not.
    """
    essay_set = EssaySet()
    for i in xrange(0, len(text)):
        essay_set.add_essay(text[i], score[i])
        if score[i] == min(score) and generate_additional == True:
            essay_set.generate_additional_essays(essay_set._cleaned_spelled_essays[len(essay_set._cleaned_spelled_essays) - 1], score[i])

    essay_set.update_prompt(prompt_string)

    return essay_set


def get_cv_error(clf, feats, scores):
    """
    Gets cross validated error for a given classifier, set of features, and scores
    clf - classifier
    feats - features to feed into the classified and cross validate over
    scores - scores associated with the features -- feature row 1 associates with score 1, etc.
    """
    results = {'success': False, 'kappa': 0, 'mae': 0}
    try:
        cv_preds = util_functions.gen_cv_preds(clf, feats, scores)
        err = numpy.mean(numpy.abs(numpy.array(cv_preds) - scores))
        kappa = util_functions.quadratic_weighted_kappa(list(cv_preds), scores)
        results['mae'] = err
        results['kappa'] = kappa
        results['success'] = True
    except ValueError as ex:
        # If this is hit, everything is fine.  It is hard to explain why the error occurs, but it isn't a big deal.
        msg = u"Not enough classes (0,1,etc) in each cross validation fold: {ex}".format(ex=ex)
        log.debug(msg)
    except:
        log.exception("Error getting cv error estimates.")

    return results


def get_algorithms(algorithm):
    """
    Gets two classifiers for each type of algorithm, and returns them.  First for predicting, second for cv error.
    type - one of util_functions.AlgorithmTypes
    """
    if algorithm == util_functions.AlgorithmTypes.classification:
        clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learn_rate=.05,
                                                          max_depth=4, random_state=1, min_samples_leaf=3)
        clf2 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learn_rate=.05,
                                                           max_depth=4, random_state=1, min_samples_leaf=3)
    else:
        clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learn_rate=.05,
                                                         max_depth=4, random_state=1, min_samples_leaf=3)
        clf2 = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learn_rate=.05,
                                                          max_depth=4, random_state=1, min_samples_leaf=3)
    return clf, clf2

def extract_features_and_generate_model_from_predictors(predictor_set, algorithm=util_functions.AlgorithmTypes.regression):
    """
    Extracts features and generates predictors based on a given predictor set
    predictor_set - a PredictorSet object that has been initialized with data
    type - one of util_functions.AlgorithmType
    """
    if (algorithm not in [util_functions.AlgorithmTypes.regression, util_functions.AlgorithmTypes.classification]):
        algorithm = util_functions.AlgorithmTypes.regression

    f = predictor_extractor.PredictorExtractor(predictor_set)

    train_feats = f.generate_features(predictor_set)

    clf, clf2 = get_algorithms(algorithm)
    cv_error_results = get_cv_error(clf2, train_feats, predictor_set._target)

    try:
        set_score = numpy.asarray(predictor_set._target, dtype=numpy.int)
        clf.fit(train_feats, set_score)
    except ValueError:
        log.exception("Not enough classes (0,1,etc) in sample.")
        set_score = predictor_set._target
        set_score[0] = 1
        set_score[1] = 0
        clf.fit(train_feats, set_score)

    return f, clf, cv_error_results


def extract_features_and_generate_model(essay_set):
    """
    Feed in an essay set to get feature vector and classifier

    Args:
        essays (EssaySet): The essay set to construct the feature extractor and model off of

    Returns:
        A tuple with the following elements in the following order:
            - The Trained Feature extractor
            - The Trained Classifier
            - Any Cross Validation results
    """
    feat_extractor = feature_extractor.FeatureExtractor(essay_set)

    features = feat_extractor.generate_features(essay_set)

    set_score = numpy.asarray(essay_set._score, dtype=numpy.int)
    algorithm = create.select_algorithm(set_score)

    predict_classifier, cv_error_classifier = get_algorithms(algorithm)

    cv_error_results = get_cv_error(cv_error_classifier, features, essay_set._score)

    try:
        predict_classifier.fit(features, set_score)
    except:
        log.exception("Not enough classes (0,1,etc) in sample.")
        set_score[0] = 1
        set_score[1] = 0
        predict_classifier.fit(features, set_score)

    return feat_extractor, predict_classifier, cv_error_results


def dump_model_to_file(prompt_string, feature_ext, classifier, text, score, model_path):
    """
    Writes out a model to a file.

    Args:
        prompt_string (str): The prompt for the set of essays
        feature_ext (FeatureExtractor): a trained FeatureExtractor Object
        classifier : a trained Classifier Object
    prompt string is a string containing the prompt
    feature_ext is a trained FeatureExtractor object
    classifier is a trained classifier
    model_path is the path of write out the model file to
    """
    model_file = {'prompt': prompt_string, 'extractor': feature_ext, 'model': classifier, 'text': text, 'score': score}
    pickle.dump(model_file, file=open(model_path, "w"))


def create_essay_set_and_dump_model(text, score, prompt, model_path, additional_array=None):
    """
    Function that creates essay set, extracts features, and writes out model
    See above functions for argument descriptions
    """
    essay_set = create_essay_set(text, score, prompt)
    feature_ext, clf = extract_features_and_generate_model(essay_set, additional_array)
    dump_model_to_file(prompt, feature_ext, clf, model_path)


