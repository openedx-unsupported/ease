"""
Functions that create a machine learning model from training data
"""

import os
import sys
import logging
import numpy

# Constructs a log
log = logging.getLogger(__name__)
# Setup base path so that we can import modules who are dependent on it
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//'))
sys.path.append(one_up_path)

#Import modules that are dependent on the base path
import util_functions
from errors import *
from datetime import datetime
import json
import sklearn.ensemble
from ease import feature_extractor
from ease.essay_set import EssaySet


def create(examples, scores, prompt_string, dump_data=False):
    """
    Creates a machine learning model from basic inputs (essays, associated scores and a prompt) and trains the model.

    The previous version of this function took an additional argument which specified the path to the model.

    Args:
        examples (list of str): the example essays that have been assigned to train the AI.
        scores (list of int): the associated scores that correspond to the essays.
        prompt_string (str): the common prompt for all of the example essays.

    Kwargs:
        dump_data (bool): whether or not a examples and scores should be set via a data input dump

    Returns:
        (dict): Has the following keys:
            'errors' (list of Exception): List of all errors that occurred during training
            'cv_kappa' (float): cv_error, measured in terms of kappa.
            'cv_mean_absolute_error' (float): cv_error, measured as the mean absolute value
            'feature_ext': feature_extractor to be used for grading
            'classifier': the classifier object which can be used to score future essays
            'success' (bool): Whether or not the training of the classifier was successful.
    """

    # If dump_data is true, then the examples and scores are loaded from json data.
    if dump_data:
        _dump_input_data(examples, scores)

    # Selects the appropriate ML algorithm to use to train the classifier
    algorithm = _determine_algorithm(scores)

    #Initialize a results dictionary to return
    results = {'errors': [], 'success': False, 'cv_kappa': 0, 'cv_mean_absolute_error': 0,
               'feature_ext': "", 'classifier': "", 'algorithm': algorithm,
               'score': scores, 'text': examples, 'prompt': prompt_string}

    if len(examples) != len(scores):
        results['errors'].append("Target and text lists must be same length.")
        log.exception("Target and text lists must be same length.")
        return results

    # Create an essay set object that encapsulates all the essays and alternate representations (tokens, etc)
    try:
        essay_set = _create_essay_set(examples, scores, prompt_string)
    except (EssaySetRequestError, ExampleCreationInternalError) as ex:
        msg = "Essay Set Creation failed (likely due to an error in essay cleaning/parsing) {}".format(ex)
        results['errors'].append(msg)
        log.exception(msg)
        return results

    # Gets the features and classifiers from the essay set and computes the error
    try:
        feature_ext, classifier, cv_error_results = _extract_features_and_generate_model(
            essay_set
        )
        results['cv_kappa'] = cv_error_results['kappa']
        results['cv_mean_absolute_error'] = cv_error_results['mae']
        results['feature_ext'] = feature_ext
        results['classifier'] = classifier
        results['algorithm'] = algorithm
        results['success'] = True

    # We cannot be sure what kind of errors .fit could throw at us. Memory, Type, Interrupt, etc.
    except ClassifierTrainingInternalError as ex:
        msg = "Feature extraction and Model Creation failed with the following error {ex}".format(ex=ex)
        results['errors'].append(msg)
        log.exception(msg)
        results['success'] = False

    return results


def _determine_algorithm(score_list):
    """
    Decides whether to use regression or classification as the ML algorithm based on the number of unique scores

    If there are more than 5 unique scores give, regression is used, if fewer than 5 unique scores are produced
    then classification is used.

    Args:
        score_list (list of int): The number of scores awarded to example essays for a given question

    Return:
        The ML algorithm used to train the classifier set and feature extractor
    """

    #Count the number of unique score points in the score list
    if len(set(score_list)) > 5:
        return util_functions.AlgorithmTypes.regression
    else:
        return util_functions.AlgorithmTypes.classification


def _create_essay_set(essays, scores, prompt_string, generate_additional=True):
    """
    Constructs an essay set from a given set of data.

    Args:
        essays (list of str): A list of essays
        scores (list of int): the corresponding scores of the essays
        prompt_string (str): the common prompt for the essays

    Kwargs:
        generate_additional (bool): Whether or not to generate additional essays at the minimum score point or not.
            DEFAULT = TRUE

    Returns:
        (EssaySet): An essay set object which encapsulates all of this information.
    """

    essay_set = EssaySet()
    essay_set.update_prompt(prompt_string)

    # Adds all essays to the essay set, and generates additional essays for the bottom scoring essay if applicable
    for i in xrange(0, len(essays)):
        essay_set.add_essay(essays[i], scores[i])
        if scores[i] == min(scores) and generate_additional == True:
            essay_set.generate_additional_essays(essay_set._cleaned_spelled_essays[-1], scores[i])

    return essay_set


def _extract_features_and_generate_model(essay_set):
    """
    Feed in an essay set to get feature vector and classifier

    Args:
        essays (EssaySet): The essay set to construct the feature extractor and model off of

    Returns:
        A tuple with the following elements in the following order:
            - The Trained Feature extractor
            - The Trained Classifier
            - Any Cross Validation results

    Raises:
        ClassifierTrainingInternalError
    """
    feat_extractor = feature_extractor.FeatureExtractor(essay_set)

    features = feat_extractor.generate_features(essay_set)

    set_scores = numpy.asarray(essay_set._scores, dtype=numpy.int)
    algorithm = _determine_algorithm(set_scores)

    predict_classifier, cv_error_classifier = _instantiate_algorithms(algorithm)

    cv_error_results = _get_cv_error(cv_error_classifier, features, essay_set._scores)

    try:
        predict_classifier.fit(features, set_scores)

    # We cannot be sure what kind of errors .fit could throw at us. Memory, Type, Interrupt, etc.
    except Exception as ex:
        str = (
            "predict_classifier.fit raised an exception in _extract_features_and_generate_model: {}"
        ).format(ex)
        log.exception(str)
        raise ClassifierTrainingInternalError(str)

    return feat_extractor, predict_classifier, cv_error_results


def _instantiate_algorithms(algorithm):
    """
    Gets two classifiers for each type of algorithm, and returns them.

    The First algorithm is used for for predicting scores,
    The second is used for calculating cv error.

    Args:
        algorithm: One of the Algorithm types defined in util_functions.AlgorithmTypes

    Returns:
        A tuple of the form (classifier, classifier), where
            The First algorithm is used for for predicting scores,
            The second is used for calculating cv error.
    """
    if algorithm == util_functions.AlgorithmTypes.classification:
        clf = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=100, learn_rate=.05, max_depth=4, random_state=1, min_samples_leaf=3
        )
        clf2 = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=100, learn_rate=.05, max_depth=4, random_state=1, min_samples_leaf=3
        )
    else:
        clf = sklearn.ensemble.GradientBoostingRegressor(
            n_estimators=100, learn_rate=.05, max_depth=4, random_state=1, min_samples_leaf=3
        )
        clf2 = sklearn.ensemble.GradientBoostingRegressor(
            n_estimators=100, learn_rate=.05, max_depth=4, random_state=1, min_samples_leaf=3
        )
    return clf, clf2


def _get_cv_error(classifier, features, scores):
    """
    Gets cross validated error for a given classifier, set of features, and scores

    Args:
        classifier: The classifier to be used for CV
        features: The features to feed into the classifier and to cross validate over.
                    Stored as a list of lists. Each row in the outer list associates with a single essay
        scores: the scores associated with each of the features.  Feature row 1 associates with score 1, etc.

    Returns:
        (dict) with the following keys:
            'mae': Mean Absolute Error (measures the average deviation between AI grade and Human Grade)
            'kappa': Quadratic weighted kappa (measures the similarity between graders (AI and Human))
            'success': Whether or not the calculation was successful.
    """
    results = {'success': False, 'kappa': 0, 'mae': 0}
    try:
        cv_preds = util_functions.gen_cv_preds(classifier, features, scores)
        err = numpy.mean(numpy.abs(numpy.array(cv_preds) - scores))
        kappa = util_functions.quadratic_weighted_kappa(list(cv_preds), scores)
        results['mae'] = err
        results['kappa'] = kappa
        results['success'] = True
    except ValueError as ex:
        # If this is hit, everything is fine.  It is hard to explain why the error occurs, but it isn't a big deal.
        # TODO Figure out why this error would occur in the first place.
        msg = u"Not enough classes (0,1,etc) in each cross validation fold: {ex}".format(ex=ex)
        log.debug(msg)

    return results


def _dump_input_data(essays, scores):
    """
    Dumps input data using json serialized objects of the form {'text': essay, 'score': score}

    Args:
        essays (list of str): A list of essays to dump
        scores (list of int): An associated list of scores
    """

    file_path = base_path + "/tests/data/json_data/"
    time_suffix = datetime.now().strftime("%H%M%S%d%m%Y")
    prefix = "test-case-"
    filename = prefix + time_suffix + ".json"
    json_data = []
    try:
        for i in xrange(0, len(essays)):
            json_data.append({'text': essays[i], 'score': scores[i]})
        with open(file_path + filename, 'w+') as outfile:
            json.dump(json_data, outfile)
    except IOError as ex:
        error = "An IO error occurred while trying to dump JSON data to a file: {ex}".format(ex=ex)
        log.exception(error)
        raise CreateRequestError(error)
