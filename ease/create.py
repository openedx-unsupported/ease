"""
Functions that create a machine learning model from training data
"""

import os
import sys
import logging
import numpy

# Define base path and add to sys path
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//'))
sys.path.append(one_up_path)

#Import modules that are dependent on the base path
import model_creator
import util_functions
from errors import *
from datetime import datetime
import json

#Make a log
log = logging.getLogger(__name__)


def dump_input_data(text, score):
    try:
        file_path = base_path + "/tests/data/json_data/"
        time_suffix = datetime.now().strftime("%H%M%S%d%m%Y")
        prefix = "test-case-"
        filename = prefix + time_suffix + ".json"
        json_data = []
        for i in xrange(0, len(text)):
            json_data.append({'text': text[i], 'score': score[i]})
        with open(file_path + filename, 'w+') as outfile:
            json.dump(json_data, outfile)
    except:
        error = "Could not dump data to file."
        log.exception(error)


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
        dump_input_data(examples, scores)

    # Selects the appropriate ML algorithm to use to train the classifier
    algorithm = select_algorithm(scores)

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
        essay_set = model_creator.create_essay_set(examples, scores, prompt_string)
    except (ExampleCreationRequestError, ExampleCreationInternalError) as ex:
        msg = "essay set creation failed due to an error in the create_essay_set method. {}".format(ex)
        results['errors'].append(msg)
        log.exception(msg)
        return results

    # Gets the features and classifiers from the essay set and computes the error
    try:
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model(
            essay_set
        )
        results['cv_kappa'] = cv_error_results['kappa']
        results['cv_mean_absolute_error'] = cv_error_results['mae']
        results['feature_ext'] = feature_ext
        results['classifier'] = classifier
        results['algorithm'] = algorithm
        results['success'] = True
    except Exception as ex:
        msg = "feature extraction and model creation failed."
        results['errors'].append(msg)
        log.exception(msg)
        log.exception(ex)

    return results


def select_algorithm(score_list):
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