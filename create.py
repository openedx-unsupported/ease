"""
Functions that create a machine learning model from training data
"""

import os
import sys
import logging
from statsd import statsd
import numpy

#Define base path and add to sys path
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

#Import modules that are dependent on the base path
import model_creator
import util_functions
import predictor_set
import predictor_extractor

#Make a log
log = logging.getLogger(__name__)

@statsd.timed('open_ended_assessment.machine_learning.creator.time')
def create(text,score,prompt_string):
    """
    Creates a machine learning model from input text, associated scores, a prompt, and a path to the model
    TODO: Remove model path argument, it is needed for now to support legacy code
    text - A list of strings containing the text of the essays
    score - a list of integers containing score values
    prompt_string - the common prompt for the set of essays
    """

    #Initialize a results dictionary to return
    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : "", 'algorithm' : util_functions.AlgorithmTypes.classification,
               'score' : score, 'text' : text, 'prompt' : prompt_string}

    if len(text)!=len(score):
        msg = "Target and text lists must be same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    #Decide what algorithm to use (regression or classification)
    try:
        if len(util_functions.f7(list(score)))>5:
            type = util_functions.AlgorithmTypes.regression
        else:
            type = util_functions.AlgorithmTypes.classification
    except:
        type = util_functions.AlgorithmTypes.regression

    try:
        #Create an essay set object that encapsulates all the essays and alternate representations (tokens, etc)
        e_set = model_creator.create_essay_set(text, score, prompt_string)
    except:
        msg = "essay set creation failed."
        results['errors'].append(msg)
        log.exception(msg)
    try:
        #Gets features from the essay set and computes error
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model(e_set, type=type)
        results['cv_kappa']=cv_error_results['kappa']
        results['cv_mean_absolute_error']=cv_error_results['mae']
        results['feature_ext']=feature_ext
        results['classifier']=classifier
        results['algorithm'] = type
        results['success']=True
    except:
        msg = "feature extraction and model creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    #Count number of successful/unsuccessful creations
    statsd.increment("open_ended_assessment.machine_learning.creator_count",
        tags=["success:{0}".format(results['success'])])

    return results


def create_generic(numeric_values, textual_values, target, algorithm = util_functions.AlgorithmTypes.regression):
    """
    Creates a model from a generic list numeric values and text values
    numeric_values - A list of lists that are the predictors
    textual_values - A list of lists that are the predictors
    (each item in textual_values corresponds to the similarly indexed counterpart in numeric_values)
    target - The variable that we are trying to predict.  A list of integers.
    algorithm - the type of algorithm that will be used
    """

    #Initialize a result dictionary to return.
    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : "", 'algorithm' : algorithm}

    if len(numeric_values)!=len(textual_values) or len(numeric_values)!=len(target):
        msg = "Target, numeric features, and text features must all be the same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    try:
        #Initialize a predictor set object that encapsulates all of the text and numeric predictors
        pset = predictor_set.PredictorSet(type="train")
        for i in xrange(0, len(numeric_values)):
            pset.add_row(numeric_values[i], textual_values[i], target[i])
    except:
        msg = "predictor set creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    try:
        #Extract all features and then train a classifier with the features
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model_predictors(pset, algorithm)
        results['cv_kappa']=cv_error_results['kappa']
        results['cv_mean_absolute_error']=cv_error_results['mae']
        results['feature_ext']=feature_ext
        results['classifier']=classifier
        results['success']=True
    except:
        msg = "feature extraction and model creation failed."
        results['errors'].append(msg)
        log.exception(msg)

        #Count number of successful/unsuccessful creations
    statsd.increment("open_ended_assessment.machine_learning.creator_count",
        tags=["success:{0}".format(results['success'])])

    return results