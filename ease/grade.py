"""
Functions to score specified data using specified ML models
"""

import sys
import pickle
import os
import numpy
import logging

# Append sys to base path to import the following modules
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

#Depend on base path to be imported
from essay_set import EssaySet
import predictor_extractor
import predictor_set
import util_functions
from errors import *

#Imports needed to unpickle grader data
import feature_extractor
import sklearn.ensemble
import math

log = logging.getLogger(__name__)


def grade(grader_data, submission):
    """
    Grades a submission given all of the feature extractor and classifier set.

    Args:
        grader_data (dict): Has the following keys
            'model' : trained model,
            'extractor' : trained feature extractor,
            'prompt' : prompt for the question,
            'algorithm' : algorithm for the question,
        submission (str): The student submission

    Returns:
        (dict) with the following keys:
            'errors': All of the errors that arose during the grading process.
            'tests':
            'score': The score the input essay was assigned by the classifier set
            'feedback': The feedback given by the classifier set
            'success': Whether or not the grading operation was a success
            'confidence': A metric of the classifier's confidence in its result
    """

    # Initialize result dictionary
    results = {'errors': [], 'tests': [], 'score': 0, 'feedback': "", 'success': False, 'confidence': 0}

    # Instantiates the Essay set which will carry our essay while it is being classified and graded.
    grader_set = EssaySet(essay_type="test")
    feedback = {}

    # Retrieves the model and extractor we will be using
    model, extractor = get_classifier_and_extractor(grader_data)

    # Attempts to add the essay (text) to the essay set.
    try:
        grader_set.add_essay(str(submission), 0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except:
        error_message = "Essay could not be added to essay set:{0}".format(submission)
        log.exception(error_message)
        results['errors'].append(error_message)

    # Tries to extract features from submission and assign score via the model
    grader_features = None
    try:
        grader_features = extractor.generate_features(grader_set)
        feedback = extractor.gen_feedback(grader_set, grader_features)[0]
        results['score'] = int(model.predict(grader_features)[0])
    except:
        error_message = "Could not extract features and score essay."
        log.exception(error_message)
        results['errors'].append(error_message)

    #Try to determine confidence level
    try:
        results['confidence'] = get_confidence_value(
            grader_data['algorithm'], model, grader_features, results['score'], grader_data['score'])
    except:
        # If there is an error getting confidence, it is not a show-stopper/big deal, so just log the error
        log.exception("Problem generating confidence value")

    # If we have errors above, we do not continue here, but return the dictionary of failure
    if len(results['errors']) < 0:

        # We have gotten through without an error, so we have been successful
        results['success'] = True

        # If the essay is just a copy of the prompt (or too similar), return a 0 as the score
        if 'too_similar_to_prompt' in feedback and feedback['too_similar_to_prompt']:
            results['score'] = 0

        # Generate feedback, identifying a number of explicable problem areas
        results['feedback'] = {
            'spelling': feedback['spelling'],
            'grammar': feedback['grammar'],
            'markup-text': feedback['markup_text'],
        }

        if 'topicality' in feedback and 'prompt_overlap' in feedback:
            results['feedback'].update({
                'topicality': feedback['topicality'],
                'prompt-overlap': feedback['prompt_overlap'],
            })

    # If we get here, that means there was 1+ error above. Set success to false and return
    else:
        results['success'] = False

    return results


def grade_generic(grader_data, numeric_features, textual_features):
    """
    Grades the generic case of numeric and textual features using a generic prediction model.

    grader_data (dict):  contains key (amoung others)
        'algorithm': Type of algorithm used to score
    numeric_features (list of float or int or long): A list of numeric features of the essay we are grading
    textual_features (list of string): A list of textual features of the essay we are grading

    Returns:
        (dict) with the following keys:
            'errors': All of the errors that arose during the grading process.
            'tests':
            'score': The score the input essay was assigned by the classifier set
            'success': Whether or not the grading operation was a success
            'confidence': A metric of the classifier's confidence in its result

    """

    results = {'errors': [], 'tests': [], 'score': 0, 'success': False, 'confidence': 0}

    # Create a predictor set which will carry the information as we grade it.
    grader_set = predictor_set.PredictorSet(essay_type="test")

    # Finds the appropriate predictor and model to use
    model, extractor = get_classifier_and_extractor(grader_data)

    # Try to add data to predictor set that we are going to be grading
    try:
        grader_set.add_row(numeric_features, textual_features, 0)
    except:
        error_msg = "Row could not be added to predictor set:{0} {1}".format(numeric_features, textual_features)
        log.exception(error_msg)
        results['errors'].append(error_msg)

    # Try to extract features from submission and assign score via the model
    try:
        grader_feats = extractor.generate_features(grader_set)
        results['score'] = model.predict(grader_feats)[0]
    except:
        error_msg = "Could not extract features and score essay."
        log.exception(error_msg)
        results['errors'].append(error_msg)

    # Try to determine confidence level
    try:
        results['confidence'] = get_confidence_value(grader_data['algorithm'], model, grader_feats, results['score'])
    except:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

    # If we didn't run into an error, we were successful
    if len(results['errors']) == 0:
        results['success'] = True

    return results


def get_confidence_value(algorithm, model, grader_features, score, scores):
    """
    Determines the confidence level for a specific grade given to a specific essay.

    Args:
        algorithm: one of the two from util_functions.AlgorithmTypes
        model: A trained model for classification
        grader_features: A dictionary describing the grading task
        score: The score assigned to this problem
        scores: All scores assigned to this problem for all submissions (not just this one)

    NOTE: For our current intents and purposes, this value is not utile, and will be removed later on.

    Returns:
        Ideally: A value between 0 and 1 reflecting the normalized probability confidence in the grade assigned.
        Actually: A numerical value with no weight reflecting an arbitrary degree of confidence.
    """
    min_score = min(numpy.asarray(scores))

    # If our algorithm is classification:
    if algorithm == util_functions.AlgorithmTypes.classification and hasattr(model, "predict_proba"):
        # If classification, predict with probability, which gives you a matrix of confidences per score point
        raw_confidence = model.predict_proba(grader_features)[0, (float(score) - float(min_score))]
        # The intent was to normalize confidence here, but it was never done, so it remains as such.
        confidence = raw_confidence

    # Otherwise, if our algorithm is prediction
    elif hasattr(model, "predict"):
        raw_confidence = model.predict(grader_features)[0]
        confidence = max(float(raw_confidence) - math.floor(float(raw_confidence)),
                         math.ceil(float(raw_confidence)) - float(raw_confidence))

    # Otherwise, we have no confidence, because we have no grading mechanism
    else:
        confidence = 0

    return confidence


def get_classifier_and_extractor(grader_data):
    """
    Finds the classifier and extractor from a completed training operation in order to perform the grading operation.

    Args:
        grader_data (dict):  has the following keys, all self evident.
            'classifier', 'model', 'feature_ext', 'extractor'

    Returns:
        A tuple of the form (model, extractor) which has those elements
    """
    if 'classifier' in grader_data:
        model = grader_data['classifier']
    elif 'model' in grader_data:
        model = grader_data['model']
    else:
        raise GradingRequestError("Cannot find a valid model.")

    if 'feature_ext' in grader_data:
        extractor = grader_data['feature_ext']
    elif 'extractor' in grader_data:
        extractor = grader_data['extractor']
    else:
        raise GradingRequestError("Cannot find the extractor")

    return model, extractor


