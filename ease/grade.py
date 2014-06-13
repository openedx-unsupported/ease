"""
Functions to score specified data using specified ML models
"""

import os
import logging

import sys


# Append sys to base path to import the following modules
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

#Depend on base path to be imported
from essay_set import EssaySet
from errors import *

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
    """

    # Initialize result dictionary
    results = {'errors': [], 'tests': [], 'score': 0, 'feedback': "", 'success': False, 'confidence': 0}

    # Instantiates the Essay set which will carry our essay while it is being classified and graded.
    grader_set = EssaySet(essay_type="test")
    feedback = {}

    # Retrieves the model and extractor we will be using
    model, extractor = _get_classifier_and_extractor(grader_data)

    # Attempts to add the essay (text) to the essay set.
    try:
        grader_set.add_essay(str(submission), 0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except (EssaySetRequestError, InputError):
        error_message = "Essay could not be added to essay set:{0}".format(submission)
        log.exception(error_message)
        results['errors'].append(error_message)

    # Tries to extract features from submission and assign score via the model
    try:
        grader_features = extractor.generate_features(grader_set)
        results['score'] = int(model.predict(grader_features)[0])
    except FeatureExtractionInternalError as ex:
        error_message = "Could not extract features and score essay: {ex}".format(ex=ex)
        log.exception(error_message)
        results['errors'].append(error_message)

    # We have gotten through without an error, so we have been successful
    if len(results['errors']) < 0:
        results['success'] = True
    # If we get here, that means there was 1+ error above. Set success to false and return
    else:
        results['success'] = False

    return results


def _get_classifier_and_extractor(grader_data):
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


