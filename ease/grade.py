"""
Functions to score specified data using specified ML models
"""

import sys
import pickle
import os
import numpy
import logging

#Append sys to base path to import the following modules
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

#Depend on base path to be imported
from essay_set import EssaySet
import predictor_extractor
import predictor_set
import util_functions

#Imports needed to unpickle grader data
import feature_extractor
import sklearn.ensemble
import math

log = logging.getLogger(__name__)

def grade(grader_data,submission):
    """
    Grades a specified submission using specified models
    grader_data - A dictionary:
    {
        'model' : trained model,
        'extractor' : trained feature extractor,
        'prompt' : prompt for the question,
        'algorithm' : algorithm for the question,
    }
    submission - The student submission (string)
    """

    #Initialize result dictionary
    results = {'errors': [],'tests': [],'score': 0, 'feedback' : "", 'success' : False, 'confidence' : 0}
    has_error=False

    grader_set=EssaySet(essaytype="test")
    feedback = {}

    model, extractor = get_classifier_and_ext(grader_data)

    #This is to preserve legacy functionality
    if 'algorithm' not in grader_data:
        grader_data['algorithm'] = util_functions.AlgorithmTypes.classification

    try:
        #Try to add essay to essay set object
        grader_set.add_essay(str(submission),0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except Exception:
        error_message = "Essay could not be added to essay set:{0}".format(submission)
        log.exception(error_message)
        results['errors'].append(error_message)
        has_error=True

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=extractor.gen_feats(grader_set)
        feedback=extractor.gen_feedback(grader_set,grader_feats)[0]
        results['score']=int(model.predict(grader_feats)[0])
    except Exception:
        error_message = "Could not extract features and score essay."
        log.exception(error_message)
        results['errors'].append(error_message)
        has_error=True

    #Try to determine confidence level
    try:
        results['confidence'] = get_confidence_value(grader_data['algorithm'], model, grader_feats, results['score'], grader_data['score'])
    except Exception:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

    if not has_error:

        #If the essay is just a copy of the prompt, return a 0 as the score
        if( 'too_similar_to_prompt' in feedback and feedback['too_similar_to_prompt']):
            results['score']=0
            results['correct']=False

        results['success']=True

        #Generate short form output--number of problem areas identified in feedback

        #Add feedback to results if available
        results['feedback'] = {}
        if 'topicality' in feedback and 'prompt_overlap' in feedback:
            results['feedback'].update({
                'topicality' : feedback['topicality'],
                'prompt-overlap' : feedback['prompt_overlap'],
                })

        results['feedback'].update(
            {
                'spelling' : feedback['spelling'],
                'grammar' : feedback['grammar'],
                'markup-text' : feedback['markup_text'],
                }
        )

    else:
        #If error, success is False.
        results['success']=False

    return results

def grade_generic(grader_data, numeric_features, textual_features):
    """
    Grades a set of numeric and textual features using a generic model
    grader_data -- dictionary containing:
    {
        'algorithm' - Type of algorithm to use to score
    }
    numeric_features - list of numeric features to predict on
    textual_features - list of textual feature to predict on

    """
    results = {'errors': [],'tests': [],'score': 0, 'success' : False, 'confidence' : 0}

    has_error=False

    #Try to find and load the model file

    grader_set=predictor_set.PredictorSet(essaytype="test")

    model, extractor = get_classifier_and_ext(grader_data)

    #Try to add essays to essay set object
    try:
        grader_set.add_row(numeric_features, textual_features,0)
    except Exception:
        error_msg = "Row could not be added to predictor set:{0} {1}".format(numeric_features, textual_features)
        log.exception(error_msg)
        results['errors'].append(error_msg)
        has_error=True

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=extractor.gen_feats(grader_set)
        results['score']=model.predict(grader_feats)[0]
    except Exception:
        error_msg = "Could not extract features and score essay."
        log.exception(error_msg)
        results['errors'].append(error_msg)
        has_error=True

    #Try to determine confidence level
    try:
        results['confidence'] = get_confidence_value(grader_data['algorithm'],model, grader_feats, results['score'])
    except Exception:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

    if not has_error:
        results['success'] = True

    return results

def get_confidence_value(algorithm,model,grader_feats,score, scores):
    """
    Determines a confidence in a certain score, given proper input parameters
    algorithm- from util_functions.AlgorithmTypes
    model - a trained model
    grader_feats - a row of features used by the model for classification/regression
    score - The score assigned to the submission by a prior model
    """
    min_score=min(numpy.asarray(scores))
    max_score=max(numpy.asarray(scores))
    if algorithm == util_functions.AlgorithmTypes.classification and hasattr(model, "predict_proba"):
        #If classification, predict with probability, which gives you a matrix of confidences per score point
        raw_confidence=model.predict_proba(grader_feats)[0,(float(score)-float(min_score))]
        #TODO: Normalize confidence somehow here
        confidence=raw_confidence
    elif hasattr(model, "predict"):
        raw_confidence = model.predict(grader_feats)[0]
        confidence = max(float(raw_confidence) - math.floor(float(raw_confidence)), math.ceil(float(raw_confidence)) - float(raw_confidence))
    else:
        confidence = 0

    return confidence

def get_classifier_and_ext(grader_data):
    if 'classifier' in grader_data:
        model = grader_data['classifier']
    elif 'model' in grader_data:
        model = grader_data['model']
    else:
        raise Exception("Cannot find a valid model.")

    if 'feature_ext' in grader_data:
        extractor = grader_data['feature_ext']
    elif 'extractor' in grader_data:
        extractor = grader_data['extractor']
    else:
        raise Exception("Cannot find the extractor")

    return model, extractor


