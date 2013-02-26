""""
Functions to score specified data using specified ML models
""""

import sys
import pickle
import os
import numpy
import logging
from statsd import statsd

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

@statsd.timed('open_ended_assessment.machine_learning.grader.time')
def grade(grader_data,grader_config,submission):
    """
    Grades a specified submission using specified models
    grader_data - A dictionary:
    {
        'model' : trained model,
        'extractor' : trained feature extractor,
        'prompt' : prompt for the question,
    }
    grader_config - Legacy, kept for compatibility with old code.  Need to remove.
    submission - The student submission (string)
    """

    #Initialize result dictionary
    results = {'errors': [],'tests': [],'score': 0, 'feedback' : "", 'success' : False, 'confidence' : 0}
    has_error=False

    grader_set=EssaySet(type="test")

    try:
        #Try to add essay to essay set object
        grader_set.add_essay(str(submission),0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except:
        results['errors'].append("Essay could not be added to essay set:{0}".format(submission))
        has_error=True

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=grader_data['extractor'].gen_feats(grader_set)
        feedback=grader_data['extractor'].gen_feedback(grader_set,grader_feats)[0]
        results['score']=int(grader_data['model'].predict(grader_feats)[0])
    except :
        results['errors'].append("Could not extract features and score essay.")
        has_error=True

    #Try to determine confidence level
    try:
        min_score=min(numpy.asarray(grader_data['score']))
        max_score=max(numpy.asarray(grader_data['score']))
        raw_confidence=grader_data['model'].predict_proba(grader_feats)[0,(results['score']-min_score)]
        #TODO: Normalize confidence somehow here
        results['confidence']=raw_confidence
    except:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

    if not has_error:

        #If the essay is just a copy of the prompt, return a 0 as the score
        if(feedback['too_similar_to_prompt']):
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

        #Only return spelling and grammar feedback for low scoring responses
        if results['score']/float(max_score)<.33:
            results['feedback'].update(
                {'spelling' : feedback['spelling'],
            'grammar' : feedback['grammar'],
            'markup-text' : feedback['markup_text'],
            })

    else:
        #If error, success is False.
        results['success']=False

    #Count number of successful/unsuccessful gradings
    statsd.increment("open_ended_assessment.machine_learning.grader_count",
        tags=["success:{0}".format(results['success'])])

    return results

def grade_generic(grader_data, grader_config, numeric_features, textual_features):
    results = {'errors': [],'tests': [],'score': 0, 'success' : False, 'confidence' : 0}

    has_error=False

    #Try to find and load the model file

    grader_set=predictor_set.PredictorSet(type="test")

    #Try to add essays to essay set object
    try:
        grader_set.add_row(numeric_features, textual_features,0)
    except:
        results['errors'].append("Row could not be added to predictor set:{0} {1}".format(numeric_features, textual_features))
        has_error=True

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=grader_data['extractor'].gen_feats(grader_set)
        results['score']=grader_data['model'].predict(grader_feats)[0]
    except :
        results['errors'].append("Could not extract features and score essay.")
        has_error=True

    #Try to determine confidence level
    try:
        min_score=min(numpy.asarray(grader_data['score']))
        max_score=max(numpy.asarray(grader_data['score']))
        if grader_data['algorithm'] == util_functions.AlgorithmTypes.classification:
            raw_confidence=grader_data['model'].predict_proba(grader_feats)[0,(results['score']-min_score)]
            #TODO: Normalize confidence somehow here
            results['confidence']=raw_confidence
        else:
            raw_confidence = grader_data['model'].predict(grader_feats)[0]
            confidence = max(raw_confidence - math.floor(raw_confidence), math.ceil(raw_confidence) - raw_confidence)
            results['confidence'] = confidence
    except:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

        #Count number of successful/unsuccessful gradings
    statsd.increment("open_ended_assessment.machine_learning.grader_count",
        tags=["success:{0}".format(results['success'])])

    if not has_error:
        results['success'] = True

    return results
