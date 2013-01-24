import os
import sys
import logging
log = logging.getLogger(__name__)

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

import model_creator
import util_functions
import predictor_set
import predictor_extractor

from statsd import statsd

@statsd.timed('open_ended_assessment.machine_learning.creator.time')
def create(text,score,prompt_string,model_path):

    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : ""}

    if len(text)!=len(score):
        msg = "Target and text lists must be same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    try:
        e_set = model_creator.create_essay_set(text, score, prompt_string)
    except:
        msg = "essay set creation failed."
        results['errors'].append(msg)
        log.exception(msg)
    try:
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model(e_set)
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


def create_generic(numeric_values, textual_values, target, model_path, algorithm = model_creator.AlgorithmTypes.regression):
    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : ""}

    if len(numeric_values)!=len(textual_values) or len(numeric_values)!=len(target):
        msg = "Target, numeric features, and text features must all be the same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    try:
        pset = predictor_set.PredictorSet(type="train")
        for i in xrange(0, len(numeric_values)):
            pset.add_row(numeric_values[i], textual_values[i], target[i])
    except:
        msg = "predictor set creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    try:
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