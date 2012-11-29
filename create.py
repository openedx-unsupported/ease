import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

import model_creator
import util_functions

from statsd import statsd

@statsd.timed('open_ended_assessment.machine_learning.creator.time')
def create(text,score,prompt_string,model_path):

    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0}
    try:
        e_set = model_creator.create_essay_set(text, score, prompt_string)
    except:
        results['errors'].append("essay set creation failed.")
    try:
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model(e_set)
        results['cv_kappa']=cv_error_results['kappa']
        results['cv_mean_absolute_error']=cv_error_results['mae']
    except:
        results['errors'].append("feature extraction and model creation failed.")
    try:
        model_creator.dump_model_to_file(prompt_string, feature_ext, classifier, text, score, model_path)
        results['success']=True
    except:
        results['errors'].append("could not write model to: {0}".format(model_path))

    #Count number of successful/unsuccessful creations
    statsd.increment("open_ended_assessment.machine_learning.creator_count",
        tags=["success:{0}".format(results['success'])])

    return results

