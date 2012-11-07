import os
import sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

import model_creator

def create(text,scores,prompt,model_path):
    if not model_path.endswith(".p"):
        model_path+=".p"

    results = {'errors': [],'created' : False}
    try:
        e_set = model_creator.create_essay_set(text, score, prompt_string)
    except:
        results['errors'].append("essay set creation failed.")
    try:
        feature_ext, classifier = model_creator.extract_features_and_generate_model(e_set)
    except:
        results['errors'].append("feature extraction and model creation failed.")
    try:
        model_creator.dump_model_to_file(prompt_string, feature_ext, classifier, text, score, args.model_path)
        results['created']=True
    except:
        results['errors'].append("could not write model to: {0}".format(model_path))

    return results


