#Grader called by pyxserver_wsgi.py
#Loads a grader file, which is a dict containing the prompt of the question,
#a feature extractor object, and a trained model.
#Extracts features and runs trained model on the submission to produce a final score.
#Correctness determined by ratio of score to max possible score.
#Requires aspell to be installed and added to the path.

import sys
import pickle
import os
import numpy

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

from essay_set import EssaySet

#Imports needed to unpickle grader data
import feature_extractor
import sklearn.ensemble

def grade(grader_path,submission,sandbox):
    results = {'errors': [],'tests': [],'correct': False,'score': 0, 'feedback' : []}

    #Try to find and load the model file

    try:
        grader_data=pickle.load(file(grader_path,"r"))
    except:
        results['errors'].append("Could not find a valid model file.")
    grader_set=EssaySet(type="test")

    #Try to add essays to essay set object
    try:
        grader_set.add_essay(str(submission),0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except:
        results['errors'].append("Essay could not be added to essay set:{0}".format(submission))

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=grader_data['extractor'].gen_feats(grader_set)
        results['feedback']=grader_data['extractor'].gen_feedback(grader_set)
        results['score']=int(grader_data['model'].predict(grader_feats)[0])
    except:
        results['errors'].append("Could not extract features and score essay.")

    #Determine maximum score and correctness of response
    max_score=numpy.max(grader_data['model'].classes_)
    if results['score']/float(max_score) >= .66:
        results['correct']=True
    else:
        results['correct']=False
    return results


    
