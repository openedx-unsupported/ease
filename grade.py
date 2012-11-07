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
import logging

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

from essay_set import EssaySet

#Imports needed to unpickle grader data
import feature_extractor
import sklearn.ensemble

log = logging.getLogger(__name__)

feedback_template = u"""
<div class="feedback">
<header>Feedback</header>
  <section>
    <div class="topicality">
      {topicality}
    </div>
    <div class="spelling">
      {spelling}
    </div>
    <div class="grammar">
        {grammar}
    </div>
    <div class="markup_text">
        {markup_text}
    </div>
  </section>
</div>
"""

error_template = u"""
<div class="feedback">
<header>Feedback</header>
  <section>
    <div class="error">
      {errors}
    </div>
  </section>
</div>
"""


def grade(grader_path,submission,sandbox=None):
    log.debug("Grader path: {0}\n Submission: {1}".format(grader_path,submission))
    results = {'errors': [],'tests': [],'correct': False,'score': 0, 'feedback' : ""}

    has_error=False

    #Try to find and load the model file

    try:
        grader_data=pickle.load(file(grader_path,"r"))
    except:
        results['errors'].append("Could not find a valid model file.")
        has_error=True
    grader_set=EssaySet(type="test")

    #Try to add essays to essay set object
    try:
        grader_set.add_essay(str(submission),0)
        grader_set.update_prompt(str(grader_data['prompt']))
    except:
        results['errors'].append("Essay could not be added to essay set:{0}".format(submission))
        has_error=True

    #Try to extract features from submission and assign score via the model
    try:
        grader_feats=grader_data['extractor'].gen_feats(grader_set)
        feedback=grader_data['extractor'].gen_feedback(grader_set)[0]
        results['score']=int(grader_data['model'].predict(grader_feats)[0])
    except :
        results['errors'].append("Could not extract features and score essay.")
        has_error=True

    #Determine maximum score and correctness of response
    max_score=numpy.max(grader_data['model'].classes_)
    if results['score']/float(max_score) >= .66:
        results['correct']=True
    else:
        results['correct']=False

    #Add feedback template to results
    if not has_error:
        results['feedback']=feedback_template.format(topicality=feedback['topicality'],
            spelling=feedback['spelling'],grammar=feedback['grammar'],markup_text=feedback['markup_text'])
    else:
        results['feedback']=error_template.format(errors=' '.join(results['errors']))

    return results


    
