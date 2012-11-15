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
import util_functions

#Imports needed to unpickle grader data
import feature_extractor
import sklearn.ensemble

log = logging.getLogger(__name__)

TEMPORARY_WANTS_CONFIG=True

feedback_template = u"""

<section>
    <header>Feedback</header>
    <div class="shortform">
        <div class="result-output">
          Number of potential problem areas identified: {problem_areas}
        </div>
    </div>
    <div class="longform">
        <div class="result-output">
          <div class="topicality">
            Topicality: {topicality}
          </div>
          <div class="spelling">
            Spelling: {spelling}
          </div>
          <div class="grammar">
            Grammar: {grammar}
          </div>
          <div class="markup-text">
            {markup_text}
          </div>
        </div>
    </div>
</section>

"""

error_template = u"""

<section>
    <div class="shortform">
        <div class="result-errors">
          There was an error with your submission.  Please contact course staff.
        </div>
    </div>
    <div class="longform">
        <div class="result-errors">
          {errors}
        </div>
    </div>
</section>

"""


def grade(grader_path,grader_config,submission,sandbox=None):

    grader_path=util_functions.create_model_path(grader_path)

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
        feedback=grader_data['extractor'].gen_feedback(grader_set,grader_feats)[0]
        results['score']=int(grader_data['model'].predict(grader_feats)[0])
    except :
        results['errors'].append("Could not extract features and score essay.")
        has_error=True

    if not has_error:
        #Determine maximum score and correctness of response
        max_score=numpy.max(grader_data['model'].classes_)
        if results['score']/float(max_score) >= .66:
            results['correct']=True
        else:
            results['correct']=False

        #Generate short form output--number of problem areas identified in feedback
        problem_areas=0
        for tag in feedback:
            if tag is not 'markup_text':
                problem_areas+=len(feedback[tag])>5

        #Add feedback template to results
        results['feedback']=feedback_template.format(topicality=feedback['topicality'],
            spelling=feedback['spelling'],grammar=feedback['grammar'],
            markup_text=feedback['markup_text'],problem_areas=problem_areas)
    else:
        #If error, add errors to template.
        results['feedback']=error_template.format(errors=' '.join(results['errors']))

    return results



    
