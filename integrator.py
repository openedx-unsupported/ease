import numpy
import re
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from itertools import chain
from sklearn.ensemble import RandomForestClassifier

base_path="C:/Users/Vik/Documents/Consulting/PyShortEssay/"
os.chdir(base_path)

sys.path.append("C:/Users/Vik/Documents/rscripts/python_ses")
from essay_set import essay_set
import util_functions
import mech_turk_interface
from feature_extractor import feature_extractor

id,e_set,score,score2,text=[],[],[],[],[]
combined_raw=open(base_path + "train.tsv").read()
raw_lines=combined_raw.splitlines()
for row in xrange(1,len(raw_lines)):
    id1,set1,score1,score12,text1 = raw_lines[row].strip().split("\t")
    id.append(int(id1))
    text.append(text1)
    e_set.append(int(set1))
    score.append(int(score1))
    score2.append(int(score12))
    
prompt_string="A group of students wrote the following procedure for their investigation. Procedure: 1. Determine the mass of four different samples. 2. Pour vinegar in each of four separate, but identical, containers. 3. Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container. 4. After 24 hours, remove the samples from the containers and rinse each sample with distilled water. 5. Allow the samples to sit and dry for 30 minutes. 6. Determine the mass of each sample. The students’ data are recorded in the table below. Sample Starting Mass (g) Ending Mass (g) Difference in Mass (g) Marble 9.8 9.4 –0.4 Limestone 10.4 9.1 –1.3 Wood 11.2 11.2 0.0 Plastic 7.2 7.1 –0.1"

question_string="After reading the group’s procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information."

x=essay_set()
m_coef=1572
for i in xrange(0,len(text)-m_coef):
    x.add_essay(text[i],score[i])
    if(score[i]==min(score)):
        x.generate_additional_essays(x._clean_text[len(x._clean_text)-1],score[i])
            
x.update_prompt(prompt_string)

all_train_toks=util_functions.f7(list(chain.from_iterable([x._tokens[t] for t in range(0,len(x._tokens)) if x._generated[t]==0])))
    
x_t=essay_set(type="test")
for i in xrange(len(text)-m_coef,len(text)):
    #te_toks=nltk.word_tokenize(text[i].lower())
    #tok_overlap=float(len([tok for tok in te_toks if tok in all_train_toks]))/len(te_toks)
    #if tok_overlap>=0:
    x_t.add_essay(text[i],score[i])
        
x_t.update_prompt(prompt_string)
    
f=feature_extractor()
f.initialize_dictionaries(x)

train_feats=f.gen_feats(x)
test_feats=f.gen_feats(x_t)

clf = GradientBoostingClassifier(n_estimators=100, learn_rate=.05,max_depth=4, random_state=1,min_samples_leaf=3)

cv_preds=util_functions.gen_cv_preds(clf,train_feats,x._score)
print "CV Train: " + str(util_functions.quadratic_weighted_kappa(cv_preds,x._score))

model=util_functions.gen_model(clf,train_feats,x._score)
preds=util_functions.gen_preds(clf,test_feats)

print "Test Err: " + str(util_functions.quadratic_weighted_kappa(preds,x_t._score))
print "Conf Mat:\n" + str(numpy.array(util_functions.confusion_matrix(preds,x_t._score)))


prompt=prompt_string
question=question_string
essay_text=text[100:110]
all_essays=text[0:100]
all_scores=score[0:100]

ACCESS_ID =
SECRET_KEY =
HOST = 'mechanicalturk.sandbox.amazonaws.com'
#HOST = 'mechanicalturk.amazonaws.com'

hit_creator=mech_turk_interface.HITCreator(ACCESS_ID,SECRET_KEY,HOST,essay_text,prompt,question,all_essays,all_scores,assignment_count=3)
hit_creator.create_hits(reward=.20,add_qualifications=True)
new_results=hit_creator.hit_container.get_all_results()
print new_results
print [util_functions.getMedian(x) for x in new_results[0]]
hit_creator.hit_container.process_approvals()

