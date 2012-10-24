#Provides interface functions to create and save models

import numpy
import re
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import sklearn.ensemble
from itertools import chain

base_path = os.path.dirname( __file__ )
sys.path.append(base_path)

from essay_set import essay_set
import util_functions
import feature_extractor

def read_in_test_data(filename):
    id,e_set,score,score2,text=[],[],[],[],[]
    combined_raw=open(filename).read()
    raw_lines=combined_raw.splitlines()
    for row in xrange(1,len(raw_lines)):
        id1,set1,score1,score12,text1 = raw_lines[row].strip().split("\t")
        id.append(int(id1))
        text.append(text1)
        e_set.append(int(set1))
        score.append(int(score1))
        score2.append(int(score12))

    return score,text

def read_in_test_prompt(filename):
    prompt_string=open(filename).read()
    return prompt_string

#Create an essay set.  text and score should be lists of strings and ints, respectively.
def create_essay_set(text,score,prompt_string,generate_additional=True):
    x=essay_set()
    for i in xrange(0,len(text)):
        x.add_essay(text[i],score[i])
        if score[i]==min(score) and generate_additional==True:
            x.generate_additional_essays(x._clean_text[len(x._clean_text)-1],score[i])

    x.update_prompt(prompt_string)

    return x

#Feed in an essay set to get feature vector and classifier
def extract_features_and_generate_model(essays):
    f=feature_extractor.feature_extractor()
    f.initialize_dictionaries(essays)

    train_feats=f.gen_feats(essays)

    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learn_rate=.05,
                                                      max_depth=4, random_state=1,
                                                      min_samples_leaf=3)

    model=util_functions.gen_model(clf,train_feats,essays._score)

    return f,clf

#Writes out model to pickle file
def dump_model_to_file(prompt_string,feature_ext,classifier,model_path):
    model_file={'prompt': prompt_string, 'extractor' : feature_ext, 'model' : classifier}
    pickle.dump(model_file,file=open(model_path,"w"))


