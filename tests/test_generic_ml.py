import os
import sys
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path=os.path.abspath(os.path.join(base_path,'..'))
sys.path.append(one_up_path)

import util_functions
import predictor_set
import predictor_extractor
import numpy

from sklearn.ensemble import GradientBoostingClassifier

if not base_path.endswith("/"):
    base_path=base_path+"/"

FILENAME="sa_data.tsv"


sa_val = file(FILENAME)
scores=[]
texts=[]
lines=sa_val.readlines()
pset = predictor_set.PredictorSet(type="train")
for i in xrange(1,len(lines)):
    score,text=lines[i].split("\t\"")
    if len(text)>t_len:
        scores.append(int(score))
        texts.append(text)
        pset.add_row([1],[text],int(score))
extractor=predictor_extractor.PredictorExtractor()
extractor.initialize_dictionaries(pset)
train_feats=extractor.gen_feats(pset)

clf=GradientBoostingClassifier(n_estimators=100, learn_rate=.05,max_depth=4, random_state=1,min_samples_leaf=3)
cv_preds=util_functions.gen_cv_preds(clf,train_feats,scores)
err=numpy.mean(numpy.abs(cv_preds-scores))
print err
kappa=util_functions.quadratic_weighted_kappa(list(cv_preds),scores)
print kappa
all_err.append(err)
all_kappa.append(kappa)

"""
outfile=open("full_cvout.tsv",'w+')
outfile.write("cv_pred" + "\t" + "actual")
for i in xrange(0,len(cv_preds)):
    outfile.write("{0}\t{1}".format(cv_preds[i],scores[i]))
"""