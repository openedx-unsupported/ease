import os
import sys
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

import util_functions
import essay_set
import feature_extractor

from sklearn.ensemble import GradientBoostingClassifier

if not base_path.endswith("/"):
    base_path=base_path+"/"

FILENAME="sa_data.tsv"

sa_val = file(FILENAME)

scores=[]
texts=[]
lines=sa_val.readlines()

eset=essay_set.EssaySet(type="train")

for i in xrange(1,len(lines)):
    score,text=lines[i].split("\t\"")
    scores.append(int(score))
    texts.append(text)
    eset.add_essay(text,int(score))
    if score==0:
        eset.generate_additional_essays(text,int(score))

extractor=feature_extractor.FeatureExtractor()
extractor.initialize_dictionaries(eset)
train_feats=extractor.gen_feats(eset)

clf=GradientBoostingClassifier(n_estimators=100, learn_rate=.05,
    max_depth=4, random_state=1,
    min_samples_leaf=3)

cv_preds=util_functions.gen_cv_preds(clf,train_feats,scores)

kappa=util_functions.quadratic_weighted_kappa(cv_preds,scores)

outfile=open("cvout.tsv",'w+')
outfile.write("cv_pred" + "\t" + "actual")
for i in xrange(0,len(cv_preds)):
    outfile.write("{0}\t{1}".format(cv_preds[i],scores[i]))



