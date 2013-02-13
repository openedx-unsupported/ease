import os
import sys
#base_path = os.path.dirname(__file__)
base_path = "/home/vik/mitx_all/machine-learning"
sys.path.append(base_path)

one_up_path=os.path.abspath(os.path.join(base_path,'..'))
sys.path.append(one_up_path)

import util_functions
import essay_set
import feature_extractor
import numpy
import math

from sklearn.ensemble import GradientBoostingClassifier

if not base_path.endswith("/"):
    base_path=base_path+"/"

data_path = "/home/vik/mitx_all/vik_sandbox/hewlett_essay_data/split_data"
if not data_path.endswith("/"):
    data_path=data_path+"/"
filenames = [str(i) +".tsv" for i in xrange(1,19)]
kappas = []
errs = []
percent_errors=[]
human_kappas=[]
human_errs=[]
human_percent_errors=[]

for filename in filenames:
    base_name = data_path + filename
    print base_name
    sa_val = file(base_name)
    id_vals=[]
    essay_set_nums=[]
    score1s=[]
    score2s=[]
    texts=[]
    lines=sa_val.readlines()
    eset=essay_set.EssaySet(type="train")
    #len(lines)
    for i in xrange(1,10):
        id_val,essay_set_num,score1,score2,text=lines[i].split("\t")
        score1s.append(int(score1))
        score2s.append(int(score2))
        texts.append(text)
        essay_set_nums.append(essay_set_num)
        id_vals.append(id_val)
        eset.add_essay(text,int(score1))
        #if int(score)==0:
        #    eset.generate_additional_essays(text,int(score))
    extractor=feature_extractor.FeatureExtractor()
    extractor.initialize_dictionaries(eset)
    train_feats=extractor.gen_feats(eset)
    print(max(score1s))
    if max(score1s)<=3:
        clf=GradientBoostingClassifier(n_estimators=100, learn_rate=.05,max_depth=4, random_state=1,min_samples_leaf=3)
    else:
        clf=GradientBoostingRegressor(n_estimators=100, learn_rate=.05, max_depth=4, random_state=1, min_samples_leaf=3)

    try:
        cv_preds=util_functions.gen_cv_preds(clf,train_feats,score1s, num_chunks = 3) # int(math.floor(len(texts)/2)
    except:
        cv_preds = score1s

    rounded_cv = [int(round(cv)) for cv in list(cv_preds)]

    err=numpy.mean(numpy.abs(numpy.array(cv_preds)-score1s))
    errs.append(err)
    print err
    kappa=util_functions.quadratic_weighted_kappa(rounded_cv, score1s)
    kappas.append(kappa)
    print kappa
    percent_error = numpy.mean(numpy.abs(score1s - numpy.array(cv_preds))/score1s)
    percent_errors.append(percent_error)
    print percent_error

    human_err=numpy.mean(numpy.abs(numpy.array(score2s)-score1s))
    human_errs.append(human_err)
    print human_err
    human_kappa=util_functions.quadratic_weighted_kappa(list(score2s),score1s)
    human_kappas.append(human_kappa)
    print human_kappa
    human_percent_error = numpy.mean(numpy.abs(score1s - numpy.array(score2s))/score1s)
    human_percent_errors.append(human_percent_error)
    print human_percent_error

    outfile=open(data_path + "outdata/" + filename,'w+')
    outfile.write("cv_pred" + "\t" + "actual1\t" + "actual2\n")
    for i in xrange(0,len(cv_preds)):
        outfile.write("{0}\t{1}\t{2}\n".format(str(cv_preds[i]),str(score1s[i]), str(score2s[i])))
    outfile.close()

outfile=open(data_path + "outdata/summary.tsv",'w+')
outfile.write("set\terr\tkappa\tpercent_error\thuman_err\thuman_kappa\thuman_percent_error\n")
for i in xrange(0,len(cv_preds)):
    outfile.write("{set}\t{err}\t{kappa}\t{percent_error}\t{human_err}\t{human_kappa}\t{human_percent_error}\n".format(
        set=i+1,err=errs[i],kappa=kappas[i],percent_error=percent_errors[i], human_err=human_errs[i],
        human_kappa=human_kappas[i], human_percent_error=human_percent_errors[i]))
outfile.close()





