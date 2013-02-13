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
from multiprocessing import Pool

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

if not base_path.endswith("/"):
    base_path=base_path+"/"

data_path = "/home/vik/mitx_all/vik_sandbox/hewlett_essay_data/split_data"
if not data_path.endswith("/"):
    data_path=data_path+"/"
filenames = [str(i) +".tsv" for i in xrange(1,19)]

run_cv = False

def run_single_worker(args):
    filename,data_path,run_cv = args
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
    for i in xrange(1,len(lines)):
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

    if run_cv:
        try:
            cv_preds=util_functions.gen_cv_preds(clf,train_feats,score1s, num_chunks = 10) # int(math.floor(len(texts)/2)
        except:
            cv_preds = score1s
    else:
        try:
            random_nums = list(numpy.random.random_integers(0, train_feats.shape[0], 100))
            out_group_rows = [row for row in xrange(0,train_feats.shape[0]) if row not in random_nums]
            in_group_scores = list(numpy.array(score1s)[random_nums])
            out_group_scores = list(numpy.array(score1s)[out_group_rows])
            out_group_score2s = list(numpy.array(score2s)[out_group_rows])
            score1s = out_group_scores
            score2s = out_group_score2s
            model = util_functions.gen_model(clf,train_feats[random_nums,:],in_group_scores)
            cv_preds = util_functions.gen_preds(model,train_feats[out_group_rows,:])
        except:
            cv_preds = score1s[100:]

    rounded_cv = [int(round(cv)) for cv in list(cv_preds)]
    added_score1 = [s1+1 for s1 in score1s]
    err=numpy.mean(numpy.abs(numpy.array(cv_preds)-score1s))
    kappa=util_functions.quadratic_weighted_kappa(rounded_cv, score1s)
    percent_error = numpy.mean(numpy.abs(score1s - numpy.array(cv_preds))/added_score1)
    human_err=numpy.mean(numpy.abs(numpy.array(score2s)-score1s))
    human_kappa=util_functions.quadratic_weighted_kappa(list(score2s),score1s)
    human_percent_error = numpy.mean(numpy.abs(score1s - numpy.array(score2s))/added_score1)

    outfile=open(data_path + "outdata/" + filename,'w+')
    outfile.write("cv_pred" + "\t" + "actual1\t" + "actual2\n")
    for i in xrange(0,len(cv_preds)):
        outfile.write("{0}\t{1}\t{2}\n".format(str(cv_preds[i]),str(score1s[i]), str(score2s[i])))
    outfile.close()

    return err, kappa,percent_error,human_err,human_kappa,human_percent_error

length = len(filenames)
np=8
p = Pool(processes=np)
errs, kappas,percent_errors,human_errs,human_kappas,human_percent_errors = zip(*p.map(run_single_worker,[(filenames[i],data_path,run_cv) for i in xrange(0,length)]))

outfile=open(data_path + "outdata/summary.tsv",'w+')
outfile.write("set\terr\tkappa\tpercent_error\thuman_err\thuman_kappa\thuman_percent_error\n")
for i in xrange(0,len(errs)):
    outfile.write("{set}\t{err}\t{kappa}\t{percent_error}\t{human_err}\t{human_kappa}\t{human_percent_error}\n".format(
        set=i+1,err=errs[i],kappa=kappas[i],percent_error=percent_errors[i], human_err=human_errs[i],
        human_kappa=human_kappas[i], human_percent_error=human_percent_errors[i]))
outfile.close()






