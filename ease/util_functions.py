"""
Collection of misc functions needed to support essay_set.py and feature_extractor.py.
Requires aspell to be installed and added to the path

The cleanup of this file is far beyond the scope of this hackathon --GBW--
"""
from fisher import pvalue

aspell_path = "aspell"
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy
from itertools import chain
import math
import nltk
import logging
import sys
import tempfile

log = logging.getLogger(__name__)

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
if not base_path.endswith("/"):
    base_path = base_path + "/"

#Paths to needed data files
ESSAY_CORPUS_PATH = base_path + "data/essaycorpus.txt"
ESSAY_COR_TOKENS_PATH = base_path + "data/essay_cor_tokens.p"


class AlgorithmTypes(object):
    """
    Defines what types of algorithm can be used
    """
    regression = "regression"
    classification = "classifiction"


def sub_chars(string):
    """
    Strips illegal characters from a string.  Used to sanitize input essays.
    Removes all non-punctuation, digit, or letter characters.
    Returns sanitized string.
    string - string
    """
    #Define replacement patterns
    sub_pat = r"[^A-Za-z\.\?!,';:]"
    char_pat = r"\."
    com_pat = r","
    ques_pat = r"\?"
    excl_pat = r"!"
    sem_pat = r";"
    col_pat = r":"
    whitespace_pat = r"\s{1,}"

    #Replace text.  Ordering is very important!
    nstring = re.sub(sub_pat, " ", string)
    nstring = re.sub(char_pat, " .", nstring)
    nstring = re.sub(com_pat, " ,", nstring)
    nstring = re.sub(ques_pat, " ?", nstring)
    nstring = re.sub(excl_pat, " !", nstring)
    nstring = re.sub(sem_pat, " ;", nstring)
    nstring = re.sub(col_pat, " :", nstring)
    nstring = re.sub(whitespace_pat, " ", nstring)

    return nstring


def spell_correct(string):
    """
    Uses aspell to spell correct an input string.
    Requires aspell to be installed and added to the path.
    Returns the spell corrected string if aspell is found, original string if not.
    string - string
    """

    # Create a temp file so that aspell could be used
    # By default, tempfile will delete this file when the file handle is closed.
    f = tempfile.NamedTemporaryFile(mode='w')
    f.write(string)
    f.flush()
    f_path = os.path.abspath(f.name)
    try:
        p = os.popen(aspell_path + " -a < " + f_path + " --sug-mode=ultra")

        # Aspell returns a list of incorrect words with the above flags
        incorrect = p.readlines()
        p.close()

    except Exception as ex:
        log.exception("aspell spell checking was not run, because it failed with the exception: {}".format(ex))
        # DESIGN CHOICE: Return original string if aspell fails
        return string, 0, string

    finally:
        f.close()

    incorrect_words = list()
    correct_spelling = list()
    for i in range(1, len(incorrect)):
        if (len(incorrect[i]) > 10):
            #Reformat aspell output to make sense
            match = re.search(":", incorrect[i])
            if hasattr(match, "start"):
                begstring = incorrect[i][2:match.start()]
                begmatch = re.search(" ", begstring)
                begword = begstring[0:begmatch.start()]

                sugstring = incorrect[i][match.start() + 2:]
                sugmatch = re.search(",", sugstring)
                if hasattr(sugmatch, "start"):
                    sug = sugstring[0:sugmatch.start()]

                    incorrect_words.append(begword)
                    correct_spelling.append(sug)

    #Create markup based on spelling errors
    newstring = string
    markup_string = string
    already_subbed = []
    for i in range(0, len(incorrect_words)):
        sub_pat = r"\b" + incorrect_words[i] + r"\b"
        sub_comp = re.compile(sub_pat)
        newstring = re.sub(sub_comp, correct_spelling[i], newstring)
        if incorrect_words[i] not in already_subbed:
            markup_string = re.sub(sub_comp, '<bs>' + incorrect_words[i] + "</bs>", markup_string)
            already_subbed.append(incorrect_words[i])

    return newstring, len(incorrect_words), markup_string


def ngrams(tokens, min_n, max_n):
    """
    Generates ngrams(word sequences of fixed length) from an input token sequence.
    tokens is a list of words.
    min_n is the minimum length of an ngram to return.
    max_n is the maximum length of an ngram to return.
    returns a list of ngrams (words separated by a space)
    """
    all_ngrams = list()
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i + min_n, min(n_tokens, i + max_n) + 1):
            all_ngrams.append(" ".join(tokens[i:j]))
    return all_ngrams


def get_vocab(essays, scores, max_features_pass_1=750, max_features_pass_2=200):
    """
    Uses a fisher test to find words that are significant in that they separate
    high scoring essays from low scoring essays.

    Args:
        essays (list of str): a list of input essays
        scores (list of int): a list of associated input scores

    Kwargs:
        max_features_pass_1 (int): the maximum number of features to consider in the first pass of the essays
        max_features_pass_2 (int): the maximum number of features to consider in the second pass of the essays
            (DEFAULTS of 750 and 200 respectively)

    Returns:
        (list of str): A list of strings which constitute the signifigant vocabulary which diferentiates between
                        strong and weak essays.

    NOTE: GBW didn't mess around with this because it is very easy to mess up, and I didn't want to mess it up.
    """
    dict = CountVectorizer(ngram_range=(1, 2), max_features=max_features_pass_1)
    dict_matrix = dict.fit_transform(essays)
    set_score = numpy.asarray(scores, dtype=numpy.int)
    med_score = numpy.median(set_score)
    new_score = set_score
    if (med_score == 0):
        med_score = 1
    new_score[set_score < med_score] = 0
    new_score[set_score >= med_score] = 1

    fish_vals = []
    for col_num in range(0, dict_matrix.shape[1]):
        loop_vec = dict_matrix.getcol(col_num).toarray()
        good_loop_vec = loop_vec[new_score == 1]
        bad_loop_vec = loop_vec[new_score == 0]
        good_loop_present = len(good_loop_vec[good_loop_vec > 0])
        good_loop_missing = len(good_loop_vec[good_loop_vec == 0])
        bad_loop_present = len(bad_loop_vec[bad_loop_vec > 0])
        bad_loop_missing = len(bad_loop_vec[bad_loop_vec == 0])
        fish_val = pvalue(good_loop_present, bad_loop_present, good_loop_missing, bad_loop_missing).two_tail
        fish_vals.append(fish_val)

    cutoff = 1
    if (len(fish_vals) > max_features_pass_2):
        cutoff = sorted(fish_vals)[max_features_pass_2]
    good_cols = numpy.asarray([num for num in range(0, dict_matrix.shape[1]) if fish_vals[num] <= cutoff])

    getVar = lambda searchList, ind: [searchList[i] for i in ind]
    vocab = getVar(dict.get_feature_names(), good_cols)

    return vocab


def gen_cv_preds(clf, arr, sel_score, num_chunks=3):
    """
    Generates cross validated predictions using an input classifier and data.
    clf is a classifier that implements that implements the fit and predict methods.
    arr is the input data array (X)
    sel_score is the target list (y).  y[n] corresponds to X[n,:]
    num_chunks is the number of cross validation folds to use
    Returns an array of the predictions where prediction[n] corresponds to X[n,:]
    """
    cv_len = int(math.floor(len(sel_score) / num_chunks))
    chunks = []
    for i in range(0, num_chunks):
        range_min = i * cv_len
        range_max = ((i + 1) * cv_len)
        if i == num_chunks - 1:
            range_max = len(sel_score)
        chunks.append(range(range_min, range_max))
    preds = []
    set_score = numpy.asarray(sel_score, dtype=numpy.int)
    chunk_vec = numpy.asarray(range(0, len(chunks)))
    for i in xrange(0, len(chunks)):
        loop_inds = list(
            chain.from_iterable([chunks[int(z)] for z, m in enumerate(range(0, len(chunks))) if int(z) != i]))
        sim_fit = clf.fit(arr[loop_inds], set_score[loop_inds])
        preds.append(list(sim_fit.predict(arr[chunks[i]])))
    all_preds = list(chain(*preds))
    return (all_preds)


stdev = lambda d: (sum((x - 1. * sum(d) / len(d)) ** 2 for x in d) / (1. * (len(d) - 1))) ** .5


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates kappa correlation between rater_a and rater_b.
    Kappa measures how well 2 quantities vary together.
    rater_a is a list of rater a scores
    rater_b is a list of rater b scores
    min_rating is an optional argument describing the minimum rating possible on the data set
    max_rating is an optional argument describing the maximum rating possible on the data set
    Returns a float corresponding to the kappa correlation
    """
    assert (len(rater_a) == len(rater_b))
    rater_a = [int(a) for a in rater_a]
    rater_b = [int(b) for b in rater_b]
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    if (num_ratings > 1):
        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                  / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items

        return 1.0 - numerator / denominator
    else:
        return 1.0


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Generates a confusion matrix between rater_a and rater_b
    A confusion matrix shows how often 2 values agree and disagree
    See quadratic_weighted_kappa for argument descriptions
    """
    assert (len(rater_a) == len(rater_b))
    rater_a = [int(a) for a in rater_a]
    rater_b = [int(b) for b in rater_b]
    min_rating = int(min_rating)
    max_rating = int(max_rating)
    if min_rating is None:
        min_rating = min(rater_a)
    if max_rating is None:
        max_rating = max(rater_a)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[int(a - min_rating)][int(b - min_rating)] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Generates a frequency count of each rating on the scale
    ratings is a list of scores
    Returns a list of frequencies
    """
    ratings = [int(r) for r in ratings]
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def get_wordnet_syns(word):
    """
    Utilize wordnet (installed with nltk) to get synonyms for words

    Args:
        word (str): the word to generate synonyms for

    Returns:
        (list of str): Unique synonyms for the word
    """
    synonyms = []
    regex = r"_"
    pat = re.compile(regex)
    synset = nltk.wordnet.wordnet.synsets(word)
    for ss in synset:
        for swords in ss.lemma_names:
            synonyms.append(pat.sub(" ", swords.lower()))
    # Makes the synonym list unique
    synonyms = list(set(synonyms))
    return synonyms
