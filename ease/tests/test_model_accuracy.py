import unittest
import os
from ease import create, grade
import random
import logging

log = logging.getLogger(__name__)

ROOT_PATH = os.path.abspath(__file__)
TEST_PATH = os.path.abspath(os.path.join(ROOT_PATH, ".."))

CHARACTER_LIMIT = 1000
TRAINING_LIMIT = 100
QUICK_TEST_LIMIT = 5

class DataLoader():
    def load_text_files(self, pathname):
        filenames = os.listdir(pathname)
        text = []
        for filename in filenames:
            data = open(os.path.join(pathname, filename)).read()
            text.append(data[:CHARACTER_LIMIT])
        return text

    def load_data(self):
        """
        Override when inheriting
        """
        pass

class PolarityLoader(DataLoader):
    def __init__(self, pathname):
        self.pathname = pathname

    def load_data(self):
        filenames = os.listdir(self.pathname)
        directories = [os.path.abspath(os.path.join(self.pathname,f)) for f in filenames if not os.path.isfile(os.path.join(self.pathname,f)) and f in ["neg", "pos"]]

        #Sort so neg is first
        directories.sort()
        #We need to have both a postive and a negative folder to classify
        if len(directories)!=2:
            raise Exception("Need a pos and a neg directory in {0}".format(self.pathname))

        neg = self.load_text_files(directories[0])
        pos = self.load_text_files(directories[1])

        scores = [0 for i in xrange(0,len(neg))] + [1 for i in xrange(0,len(pos))]
        text = neg + pos

        return scores, text

class ModelCreator():
    def __init__(self, scores, text):
        self.scores = scores
        self.text = text

        #Governs which creation function in the ease.create module to use.  See module for info.
        if isinstance(text[0], basestring):
            self.create_model_generic = False
        else:
            self.create_model_generic = True

    def create_model(self):
        if not self.create_model_generic:
            return create.create(self.text, self.scores, "")
        else:
            return create.create_generic(self.text.get('numeric_values', []), self.text.get('textual_values', []), self.scores)

class Grader():
    def __init__(self, model_data):
        self.model_data = model_data

    def grade(self, submission):
        if isinstance(submission, basestring):
            return grade.grade(self.model_data, submission)
        else:
            return grade.grade_generic(self.model_data, submission.get('numeric_features', []), submission.get('textual_features', []))

class GenericTest(object):
    loader = DataLoader
    data_path = ""
    expected_kappa_min = 0
    expected_mae_max = 0

    def generic_setup(self):
        data_loader = self.loader(os.path.join(TEST_PATH, self.data_path))
        scores, text = data_loader.load_data()

        #Shuffle to mix up the classes, set seed to make it repeatable
        random.seed(1)
        shuffled_scores = []
        shuffled_text = []
        indices = [i for i in xrange(0,len(scores))]
        random.shuffle(indices)
        for i in indices:
            shuffled_scores.append(scores[i])
            shuffled_text.append(text[i])

        self.text = shuffled_text[:TRAINING_LIMIT]
        self.scores = shuffled_scores[:TRAINING_LIMIT]

    def test_model_creation_and_grading(self):
        score_subset = self.scores[:QUICK_TEST_LIMIT]
        text_subset = self.text[:QUICK_TEST_LIMIT]
        model_creator = ModelCreator(score_subset, text_subset)
        results = model_creator.create_model()
        self.assertTrue(results['success'])

        grader = Grader(results)
        grader.grade(self.text[0])
        self.assertTrue(results['success'])

    def test_scoring_accuracy(self):
        random.seed(1)
        model_creator = ModelCreator(self.scores, self.text)
        results = model_creator.create_model()
        self.assertTrue(results['success'])
        cv_kappa = results['cv_kappa']
        cv_mae = results['cv_mean_absolute_error']
        self.assertGreaterEqual(cv_kappa, self.expected_kappa_min)
        self.assertLessEqual(cv_mae, self.expected_mae_max)

class PolarityTest(unittest.TestCase,GenericTest):
    loader = PolarityLoader
    data_path = "data/polarity"

    #These will increase if we allow more data in.
    #I am setting the amount of data low to allow tests to finish quickly (40 training essays, 1000 character max for each)
    expected_kappa_min = .26
    expected_mae_max = .2

    def setUp(self):
        self.generic_setup()
