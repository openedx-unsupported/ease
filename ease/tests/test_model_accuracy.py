import unittest
import os
import random
import logging
import json

from ease import create, grade


log = logging.getLogger(__name__)

ROOT_PATH = os.path.abspath(__file__)
TEST_PATH = os.path.abspath(os.path.join(ROOT_PATH, ".."))

CHARACTER_LIMIT = 1000
TRAINING_LIMIT = 50
QUICK_TEST_LIMIT = 5

# noinspection PyClassHasNoInit
class DataLoader():
    @staticmethod
    def load_text_files(pathname):
        filenames = os.listdir(pathname)
        text = []
        for filename in filenames:
            data = open(os.path.join(pathname, filename)).read()
            text.append(data[:CHARACTER_LIMIT])
        return text

    @staticmethod
    def load_json_file(filename):
        datafile = open(os.path.join(filename))
        data = json.load(datafile)
        return data

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
        directories = [os.path.abspath(os.path.join(self.pathname, f)) for f in filenames if
                       not os.path.isfile(os.path.join(self.pathname, f)) and f in ["neg", "pos"]]

        # Sort so neg is first
        directories.sort()
        # We need to have both a postive and a negative folder to classify
        if len(directories) != 2:
            raise Exception("Need a pos and a neg directory in {0}".format(self.pathname))

        neg = self.load_text_files(directories[0])
        pos = self.load_text_files(directories[1])

        scores = [0 for i in xrange(0, len(neg))] + [1 for i in xrange(0, len(pos))]
        text = neg + pos

        return scores, text


class JSONLoader(DataLoader):
    def __init__(self, pathname):
        self.pathname = pathname

    def load_data(self):
        filenames = os.listdir(self.pathname)
        files = [os.path.abspath(os.path.join(self.pathname, f)) for f in filenames if
                 os.path.isfile(os.path.join(self.pathname, f)) if f.endswith(".json")]

        files.sort()
        # We need to have both a postive and a negative folder to classify
        if len(files) == 0:
            return [], []

        data = []
        for f in files:
            f_data = self.load_json_file(f)
            data.append(f_data)

        all_scores = []
        all_text = []
        for i in xrange(0, len(data)):
            scores = [d['score'] for d in data[i]]
            text = [d['text'] for d in data[i]]

            if isinstance(scores[0], list):
                new_text = []
                new_scores = []
                for j in xrange(0, len(scores)):
                    text = scores[j]
                    s = scores[j]
                    for k in s:
                        new_text.append(text)
                        new_scores.append(k)
                text = new_text
                scores = new_scores

            all_scores.append(scores)
            all_text.append(text)

        return all_scores, all_text


class ModelCreator():
    def __init__(self, scores, text):
        self.scores = scores
        self.text = text

        # Governs which creation function in the ease.create module to use.  See module for info.
        if isinstance(text, list):
            self.create_model_generic = False
        else:
            self.create_model_generic = True

    def create_model(self):
        if not self.create_model_generic:
            return create.create(self.text, self.scores, "")
        else:
            return create.create_generic(self.text.get('numeric_values', []), self.text.get('textual_values', []),
                                         self.scores)


class Grader():
    def __init__(self, model_data):
        self.model_data = model_data

    def grade(self, submission):
        if isinstance(submission, basestring):
            return grade.grade(self.model_data, submission)
        else:
            return grade.grade_generic(self.model_data, submission.get('numeric_values', []),
                                       submission.get('textual_values', []))


class GenericTest(object):
    loader = DataLoader
    data_path = ""
    expected_kappa_min = 0
    expected_mae_max = 0


    def load_data(self):
        data_loader = self.loader(os.path.join(TEST_PATH, self.data_path))
        scores, text = data_loader.load_data()
        return scores, text

    def generic_setup(self, scores, text):
        # Shuffle to mix up the classes, set seed to make it repeatable
        random.seed(1)
        shuffled_scores = []
        shuffled_text = []
        indices = [i for i in xrange(0, len(scores))]
        random.shuffle(indices)
        for i in indices:
            shuffled_scores.append(scores[i])
            shuffled_text.append(text[i])

        self.text = shuffled_text[:TRAINING_LIMIT]
        self.scores = shuffled_scores[:TRAINING_LIMIT]

    def model_creation_and_grading(self):
        score_subset = self.scores[:QUICK_TEST_LIMIT]
        text_subset = self.text[:QUICK_TEST_LIMIT]
        model_creator = ModelCreator(score_subset, text_subset)
        results = model_creator.create_model()
        assert results['success'] == True

        grader = Grader(results)
        results = grader.grade(self.text[0])
        assert results['success'] == True

    def scoring_accuracy(self):
        random.seed(1)
        model_creator = ModelCreator(self.scores, self.text)
        results = model_creator.create_model()
        assert results['success'] == True
        cv_kappa = results['cv_kappa']
        cv_mae = results['cv_mean_absolute_error']
        assert cv_kappa >= self.expected_kappa_min
        assert cv_mae <= self.expected_mae_max

    def generic_model_creation_and_grading(self):
        log.info(self.scores)
        log.info(self.text)
        score_subset = [random.randint(0, 100) for i in xrange(0, min([QUICK_TEST_LIMIT, len(self.scores)]))]
        text_subset = self.text[:QUICK_TEST_LIMIT]
        text_subset = {
            'textual_values': [[t] for t in text_subset],
            'numeric_values': [[1] for i in xrange(0, len(text_subset))]
        }
        model_creator = ModelCreator(score_subset, text_subset)
        results = model_creator.create_model()
        assert results['success'] == True

        grader = Grader(results)
        test_text = {
            'textual_values': [self.text[0]],
            'numeric_values': [1]
        }
        results = grader.grade(test_text)
        assert results['success'] == True


class PolarityTest(unittest.TestCase, GenericTest):
    loader = PolarityLoader
    data_path = "data/polarity"

    # These will increase if we allow more data in.
    # I am setting the amount of data low to allow tests to finish quickly (40 training essays, 1000 character max for each)
    expected_kappa_min = -.2
    expected_mae_max = 1

    def setUp(self):
        scores, text = self.load_data()
        self.generic_setup(scores, text)

    def test_model_creation_and_grading(self):
        self.model_creation_and_grading()

    def test_scoring_accuracy(self):
        self.scoring_accuracy()

    def test_generic_model_creation_and_grading(self):
        self.generic_model_creation_and_grading()


class JSONTest(GenericTest):
    loader = JSONLoader
    data_path = "data/json_data"

    # These will increase if we allow more data in.
    # I am setting the amount of data low to allow tests to finish quickly (40 training essays, 1000 character max for each)
    expected_kappa_min = -.2
    expected_mae_max = 1

    def setUp(self):
        self.scores, self.text = self.load_data()
        return self.scores, self.text


def test_loop():
    json_test = JSONTest()
    scores, text = json_test.setUp()
    for i in xrange(0, len(scores)):
        json_test.generic_setup(scores[i], text[i])
        yield json_test.model_creation_and_grading
        yield json_test.scoring_accuracy
        yield json_test.generic_model_creation_and_grading

