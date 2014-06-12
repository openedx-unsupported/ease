"""
Defines a predictor set, which is a way of taking textual and numerical data and computing it into a format which
can be used by a ML algorithm to generate objects necessary to grade future essays.
"""

import sys
import os
import logging
import essay_set

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions

if not base_path.endswith("/"):
    base_path = base_path + "/"

log = logging.getLogger(__name__)


class PredictorSet(object):
    """
    The Predictor Set Class
    """
    # TODO This class is wildly incomplete.

    def __init__(self, essay_type="train"):
        """
        Instantiates a new predictor set, which will be used to place data into for classifier training.

        Args:
            essay_type (str): Either 'train' or 'test', indicating whether the essays are meant to be trained or ar
                            in test mode.  If nothing (or anything we don't recognize) is specified, default to train.
        """
        if essay_type != "train" and essay_type != "test":
            essay_type = "train"

        self._type = essay_type
        self._target = []
        self._textual_features = []
        self._numeric_features = []
        self._essay_sets = []

    def add_row(self, numeric_features, textual_features, target):
        """
        Adds a row to the Predictor set from numeric_features, textual_features, and a target.
        """
        #TODO This docstring

        # Type input checking
        if not isinstance(target, (int, long, float)):
            raise log_error(target, "Argument target was not entered as a numeric value.")

        if not isinstance(numeric_features, list):
            raise log_error(numeric_features, "Argument numeric_features must be a list of numeric data.")

        if not isinstance(textual_features, list):
            raise log_error(textual_features, "Argument textual_features must be a list of textual data")

        # Make sure the feature sets we are trying to add are of the same length as previous sets
        if len(self._numeric_features) > 0:
            current_numeric_length = len(self._numeric_features[-1])
            if len(numeric_features) != current_numeric_length:
                raise log_error(numeric_features, "Numeric features are an improper length.")

        if len(self._textual_features) > 0:
            current_textual_length = len(self._textual_features[-1])
            if len(textual_features) != current_textual_length:
                raise log_error(textual_features, "Textual features are an improper length.")

        # Now check to see if text features and numeric features are individually of the right type
        for i in xrange(0, len(numeric_features)):
            try:
                numeric_features[i] = float(numeric_features[i])
            except TypeError:
                raise log_error(numeric_features, "Numeric feature {0} not numeric.".format(numeric_features[i]))

        for i in xrange(0, len(textual_features)):
            try:
                textual_features[i] = str(textual_features[i].encode('ascii', 'ignore'))
            except TypeError:
                raise log_error(textual_features, "Textual feature {0} not numeric.".format(textual_features[i]))
            except UnicodeError:
                raise log_error(textual_features,"Textual feature {} could not be decoded.".format(textual_features[i]))

        # Create essay sets for textual features if needed
        # TODO Understand this logic and change it, I don't think it is right.
        if len(self._textual_features) == 0:
            for i in xrange(0, len(textual_features)):
                self._essay_sets.append(essay_set.EssaySet(essay_type=self._type))

        # Add numeric and textual features
        self._numeric_features.append(numeric_features)
        self._textual_features.append(textual_features)

        # Add targets
        self._target.append(target)

        # Add textual features to essay sets
        for i in xrange(0, len(textual_features)):
            self._essay_sets[i].add_essay(textual_features[i], target)

def log_error(self, error_name, error_message):
    """
    A helper method to avoid redundancy.  Logs an error and returns it to be raised.
    """
    log.exception(error_message)
    return util_functions.InputError(error_name, error_message)
