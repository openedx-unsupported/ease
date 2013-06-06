import numpy
import nltk
import sys
import random
import os
import logging
import essay_set

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions

if not base_path.endswith("/"):
    base_path=base_path+"/"

log=logging.getLogger(__name__)

class PredictorSet(object):
    def __init__(self, essaytype = "train"):
        """
        Initialize variables and check essay set type
        """
        if(essaytype != "train" and essaytype != "test"):
            essaytype = "train"

        self._type = essaytype
        self._target=[]
        self._textual_features=[]
        self._numeric_features=[]
        self._essay_sets=[]

    def add_row(self, numeric_features, textual_features, target):
        #Basic input checking
        if not isinstance(target, (int, long, float)):
            error_message = "Target is not a numeric value."
            log.exception(error_message)
            raise util_functions.InputError(target, error_message)

        if not isinstance(numeric_features, list):
            error_message = "Numeric features are not a list."
            log.exception(error_message)
            raise util_functions.InputError(numeric_features, error_message)

        if not isinstance(textual_features, list):
            error_message = "Textual features are not a list."
            log.exception(error_message)
            raise util_functions.InputError(textual_features, error_message)

        #Do some length checking for parameters
        if len(self._numeric_features)>0:
            numeric_length  = len(self._numeric_features[-1])
            current_numeric_length = len(numeric_features)
            if numeric_length != current_numeric_length:
                error_message = "Numeric features are an improper length."
                log.exception(error_message)
                raise util_functions.InputError(numeric_features, error_message)

        if len(self._textual_features)>0:
            textual_length  = len(self._textual_features[-1])
            current_textual_length = len(textual_features)
            if textual_length != current_textual_length:
                error_message = "Textual features are an improper length."
                log.exception(error_message)
                raise util_functions.InputError(textual_features, error_message)

        #Now check to see if text features and numeric features are individually correct

        for i in xrange(0,len(numeric_features)):
            try:
                numeric_features[i] = float(numeric_features[i])
            except:
                error_message = "Numeric feature {0} not numeric.".format(numeric_features[i])
                log.exception(error_message)
                raise util_functions.InputError(numeric_features, error_message)


        for i in xrange(0,len(textual_features)):
            try:
                textual_features[i] = str(textual_features[i].encode('ascii', 'ignore'))
            except:
                error_message = "Textual feature {0} not string.".format(textual_features[i])
                log.exception(error_message)
                raise util_functions.InputError(textual_features, error_message)

        #Create essay sets for textual features if needed
        if len(self._textual_features)==0:
            for i in xrange(0,len(textual_features)):
                self._essay_sets.append(essay_set.EssaySet(essaytype=self._type))

        #Add numeric and textual features
        self._numeric_features.append(numeric_features)
        self._textual_features.append(textual_features)

        #Add targets
        self._target.append(target)

        #Add textual features to essay sets
        for i in xrange(0,len(textual_features)):
            self._essay_sets[i].add_essay(textual_features[i], target)

