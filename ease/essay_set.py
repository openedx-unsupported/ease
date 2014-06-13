"""
Defines an essay set object, which encapsulates essays from training and test sets.
Performs spell and grammar checking, tokenization, and stemming.
"""

import random
import os
import logging

import nltk
import sys
from errors import *


base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions

if not base_path.endswith("/"):
    base_path += "/"

log = logging.getLogger(__name__)

MAXIMUM_ESSAY_LENGTH = 20000


class EssaySet(object):
    """
    The essay set object which encapsulates essays into sets for two purposes:
        Testing
        Training
    Additionally, the addition of essays into one of these sets performs all spell/grammar
    checking, tokenization of the essay, and stemming.

    Essays in an essay set can be assumed to have these properties.
    """

    def __init__(self, essay_type="train"):
        """
        Initialize variables and check essay set type

        Args:
            essay_type (string): Either 'train' or 'grade', defines the type of the essay set.
                                    If not recognized, we default to "train"
        """

        if essay_type != "train" and essay_type != "test":
            essay_type = "train"

        self._type = essay_type
        self._scores = []
        self._cleaned_essays = []
        self._ids = []
        self._cleaned_spelled_essays = []
        self._tokens = []
        self._pos_tags = []
        self._cleaned_stemmed_essays = []
        self._generated = []
        self._prompt = ""
        self._spelling_errors = []
        self._markup_text = []

    def add_essay(self, essay_text, essay_score, essay_generated=0):
        """
        Adds a new pair of (essay_text, essay_score) to the essay set.

        In the context of training, this occurs when a human creates another example
        for the AI assessment to be based on

        NOTE:
        essay_generated should not be changed by the user.

        Args:
            essay_text (string): The text of the essay
            essay_score (int): The score assigned to the essay by a human.

        Kwargs:
            essay_generated (int):

        Returns:
            A string confirmation that essay was added.

        Raises
            EssaySetRequestError
        """

        # Get maximum current essay id (the newest essay), or set to 0 if this is the first essay added
        if len(self._ids) > 0:
            max_id = max(self._ids)
        else:
            max_id = 0

        # Encodes the essay into ascii.  Note that un-recognized characters will be ignored
        # Also note that if we first fail to encode, we will try to decode from utf-8 then encode.
        try:
            essay_text = essay_text.encode('ascii', 'ignore')
        except UnicodeError:
            try:
                essay_text = (essay_text.decode('utf-8', 'replace')).encode('ascii', 'ignore')
            except UnicodeError as ex:
                msg = "Could not parse essay text into ascii: {}".format(ex)
                log.exception(msg)
                raise EssaySetRequestError(msg)

        # Validates that score is an integer and essay_text is a string and essay_generated is a 0 or a 1.
        try:
            essay_score = int(essay_score)
            essay_text = str(essay_text)
            essay_generated = int(essay_generated)
            bool(essay_generated)
        except TypeError:
            ex = "Invalid type for essay score : {0} or essay text : {1}".format(type(essay_score), type(essay_text))
            log.exception(ex)
            raise EssaySetRequestError(ex)

        # Validates to make sure that the essay is at least five characters long.
        if len(essay_text) < 5:
            essay_text = "Invalid essay."

        # If we reach this point, we are not going to raise an exception beyond it, so we can add any and all
        # variables to our lists while maintaining internal consistency.  This is a new fix as of 6-12-14 GBW

        # Assigns a new ID to the essay, adds fields passed in.
        self._ids.append(max_id + 1)
        self._scores.append(essay_score)
        self._generated.append(essay_generated)

        # Cleans text by removing non digit/work/punctuation characters
        cleaned_essay = util_functions.sub_chars(essay_text).lower()
        # Checks to see if the essay is longer than we allow. Truncates if longer
        if len(cleaned_essay) > MAXIMUM_ESSAY_LENGTH:
            cleaned_essay = cleaned_essay[0:MAXIMUM_ESSAY_LENGTH]
        self._cleaned_essays.append(cleaned_essay)

        # Spell correct text using aspell
        cleaned_spelled_essay, spell_errors, markup_text = util_functions.spell_correct(cleaned_essay)
        self._cleaned_spelled_essays.append(cleaned_spelled_essay)
        self._spelling_errors.append(spell_errors)
        self._markup_text.append(markup_text)

        # Create tokens for the text and part of speech tags
        tokens = nltk.word_tokenize(cleaned_spelled_essay)
        pos_tags = nltk.pos_tag(cleaned_spelled_essay.split(" "))
        self._tokens.append(tokens)
        self._pos_tags.append(pos_tags)

        # Applies Porter stemming algorithm, a process for removing the commoner morphological and inflexional endings
        # from words in English.
        porter = nltk.PorterStemmer()
        porter_tokens = " ".join([porter.stem(token) for token in tokens])
        self._cleaned_stemmed_essays.append(porter_tokens)

        return "Essay Added. Text: " + cleaned_essay + " Score: " + str(essay_score)

    def update_prompt(self, prompt_text):
        """
        Updates the default prompt (an empty string) to a user specified string

        Args:
            prompt_text (str): the value to set the prompt to

        Returns:
            (str): The prompt, if it was stored successfully.

        Raises:
            InputError
        """
        if isinstance(prompt_text, basestring):
            self._prompt = util_functions.sub_chars(prompt_text)
        else:
            raise InputError('prompt_text', "Invalid prompt. Need to enter a string value.")
        return self._prompt

    def generate_additional_essays(self, original_essay, original_score, to_generate=3):
        """
        Generates and adds additional essays to the essay set from a base essay by substituting synonyms.

        Args:
            original_essay (str): The original essay to generate off of.
            original_score (int): The integer score assigned to the input essay.

        Kwargs:
            FEATURE REMOVED (GBW): dictionary (dict): A static dictionary of words to replace. Defaults to none.
                                    Feature was removed because it was not implemented fully to begin with.
            to_generate (int): The number of additional essays to generate based on synonym substitution
        """

        original_tokens = nltk.word_tokenize(original_essay)
        synonym_matrix = []

        # Iterates through the words in the original essay
        for word in original_tokens:
            synonyms = util_functions.get_wordnet_syns(word)
            # Only substitute on a token if one could generate N=max_syns unique essays on that token.
            if len(synonyms) > to_generate:
                # Adds one word on to the list of synonyms, one for each of the new essays
                synonyms = random.sample(synonyms, to_generate)
            synonym_matrix.append(synonyms)

        new_essays = []
        # Generates each essay
        for i in range(0, to_generate):
            # Start out from the same base essay
            new_tokens = original_tokens
            for z in range(0, len(original_tokens)):
                # Replace a given token ONLY if it is not the first token in the dictionary??!?!?!!?!
                if len(synonym_matrix[z]) > i:
                    new_tokens[z] = synonym_matrix[z][i]
            new_essays.append(" ".join(new_tokens))

        # Adds each new essay to the list of essays in this essay set
        for i in xrange(0, len(new_essays)):
            self.add_essay(new_essays[i], original_score, 1)
