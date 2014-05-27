"""
Defines an essay set object, which encapsulates essays from training and test sets.
Performs spell and grammar checking, tokenization, and stemming.
"""

import numpy
import nltk
import sys
import random
import os
import logging

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import util_functions

if not base_path.endswith("/"):
    base_path = base_path + "/"

log = logging.getLogger(__name__)

MAXIMUM_ESSAY_LENGTH = 20000


class EssaySet(object):
    def __init__(self, essaytype="train"):
        """
        Initialize variables and check essay set type
        """
        if(essaytype != "train" and essaytype != "test"):
            essaytype = "train"

        self._type = essaytype
        self._score = []
        self._text = []
        self._id = []
        self._clean_text = []
        self._tokens = []
        self._pos = []
        self._clean_stem_text = []
        self._generated = []
        self._prompt = ""
        self._spelling_errors = []
        self._markup_text = []

    def add_essay(self, essay_text, essay_score, essay_generated=0):
        """
        Add new (essay_text,essay_score) pair to the essay set.
        essay_text must be a string.
        essay_score must be an int.
        essay_generated should not be changed by the user.
        Returns a confirmation that essay was added.
        """
        # Get maximum current essay id, or set to 0 if this is the first essay added
        if(len(self._id) > 0):
            max_id = max(self._id)
        else:
            max_id = 0
            # Verify that essay_score is an int, essay_text is a string, and essay_generated equals 0 or 1

        try:
            essay_text = essay_text.encode('ascii', 'ignore')
            if len(essay_text) < 5:
                essay_text = "Invalid essay."
        except:
            log.exception("Could not parse essay into ascii.")

        try:
            # Try conversion of types
            essay_score = int(essay_score)
            essay_text = str(essay_text)
        except:
            # Nothing needed here, will return error in any case.
            log.exception("Invalid type for essay score : {0} or essay text : {1}".format(type(essay_score), type(essay_text)))

        if isinstance(essay_score, int) and isinstance(essay_text, basestring)\
                and (essay_generated == 0 or essay_generated == 1):
            self._id.append(max_id + 1)
            self._score.append(essay_score)
            # Clean text by removing non digit/work/punctuation characters
            try:
                essay_text = str(essay_text.encode('ascii', 'ignore'))
            except:
                essay_text = (essay_text.decode('utf-8', 'replace')).encode('ascii', 'ignore')
            cleaned_essay = util_functions.sub_chars(essay_text).lower()
            if(len(cleaned_essay) > MAXIMUM_ESSAY_LENGTH):
                cleaned_essay = cleaned_essay[0:MAXIMUM_ESSAY_LENGTH]
            self._text.append(cleaned_essay)
            # Spell correct text using aspell
            cleaned_text, spell_errors, markup_text = util_functions.spell_correct(self._text[len(self._text) - 1])
            self._clean_text.append(cleaned_text)
            self._spelling_errors.append(spell_errors)
            self._markup_text.append(markup_text)
            # Tokenize text
            self._tokens.append(nltk.word_tokenize(self._clean_text[len(self._clean_text) - 1]))
            # Part of speech tag text
            self._pos.append(nltk.pos_tag(self._clean_text[len(self._clean_text) - 1].split(" ")))
            self._generated.append(essay_generated)
            # Stem spell corrected text
            porter = nltk.PorterStemmer()
            por_toks = " ".join([porter.stem(w) for w in self._tokens[len(self._tokens) - 1]])
            self._clean_stem_text.append(por_toks)

            ret = "text: " + self._text[len(self._text) - 1] + " score: " + str(essay_score)
        else:
            raise util_functions.InputError(essay_text, "arguments need to be in format "
                                                        "(text,score). text needs to be string,"
                                                        " score needs to be int.")

    def update_prompt(self, prompt_text):
        """
        Update the default prompt string, which is "".
        prompt_text should be a string.
        Returns the prompt as a confirmation.
        """
        if(isinstance(prompt_text, basestring)):
            self._prompt = util_functions.sub_chars(prompt_text)
            ret = self._prompt
        else:
            raise util_functions.InputError(prompt_text, "Invalid prompt. Need to enter a string value.")
        return ret

    def generate_additional_essays(self, e_text, e_score, dictionary=None, max_syns=3):
        """
        Substitute synonyms to generate extra essays from existing ones.
        This is done to increase the amount of training data.
        Should only be used with lowest scoring essays.
        e_text is the text of the original essay.
        e_score is the score of the original essay.
        dictionary is a fixed dictionary (list) of words to replace.
        max_syns defines the maximum number of additional essays to generate.  Do not set too high.
        """
        e_toks = nltk.word_tokenize(e_text)
        all_syns = []
        for word in e_toks:
            synonyms = util_functions.get_wordnet_syns(word)
            if(len(synonyms) > max_syns):
                synonyms = random.sample(synonyms, max_syns)
            all_syns.append(synonyms)
        new_essays = []
        for i in range(0, max_syns):
            syn_toks = e_toks
            for z in range(0, len(e_toks)):
                if len(all_syns[z]) > i and (dictionary == None or e_toks[z] in dictionary):
                    syn_toks[z] = all_syns[z][i]
            new_essays.append(" ".join(syn_toks))
        for z in xrange(0, len(new_essays)):
            self.add_essay(new_essays[z], e_score, 1)
