from unittest import TestCase

from nose.tools import assert_equal
from mock import patch
from ease.util_functions import spell_correct


class SpellCheckUnitTest(TestCase):
    """
    Test that spell-check works correctly.
    """

    def test_all_correct(self):
        self._assert_spell(
            "The quick brown dog jumped over the lazy fox.",
            "The quick brown dog jumped over the lazy fox.",
            0, "The quick brown dog jumped over the lazy fox."
        )

    def test_some_misspelled(self):
        self._assert_spell(
            "The quuick brown dog jimped over the lazy fox.",
            "The quick brown dog jumped over the lazy fox.",
            2, "The <bs>quuick</bs> brown dog <bs>jimped</bs> over the lazy fox.",
        )

    def test_empty_str(self):
        self._assert_spell("", "", 0, "")

    def test_unicode(self):
        self._assert_spell(
            "The quick brown dog jimped over the \u00FCber fox.",
            "The quick brown dog jumped over the \u00FCber fox.",
            2, "The quick brown dog <bs>jimped</bs> over the \u00FCber fox.",
        )

    @patch("util_functions.os.popen")
    def test_aspell_not_found(self, popen_mock):
        # Expected behavior when aspell is not installed is to return the original
        # string with no corrections.
        popen_mock.side_effect = OSError
        self._assert_spell("Test striiing", "Test striiing", 0, "Test striiing")

    def _assert_spell(self, input_str, expected_str, expected_num_changed, expected_markup):
        result = spell_correct(input_str)
        assert_equal(result, (expected_str, expected_num_changed, expected_markup))
