==================================
Usage
==================================

This repo offers the ability to create models and to grade new text.  There are additional lower-level functions, as well.

Essay Grading
-------------------------------------

Essay grading can be done via the "grade" function in grade.py and the "create" function in create.py.  Call the create function, and pass in the appropriate data (see documentation there), in order to obtain a created model.  That model can then be used in conjunction with the "grade" function to get scores for new text.

Arbitrary sets of predictors and text scoring
----------------------------------------------------

This repo can also be used to compute scores for arbitrary sets of numeric and textual predictors.  For example, you could predict whether the stock market will rise or fall tomorrow by passing in a set of article headlines, article text, and publication times.  Use the functions "create_generic" in create.py and "grade_generic" in grade.py to do this.

