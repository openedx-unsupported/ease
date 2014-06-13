"""
Errors for the EASE repository
"""

class EaseError(Exception):
    pass


class EssaySetRequestError(EaseError):
    """
    There was a problem with a request sent to the Essay Set module.
    """
    pass


class GradingRequestError(EaseError):
    """
    There was a problem with a request sent to the Grading module.
    """
    pass


class ClassifierTrainingInternalError(EaseError):
    """
    An unexpected error occurred when training a classifier.
    """
    pass


class CreateRequestError(EaseError):
    """
    There was a problem with a request sent to the Create Module.
    """
    pass


class FeatureExtractionInternalError(EaseError):
    """
    An unexpected error occurred while extracting features from an essay.
    """
    pass


class InputError(EaseError):
    """
    The user supplied an argument which was incorrect.
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

    def __str__(self):
        "An input error occurred at '{0}': {1}".format(self.expr, self.msg)