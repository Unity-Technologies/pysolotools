class PySoloException(Exception):
    """Base class for Exceptions"""

    def __init__(self, message, source=None):
        """
        Args:
            message (str): Description of exception
            source (Exception): The actual exception thrown.
        """

        super().__init__(message, source)
        self.message = message
        self.source = source


class UnrecognizedAuthException(PySoloException):
    """Raise when unknown authentiction methods are used"""

    pass


class UCVDException(PySoloException):
    pass


class AuthenticationException(PySoloException):
    """Raised when no Auth Token is provided"""

    pass


class TimeoutException(PySoloException):
    """Raised when request times-out."""

    pass


class MalformedQueryException(PySoloException):
    pass


class DatasetException(PySoloException):
    pass


class MissingCaptureException(PySoloException):
    """Raise when capture is missing in the dataset."""

    pass


class MissingKeypointAnnotatorException(PySoloException):
    """Raised when keypoint annotator is missing"""

    pass


class DatasetNotFoundException(PySoloException):
    pass
