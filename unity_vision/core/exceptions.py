class UnityVisionException(Exception):
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

class UnrecognizedAuthException(UnityVisionException):
    """Raise when unknown authentiction methods are used"""
    pass

class AuthenticationException(UnityVisionException):
    """Raised when no Auth Token is provided"""
    pass

class TimeoutException(UnityVisionException):
    """Raised when request times-out."""
    pass

class MalformedQueryException(UnityVisionException):
    pass

class DatasetException(UnityVisionException):
    pass
