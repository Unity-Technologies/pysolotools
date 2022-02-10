from abc import ABC, abstractmethod


class Authenticator(ABC):
    """Interface defining the methods allowed for authentication."""

    AUTH_BASIC = "Basic"
    AUTH_BEARERTOKEN = "Bearer"
    AUTH_NONE = "None"

    @abstractmethod
    def authenticate(self, req: dict) -> None:
        """Authentication steps necessary for the selected authentication scheme.

            AUTH_BASIC: "Basic <token>"
            AUTH_BEARER: "Bearer <token>"
            AUTH_NONE: None

            For <token> please get the service account key and the api secret from Unity Dashboard.

        Attributes:
            req (dict): This request will be updated with the required auth headers

        To be implemented by subclass
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validates the current authentication scheme configuration

        Raises:
            ValueError: In case the given configuration is not valid for
                        the selected authentication scheme.

        To be implemented by subclass.
        """
        pass

    @abstractmethod
    def authentication_type(self) -> str:
        """Returns the selected authentication scheme.

        To be implemented by subclass.
        """
        return Authenticator.AUTH_NONE
