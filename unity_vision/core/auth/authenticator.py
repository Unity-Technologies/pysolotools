from abc import ABC, abstractmethod

class Authenticator(ABC):
    AUTH_BASIC = 'Basic'
    AUTH_BEARERTOKEN = 'Bearer'
    AUTH_NONE = 'None'

    @abstractmethod
    def authenticate(self, req: dict):
        pass

    @abstractmethod
    def validate(self) -> None:
        pass

    @abstractmethod
    def authentication_type(self) -> str:
        return Authenticator.AUTH_NONE