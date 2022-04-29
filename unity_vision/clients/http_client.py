# from typing import Optional
#
# import requests
#
# from unity_vision.core.auth.auth import Authenticator
#
# BASE_URI_V1 = "https://perception-api.simulation.unity3d.com"
# DEFAULT_TIMEOUT = 1800  # in seconds
# DEFAULT_MAX_RETRIES = 5
#
#
# class HttpClient:
#     methods = {"get", "post", "put", "patch", "delete"}
#
#     def __init__(
#         self, api_version, auth, headers=None
#     ):
#         self.api_version = api_version
#         if headers is None:
#             self.headers = {}
#         self.headers = headers
#         self.auth = auth
#
#     def make_request(self, req):
#         res = requests.request(auth=self.auth, **req)
#         return res
