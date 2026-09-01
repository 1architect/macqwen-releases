"""Who may reach the local server, and how the key is compared.

A browser page on any site can send a request to a server bound to
localhost. Answering every Origin would let a site you visit use your model
and read its output, so an Origin is honoured only when allowed explicitly.
"""
from __future__ import annotations

import unittest

from macqwen import server


class FakeServer:
    def __init__(self, api_key=None, allowed_origins=()):
        self.api_key = api_key
        self.allowed_origins = tuple(allowed_origins)


class FakeHandler(server.MacqwenHandler):
    """Exercise the access checks without opening a socket."""

    def __init__(self, headers, api_key=None, allowed_origins=()):
        self.headers = headers
        self.server = FakeServer(api_key, allowed_origins)
        self.errors = []

    def _error(self, message, status=400):
        self.errors.append((status, str(message)))


class OriginTests(unittest.TestCase):
    def test_no_origin_is_untouched(self):
        # curl, SDKs and every non-browser client send no Origin
        handler = FakeHandler({})
        self.assertIsNone(handler._allowed_origin())
        self.assertFalse(handler._reject_foreign_origin())

    def test_unlisted_origin_is_refused(self):
        handler = FakeHandler({"Origin": "https://evil.example"})
        self.assertTrue(handler._reject_foreign_origin())
        self.assertEqual(handler.errors[0][0], 403)

    def test_listed_origin_is_echoed_back(self):
        handler = FakeHandler({"Origin": "http://localhost:3000"},
                              allowed_origins=("http://localhost:3000",))
        self.assertFalse(handler._reject_foreign_origin())
        self.assertEqual(handler._allowed_origin(), "http://localhost:3000")

    def test_wildcard_allows_any_origin(self):
        handler = FakeHandler({"Origin": "https://anywhere.example"},
                              allowed_origins=("*",))
        self.assertFalse(handler._reject_foreign_origin())
        self.assertEqual(handler._allowed_origin(), "https://anywhere.example")

    def test_one_allowed_origin_does_not_admit_another(self):
        handler = FakeHandler({"Origin": "https://evil.example"},
                              allowed_origins=("http://localhost:3000",))
        self.assertTrue(handler._reject_foreign_origin())


class AuthorizationTests(unittest.TestCase):
    def test_no_key_configured_allows_everything(self):
        self.assertTrue(FakeHandler({}, api_key=None)._authorized())

    def test_bearer_and_x_api_key_both_work(self):
        for headers in ({"Authorization": "Bearer secret"},
                        {"x-api-key": "secret"}):
            with self.subTest(headers=headers):
                self.assertTrue(FakeHandler(headers, api_key="secret")._authorized())

    def test_wrong_key_is_refused(self):
        for headers in ({"Authorization": "Bearer wrong"},
                        {"x-api-key": "wrong"},
                        {"Authorization": "secret"},   # missing the scheme
                        {}):
            with self.subTest(headers=headers):
                self.assertFalse(FakeHandler(headers, api_key="secret")._authorized())

    def test_comparison_is_constant_time(self):
        import inspect

        source = inspect.getsource(server.MacqwenHandler._authorized)
        self.assertIn("compare_digest", source)
        self.assertNotIn("== f\"Bearer", source)


if __name__ == "__main__":
    unittest.main()
