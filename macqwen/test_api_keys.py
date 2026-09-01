from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macqwen.api_keys import KeyStore, sanitized_environment


class KeyStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "Application Support" / "api_keys.json"

    def tearDown(self):
        self.directory.cleanup()

    def test_set_creates_private_application_data(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TAVILY_API_KEY", None)
            store = KeyStore(self.path)
            store.set("tavily", "private-value")
            self.assertEqual(json.loads(self.path.read_text())["tavily"], "private-value")
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(os.environ["TAVILY_API_KEY"], "private-value")
            self.assertNotIn("private-value", store.status())

    def test_saved_key_loads_into_environment(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"context7": "saved-value"}))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTEXT7_API_KEY", None)
            KeyStore(self.path)
            self.assertEqual(os.environ["CONTEXT7_API_KEY"], "saved-value")

    def test_existing_store_permissions_are_corrected(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{}")
        os.chmod(self.path.parent, 0o755)
        os.chmod(self.path, 0o644)
        KeyStore(self.path)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_delete_restores_external_value(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "external-value"}):
            store = KeyStore(self.path)
            store.set("tavily", "saved-value")
            self.assertTrue(store.delete("tavily"))
            self.assertEqual(os.environ["TAVILY_API_KEY"], "external-value")

    def test_unknown_service_is_rejected(self):
        store = KeyStore(self.path)
        with self.assertRaises(ValueError):
            store.set("unknown", "value")

    def test_child_environment_excludes_secrets(self):
        with patch.dict(
            os.environ,
            {"NORMAL_VALUE": "keep", "PRIVATE_API_KEY": "remove", "LOGIN_TOKEN": "remove"},
        ):
            environment = sanitized_environment()
        self.assertEqual(environment["NORMAL_VALUE"], "keep")
        self.assertNotIn("PRIVATE_API_KEY", environment)
        self.assertNotIn("LOGIN_TOKEN", environment)


if __name__ == "__main__":
    unittest.main()
