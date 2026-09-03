from __future__ import annotations

import builtins
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from macqwen.backends.flashnext import (
    _TRANSFORMERS_ADVISORY_ENV,
    _load_transformers_tokenizer,
)


class FlashNextTransformersImportTests(unittest.TestCase):
    def test_advisory_suppression_is_scoped_to_the_import(self):
        tokenizer = object()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = tokenizer
        observed = []
        real_import = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "transformers":
                observed.append(os.environ.get(_TRANSFORMERS_ADVISORY_ENV))
            return real_import(name, *args, **kwargs)

        with patch.dict(
            os.environ, {_TRANSFORMERS_ADVISORY_ENV: "caller-value"}, clear=False
        ), patch.dict(sys.modules, {"transformers": transformers}), patch(
            "builtins.__import__", side_effect=importing
        ):
            self.assertIs(_load_transformers_tokenizer(), tokenizer)
            self.assertEqual(
                os.environ.get(_TRANSFORMERS_ADVISORY_ENV), "caller-value"
            )

        self.assertEqual(observed, ["1"])

    def test_advisory_suppression_is_restored_when_import_fails(self):
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "transformers":
                raise RuntimeError("import failed")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {}, clear=True), patch(
            "builtins.__import__", side_effect=failing_import
        ):
            with self.assertRaisesRegex(RuntimeError, "import failed"):
                _load_transformers_tokenizer()
            self.assertNotIn(_TRANSFORMERS_ADVISORY_ENV, os.environ)

    def test_real_transformers_import_does_not_emit_pytorch_advisory(self):
        code = (
            "from macqwen.backends.flashnext import "
            "_load_transformers_tokenizer; _load_transformers_tokenizer()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("PyTorch was not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
