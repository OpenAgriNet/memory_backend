"""e5 prefixes: without them the vectors still have the right shape, just worse ones."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import embedding  # noqa: E402


class _Echo:
    def embed(self, texts):
        return [type("V", (), {"tolist": lambda self, t=t: t})() for t in texts]


class EmbeddingPrefixTest(unittest.TestCase):
    def test_kind_selects_e5_prefix(self):
        with patch.object(embedding, "_model", return_value=_Echo()):
            self.assertEqual(embedding.embed(["x"], "document"), ["passage: x"])
            self.assertEqual(embedding.embed(["x"], "query"), ["query: x"])
            with self.assertRaises(ValueError):
                embedding.embed(["x"], "other")


if __name__ == "__main__":
    unittest.main()
