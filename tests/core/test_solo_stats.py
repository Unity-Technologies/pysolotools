import unittest

from pysolo.consumers import Solo
from pysolo.core.stats import SoloStats


class TestFramesIterator(unittest.TestCase):
    solo = Solo("tests/data/solo")

    def test_stats(self):
        self.assertIsInstance(self.solo.stats, SoloStats)

    def test_get_categories(self):
        cats = self.solo.stats.get_categories()
        self.assertEqual(
            cats, {1: "Crate", 2: "Cube", 3: "Box", 4: "Terrain", 5: "Character"}
        )
