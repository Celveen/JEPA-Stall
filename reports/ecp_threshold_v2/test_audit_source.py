import unittest

from analyze_ada045 import recorded_source_hash


class SourceReceiptTests(unittest.TestCase):
    def test_preflight_and_queue_source_key_formats_are_both_accepted(self):
        expected = 'abc123'
        old = {'source_sha256': {'planning/stagnation.py': expected}}
        new = {'source_sha256': {'stagnation.py': expected}}
        self.assertEqual(recorded_source_hash(old, 'planning/stagnation.py'), expected)
        self.assertEqual(recorded_source_hash(new, 'planning/stagnation.py'), expected)


if __name__ == '__main__':
    unittest.main()
