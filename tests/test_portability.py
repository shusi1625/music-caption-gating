"""Checks introduced by the portable reproduction entry point."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reproduce


class PortableReproductionTests(unittest.TestCase):
    def test_outputs_cannot_overlap_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'input'
            data.mkdir()
            for output in [data, data/'outputs', root]:
                with self.assertRaises(ValueError):
                    reproduce.separate_directories(data, output)
            actual_data, actual_output = reproduce.separate_directories(data, root/'output')
            self.assertEqual(actual_data, data.resolve())
            self.assertEqual(actual_output, (root/'output').resolve())

    def test_reference_comparison_does_not_hide_changed_ranks_or_metrics(self):
        original = {'ranks': [1, 3, 12], 'MRR': .25}
        reproduce.compare_values({'ranks': [1, 3, 12], 'MRR': .25+1e-15}, original)
        with self.assertRaises(AssertionError):
            reproduce.compare_values({'ranks': [1, 3, 11], 'MRR': .25}, original)
        with self.assertRaises(AssertionError):
            reproduce.compare_values({'ranks': [1, 3, 12], 'MRR': .250001}, original)


if __name__ == '__main__':
    unittest.main()
