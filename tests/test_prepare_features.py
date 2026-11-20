"""Preprocessing boundary and model-integrity checks without GPU imports."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_features


class SourcePreparationTests(unittest.TestCase):
    def test_exact_fifty_millisecond_boundary_is_allowed(self):
        self.assertEqual(prepare_features.padding_plan(5760000-2400), (2400, True))
        self.assertEqual(prepare_features.padding_plan(5760000-2401), (0, False))
        self.assertEqual(prepare_features.padding_plan(5760000-529), (529, True))
        self.assertEqual(prepare_features.padding_plan(5760000), (0, True))

    def test_short_recording_cannot_be_padded_to_twelve_windows(self):
        # Rounding a ten-second boundary cannot make an 80s recording 120s.
        self.assertEqual(prepare_features.padding_plan(3840000-1), (1, False))
        self.assertEqual(prepare_features.padding_plan(round(119.9060625*48000)), (0, False))
        self.assertEqual(prepare_features.padding_plan(130*48000), (0, True))

    def test_bad_weight_file_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'pytorch_model.bin').write_bytes(b'not the pinned weight')
            (directory/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'weight SHA256 mismatch'):
                prepare_features.prepare_weights(directory, {'settings': {'config.json': 'unused'}}, False)


if __name__ == '__main__':
    unittest.main()
