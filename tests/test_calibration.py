"""Independent CPU tests for calibration and evaluation-data isolation.

Run from the repository root: python -m unittest discover -s tests
"""
import tempfile
import unittest
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pipeline


class CalibrationTests(unittest.TestCase):
    def test_rectangular_retrieval_uses_source_id_not_diagonal(self):
        catalog = [str(i) for i in range(100, 0, -1)]
        source_columns = [3 * i + 2 for i in range(21)]
        scores = np.zeros((21, 100), dtype=np.float64)
        scores[np.arange(21), source_columns] = 1.0
        sources = [catalog[i] for i in source_columns]
        result = pipeline.rank_metrics(scores, catalog, sources)
        self.assertEqual(result['ranks'], [1] * 21)
        self.assertEqual(result['MRR'], 1.0)
        self.assertEqual(result['R1'], 1.0)

    def test_tied_scores_use_numeric_track_id_order(self):
        # In numeric order the tied columns are IDs 2, 10, 30.
        result = pipeline.rank_metrics(np.zeros((3, 3)),
                                       ['30', '2', '10'], ['30', '2', '10'])
        self.assertEqual(result['ranks'], [3, 1, 2])
        self.assertAlmostEqual(result['MRR'], (1/3 + 1 + 1/2) / 3)

    def test_lambda_ties_choose_zero_even_if_grid_is_not_sorted(self):
        # Every score and ranking is identical at every weight.
        audio = np.zeros((100, 12, 3)); audio[:, :, 0] = 1
        texts = np.zeros((100, 3)); texts[:, 0] = 1
        result = pipeline.calibrate(audio, texts, texts[:21],
                                    [str(i) for i in range(1, 101)],
                                    [str(i) for i in range(1, 22)],
                                    [.5, .25, 0, 1])
        self.assertEqual(result['selected_lambda'], 0)
        # Equality at tau is accepted, rather than forcing exactly 90%.
        for key in ['first_window', 'normalized_12_window_mean']:
            self.assertAlmostEqual(result['thresholds'][key], 1.0)
            self.assertEqual(result['retained_fraction'][key], 1.0)

    def test_thresholds_use_all_catalog_captions_and_linear_quantile(self):
        audio = np.zeros((100, 12, 3)); audio[:, :, 0] = 1
        values = np.linspace(.005, .995, 100)
        texts = np.column_stack([values, np.sqrt(1-values**2), np.zeros(100)])
        queries = np.tile([1., 0., 0.], (21, 1))
        result = pipeline.calibrate(audio, texts, queries,
                                    [str(i) for i in range(1, 101)],
                                    [str(i) for i in range(1, 22)], [0, .5, 1])
        # Linear interpolation between scores 9 and 10 (zero-based) gives .104.
        # Taking only query tracks or rounding the quantile cannot pass.
        for key in ['first_window', 'normalized_12_window_mean']:
            self.assertAlmostEqual(result['thresholds'][key], .104, places=6)
            self.assertAlmostEqual(result['retained_fraction'][key], .90)

    def test_development_loader_does_not_open_evaluation_cache(self):
        manifest = [
            {'track_id':'10', 'split':'development', 'captions':[
                {'caption_id':'2', 'caption':'index ten'},
                {'caption_id':'20', 'caption':'query ten'}]},
            {'track_id':'999', 'split':'evaluation', 'captions':[
                {'caption_id':'1', 'caption':'evaluation sentinel'},
                {'caption_id':'3', 'caption':'evaluation sentinel query'}]},
            {'track_id':'2', 'split':'development', 'captions':[
                {'caption_id':'4', 'caption':'index two'}]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = np.zeros((12, 3)); audio[:, 0] = 1
            np.savez(directory/'10.npz', audio=audio,
                     text=np.array([[1., 0., 0.], [0., 1., 0.]]))
            np.savez(directory/'2.npz', audio=audio,
                     text=np.array([[0., 0., 1.]]))
            real_load = np.load
            opened = []
            def guarded_load(path, *args, **kwargs):
                opened.append(Path(path).stem)
                if Path(path).stem == '999':
                    raise AssertionError('Evaluation cache was accessed during development')
                return real_load(path, *args, **kwargs)
            with patch.object(pipeline.np, 'load', side_effect=guarded_load):
                result = pipeline.load_development(manifest, directory)
            self.assertEqual(set(opened), {'2', '10'})
            self.assertEqual(list(result['catalog_ids']), ['2', '10'])
            self.assertEqual(list(result['source_ids']), ['10'])
            np.testing.assert_array_equal(result['index_text'],
                                          [[0., 0., 1.], [1., 0., 0.]])
            np.testing.assert_array_equal(result['query_text'], [[0., 1., 0.]])

    def test_length_exclusion_changes_threshold_population_to_99(self):
        # One catalog-only track has been excluded before calibration. The
        # remaining 21 query tracks are unchanged; threshold calibration must
        # use 99 source captions rather than a hard-coded 100 or 21.
        audio = np.zeros((99, 12, 3)); audio[:, :, 0] = 1
        values = np.arange(99) / 100
        texts = np.column_stack([values, np.sqrt(1-values**2), np.zeros(99)])
        result = pipeline.calibrate(audio, texts, np.tile([1., 0., 0.], (21, 1)),
                                    [str(i) for i in range(1, 100)],
                                    [str(i) for i in range(1, 22)], [0, .5, 1])
        self.assertEqual(result['threshold_input_count'], 99)
        self.assertEqual(result['query_count'], 21)
        for key in ['first_window', 'normalized_12_window_mean']:
            self.assertAlmostEqual(result['thresholds'][key], .098, places=6)
            self.assertAlmostEqual(result['retained_fraction'][key], 89/99)


if __name__ == '__main__':
    unittest.main()
