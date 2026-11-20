"""CPU checks for the frozen evaluation's randomization and clustered CIs.

These use synthetic examples, never load study features, and do not choose
settings from evaluation performance.
"""
import unittest
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

import evaluate


class EvaluationDesignTests(unittest.TestCase):
    def test_zero_corruption_preserves_every_catalog_column(self):
        artists = ['b', 'a', 'a', 'c', 'd']
        donors, selected, attempts = evaluate.make_assignment(artists, 0, 7)
        np.testing.assert_array_equal(donors, np.arange(5))
        self.assertEqual(len(selected), 0)
        self.assertEqual(attempts, 0)

    def test_corruption_is_exact_selected_cross_artist_permutation(self):
        # Match the actual catalog size; round-half-up produces 77, not 78.
        artists = [f'artist_{i // 3}' for i in range(309)]
        donors, selected, attempts = evaluate.make_assignment(artists, .25, 4)
        expected_selected = np.sort(np.random.default_rng(4).choice(309, 77, replace=False))
        np.testing.assert_array_equal(selected, expected_selected)
        np.testing.assert_array_equal(np.sort(donors), np.arange(309))
        np.testing.assert_array_equal(np.sort(donors[selected]), selected)
        others = np.setdiff1d(np.arange(309), selected)
        np.testing.assert_array_equal(donors[others], others)
        self.assertTrue(all(artists[i] != artists[donors[i]] for i in selected))
        self.assertGreaterEqual(attempts, 1)
        self.assertLessEqual(attempts, 10000)
        repeat = evaluate.make_assignment(artists, .25, 4)
        np.testing.assert_array_equal(repeat[0], donors)
        np.testing.assert_array_equal(repeat[1], selected)
        self.assertEqual(repeat[2], attempts)

    def test_round_half_up_and_impossible_selected_set(self):
        # floor(.25*10 + .5) = 3, whereas banker's round would produce 2.
        donors, selected, _ = evaluate.make_assignment(list('abcdefghij'), .25, 11)
        self.assertEqual(len(selected), 3)
        self.assertEqual(np.count_nonzero(donors != np.arange(10)), 3)
        # Four members of one artist cannot all map to the sole other artist.
        with self.assertRaises((ValueError, RuntimeError)):
            evaluate.make_assignment(['a', 'a', 'a', 'a', 'b'], 1, 0, max_attempts=10)

    def test_random_gates_match_counts_and_share_one_nested_order(self):
        counts = {'none': 0, 'first': 2, 'mean': 5, 'all': 9}
        gates = evaluate.random_gates(counts, 9, 3)
        order = np.random.default_rng(10003).permutation(9)
        for key, rejected in counts.items():
            self.assertEqual(gates[key].dtype, np.dtype(bool))
            expected = np.ones(9, dtype=bool)
            expected[order[:rejected]] = False
            np.testing.assert_array_equal(gates[key], expected)
            self.assertEqual(np.count_nonzero(~gates[key]), rejected)
        self.assertTrue(set(np.flatnonzero(~gates['first'])).issubset(
            set(np.flatnonzero(~gates['mean']))))

    def test_fusion_gate_endpoints_preserve_scores_columns_and_float32(self):
        rng = np.random.default_rng(20261120)
        audio = rng.uniform(-1, 1, size=(7, 13)).astype(np.float32)
        text = rng.uniform(-1, 1, size=(7, 13)).astype(np.float32)
        accept = np.ones(13, dtype=bool)
        reject = np.zeros(13, dtype=bool)
        mixed = np.arange(13) % 3 == 0
        ungated = evaluate.fusion_scores(audio, text, .25)
        all_yes = evaluate.fusion_scores(audio, text, .25, accept)
        all_no = evaluate.fusion_scores(audio, text, .25, reject)
        selective = evaluate.fusion_scores(audio, text, .25, mixed)
        zero_weight = evaluate.fusion_scores(audio, text, 0, mixed)
        for result in [ungated, all_yes, all_no, selective, zero_weight]:
            self.assertEqual(result.shape, audio.shape)
            self.assertEqual(result.dtype, np.dtype(np.float32))
        # These are exact array comparisons, not approximate score equality.
        np.testing.assert_array_equal(all_yes, ungated)
        np.testing.assert_array_equal(all_no, audio)
        np.testing.assert_array_equal(selective[:, mixed], ungated[:, mixed])
        np.testing.assert_array_equal(selective[:, ~mixed], audio[:, ~mixed])
        np.testing.assert_array_equal(zero_weight, audio)

    def test_artist_bootstrap_repeats_whole_clusters_and_weights_tracks(self):
        # Cluster a has three tracks, b has one. Valid draws aa, ab/ba, bb
        # have respectively 6, 4, 2 tracks after retaining multiplicities.
        artists = ['b', 'a', 'a', 'a']
        weights = evaluate.cluster_weights(artists, 2000, 20261120)
        self.assertEqual(weights.shape, (2000, 4))
        self.assertEqual(weights.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-15)
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_array_equal(weights[:, 1], weights[:, 2])
        np.testing.assert_array_equal(weights[:, 2], weights[:, 3])
        allowed = np.array([[0, 1/3, 1/3, 1/3], [.25, .25, .25, .25], [1, 0, 0, 0]])
        present = []
        for row in allowed:
            matched = np.all(np.isclose(weights, row, atol=1e-15, rtol=0), axis=1)
            present.append(int(matched.sum()))
        self.assertEqual(sum(present), 2000)
        self.assertTrue(all(n > 0 for n in present))
        np.testing.assert_array_equal(weights,
            evaluate.cluster_weights(artists, 2000, 20261120))

    def test_percentile_interval_is_track_mean_with_97_5_level(self):
        values = np.array([1., 0., 0., 0.])
        weights = np.array([[0, 1/3, 1/3, 1/3], [.25, .25, .25, .25], [1, 0, 0, 0]])
        result = evaluate.interval(values, weights, .975)
        self.assertEqual(result['estimate'], .25)
        self.assertEqual(result['level'], .975)
        # Sorted bootstrap means [0, .25, 1]; explicit linear quantiles
        # at .0125 and .9875, not .025 and .975.
        self.assertAlmostEqual(result['lower'], .00625, places=14)
        self.assertAlmostEqual(result['upper'], .98125, places=14)

    def test_paired_contrast_has_zero_width_when_methods_match(self):
        artists = ['a', 'a', 'b', 'c', 'c', 'c']
        weights = evaluate.cluster_weights(artists, 2000, 13)
        method = np.array([1/7, 1., .5, .2, .1, 1/6])
        result = evaluate.interval(method - method, weights, .975)
        self.assertEqual(result['estimate'], 0)
        self.assertEqual(result['lower'], 0)
        self.assertEqual(result['upper'], 0)


if __name__ == '__main__':
    unittest.main()
