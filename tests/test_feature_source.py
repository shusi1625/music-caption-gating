"""Metadata-only safeguards for private regenerated embeddings."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from feature_source import MODEL_WEIGHT_SHA256, canonical, digest, feature_source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class FeatureSourceTests(unittest.TestCase):
    def fixture(self, root):
        data, generated = root/'input', root/'features'
        config = {'model': 'laion/clap-htsat-unfused', 'revision': 'fixed-test-revision'}
        row = {'track_id': '1', 'split': 'development', 'artist_id': '10',
               'captions': [{'caption_id': '1', 'caption': 'source caption'}],
               'lineage': {'audio_sha256': 'a'*64}}
        excluded = {'track_id': '2', 'split': 'evaluation', 'artist_id': '20',
                    'captions': [{'caption_id': '2', 'caption': 'short recording'}],
                    'reason': 'fewer_than_twelve_complete_windows',
                    'original_seconds': 100, 'padding_samples': 0}
        write(data/'minimum_experiment.json', config)
        write(data/'cache/active_manifest.json', [row])
        write(data/'cache/exclusions.json', [excluded])
        write(data/'data/experiment_manifest.json', [row, excluded])
        write(data/'frozen_development_settings.json', {'protocol_sha256': digest(data/'minimum_experiment.json')})
        write(data/'development_calibration.json', {
            'active_manifest_sha256': digest(data/'cache/active_manifest.json'),
            'exclusions_sha256': digest(data/'cache/exclusions.json')})
        spec = {**config, 'sample_rate': 48000, 'windows': 12, 'window_samples': 480000}
        fresh = deepcopy(row)
        fresh['lineage'].update({'spec': canonical(spec), 'captions': row['captions']})
        fresh.update({'cache_key': canonical(fresh['lineage']), 'npz_sha256': 'b'*64})
        write(generated/'manifest.json', [fresh])
        write(generated/'clap/1.json', fresh)
        write(generated/'exclusions.json', [excluded])
        write(generated/'model_spec.json', spec)
        write(generated/'provenance.json', {
            'original_protocol_sha256': digest(data/'minimum_experiment.json'),
            'original_active_manifest_sha256': digest(data/'cache/active_manifest.json'),
            'original_frozen_settings_sha256': digest(data/'frozen_development_settings.json'),
            'model_weight_sha256': MODEL_WEIGHT_SHA256, **config})
        return data, generated, fresh

    def test_metadata_validation_leaves_input_unchanged_and_fixes_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            data, generated, row = self.fixture(Path(directory))
            original_bytes = (data/'frozen_development_settings.json').read_bytes()
            source = feature_source(data, generated)
            self.assertEqual(source['rows'], [row])
            self.assertEqual(source['cache_dir'], generated/'clap')
            self.assertEqual(source['details']['score_absolute_tolerance_to_original'], 1e-5)
            self.assertEqual(source['details']['metric_absolute_tolerance_to_original'], 2e-14)
            self.assertTrue(source['details']['ranks_gates_assignments_must_match_original_exactly'])
            self.assertEqual((data/'frozen_development_settings.json').read_bytes(), original_bytes)
            # No NPZ exists: this helper must not inspect evaluation features.
            self.assertFalse((generated/'clap/1.npz').exists())

    def test_regeneration_cannot_change_caption_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            data, generated, row = self.fixture(Path(directory))
            row['captions'][0]['caption'] = 'changed caption'
            write(generated/'manifest.json', [row])
            write(generated/'clap/1.json', row)
            with self.assertRaises(AssertionError):
                feature_source(data, generated)

    def test_regeneration_cannot_drop_selected_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            data, generated, _ = self.fixture(Path(directory))
            write(generated/'manifest.json', [])
            with self.assertRaises(AssertionError):
                feature_source(data, generated)

    def test_regeneration_cannot_change_model_weight_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            data, generated, _ = self.fixture(Path(directory))
            provenance = json.loads((generated/'provenance.json').read_text())
            provenance['model_weight_sha256'] = 'f'*64
            write(generated/'provenance.json', provenance)
            with self.assertRaises(AssertionError):
                feature_source(data, generated)

    def test_regeneration_cannot_change_duration_exclusion_population(self):
        with tempfile.TemporaryDirectory() as directory:
            data, generated, _ = self.fixture(Path(directory))
            write(generated/'exclusions.json', [])
            with self.assertRaises(AssertionError):
                feature_source(data, generated)


if __name__ == '__main__':
    unittest.main()
