"""Validate feature provenance without reading embeddings or scoring queries.

The shared helper validates paths and metadata only. Independent arithmetic
verifiers still implement their own normalization, ranks, TF-IDF and intervals.
"""
from pathlib import Path
import hashlib
import json
import re


MODEL_WEIGHT_SHA256 = '1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(part)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def feature_source(data_dir, regenerated_dir=None):
    data = Path(data_dir).resolve()
    original = read(data/'cache/active_manifest.json')
    frozen = read(data/'frozen_development_settings.json')
    calibration = read(data/'development_calibration.json')
    config = read(data/'minimum_experiment.json')
    assert digest(data/'cache/active_manifest.json') == calibration['active_manifest_sha256']
    assert digest(data/'minimum_experiment.json') == frozen['protocol_sha256']
    assert digest(data/'cache/exclusions.json') == calibration['exclusions_sha256']
    details = {
        'mode': 'original_cache' if regenerated_dir is None else 'regenerated_features',
        'score_absolute_tolerance_to_original': 1e-7 if regenerated_dir is None else 1e-5,
        'development_threshold_absolute_tolerance': 1e-7 if regenerated_dir is None else 1e-5,
        'metric_absolute_tolerance_to_original': 2e-14,
        'ranks_gates_assignments_must_match_original_exactly': True,
        'original_frozen_thresholds_and_lambda_used': True,
        'original_active_manifest_sha256': digest(data/'cache/active_manifest.json'),
    }
    if regenerated_dir is None:
        return {'cache_dir': data/'cache/clap', 'rows': original,
                'model_spec': read(data/'cache/model_spec.json'), 'details': details}

    root = Path(regenerated_dir).resolve()
    provenance = read(root/'provenance.json')
    expected = {
        'original_protocol_sha256': digest(data/'minimum_experiment.json'),
        'original_active_manifest_sha256': digest(data/'cache/active_manifest.json'),
        'original_frozen_settings_sha256': digest(data/'frozen_development_settings.json'),
        'model_weight_sha256': MODEL_WEIGHT_SHA256,
        'model': config['model'], 'revision': config['revision'],
    }
    assert all(provenance.get(key) == value for key, value in expected.items()), 'Regenerated provenance changed'
    spec = read(root/'model_spec.json')
    assert spec['model'] == config['model'] and spec['revision'] == config['revision']
    assert spec['sample_rate'] == 48000 and spec['windows'] == 12 and spec['window_samples'] == 480000
    if 'weight_sha256' in spec:
        assert spec['weight_sha256'] == MODEL_WEIGHT_SHA256
    for field, expected_hash in [
        ('model_spec_sha256', canonical(spec)),
        ('original_exclusions_sha256', digest(data/'cache/exclusions.json')),
    ]:
        if field in provenance:
            assert provenance[field] == expected_hash, field
    if 'download_manifest_sha256' in provenance:
        assert provenance['download_manifest_sha256'] == digest(data/'download_manifest.json')
    rows = read(root/'manifest.json')
    by_id = {str(row['track_id']): row for row in rows}
    original_by_id = {str(row['track_id']): row for row in original}
    assert len(rows) == len(by_id) == len(original) and set(by_id) == set(original_by_id)
    for tid, row in by_id.items():
        assert tid.isdecimal(), 'Unsafe track ID'
        prior = original_by_id[tid]
        assert all(row[key] == prior[key] for key in ['track_id', 'split', 'artist_id', 'captions']), tid
        assert row['lineage']['audio_sha256'] == prior['lineage']['audio_sha256'], tid
        assert row['lineage']['captions'] == prior['captions'], tid
        assert row['lineage']['spec'] == canonical(spec), tid
        assert row['cache_key'] == canonical(row['lineage']), tid
        assert re.fullmatch('[0-9a-f]{64}', row['npz_sha256']), tid
        assert read(root/'clap'/f'{tid}.json') == row, tid
    exclusions = read(root/'exclusions.json')
    original_exclusions = read(data/'cache/exclusions.json')
    excluded_ids = {str(row['track_id']) for row in exclusions}
    assert len(exclusions) == len(excluded_ids) == len(original_exclusions)
    assert excluded_ids == {str(row['track_id']) for row in original_exclusions}
    planned = {str(row['track_id']): row for row in read(data/'data/experiment_manifest.json')}
    for row in exclusions:
        prior = planned[str(row['track_id'])]
        assert all(row[key] == prior[key] for key in ['track_id', 'split', 'artist_id', 'captions'])
        assert row['reason'] == 'fewer_than_twelve_complete_windows'
        assert 0 <= row['padding_samples'] <= 2400
        assert row['original_seconds'] + row['padding_samples']/48000 < 120
    details.update({
        'regenerated_manifest_sha256': digest(root/'manifest.json'),
        'regenerated_exclusions_sha256': digest(root/'exclusions.json'),
        'regenerated_model_spec_sha256': digest(root/'model_spec.json'),
        'regenerated_provenance_sha256': digest(root/'provenance.json'),
        'regeneration_provenance': provenance,
    })
    return {'cache_dir': root/'clap', 'rows': rows, 'model_spec': spec, 'details': details}
