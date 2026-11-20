"""Reproduce the fixed cache-based experiment without changing any input.

No audio/weight download, GPU inference, or parameter retuning is performed.
The historical protocol's relative split path is explicitly remapped to the
published input root; the protocol JSON and frozen scientific values are kept.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import platform
import time

import numpy as np

import evaluate
import pipeline
import verify_development
import verify_evaluation
from feature_source import feature_source


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def separate_directories(data_dir, output_dir):
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    if not data.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {data}')
    if data == output or data in output.parents or output in data.parents:
        raise ValueError('Input and output directories must be separate, non-nested trees')
    return data, output


def compare_values(actual, expected, path='value', tolerance=2e-14):
    """Compare all nested fields; floats use the original audit tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            compare_values(actual[key], expected[key], f'{path}.{key}', tolerance)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare_values(left, right, f'{path}[{i}]', tolerance)
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def verify_input_accounting(data):
    conf = read(data/'minimum_experiment.json')
    frozen = read(data/'frozen_development_settings.json')
    original = read(data/'development_calibration.json')
    planned = read(data/'data/experiment_manifest.json')
    active = read(data/'cache/active_manifest.json')
    excluded = read(data/'cache/exclusions.json')
    assert pipeline.sha(data/'minimum_experiment.json') == frozen['protocol_sha256']
    # Retain the historical serialized path without recreating its directory tree.
    assert pipeline.sha(data/'planned_split.json') == conf['split_sha256'] == frozen['split_sha256']
    assert pipeline.sha(data/'cache/active_manifest.json') == original['active_manifest_sha256']
    assert pipeline.sha(data/'cache/exclusions.json') == original['exclusions_sha256']
    assert pipeline.sha(data/'cache/model_spec.json') == frozen['model_spec_sha256']
    assert all(original[key] == value for key, value in frozen.items())
    groups = [{str(r['track_id']) for r in rows} for rows in [planned, active, excluded]]
    assert all(len(group) == len(rows) for group, rows in zip(groups, [planned, active, excluded]))
    assert groups[1].isdisjoint(groups[2]) and groups[1] | groups[2] == groups[0]
    original_by_id = {str(row['track_id']): row for row in planned}
    for row in active:
        tid = str(row['track_id'])
        assert tid.isdecimal(), 'Track ID must be a safe numeric filename'
        assert all(row[key] == original_by_id[tid][key] for key in ['split', 'artist_id', 'captions'])
        assert row['captions'] == sorted(row['captions'], key=lambda x: int(x['caption_id']))
        assert row['split'] in ['development', 'evaluation']
    assert len(active) == 408 and len(excluded) == 31
    dev_artists = {r['artist_id'] for r in active if r['split'] == 'development'}
    eval_artists = {r['artist_id'] for r in active if r['split'] == 'evaluation'}
    assert dev_artists.isdisjoint(eval_artists)
    return conf, frozen, original, active


def reproduce_development(data, output, conf, frozen, original, active, feature):
    # This loader structurally filters development before opening any NPZ.
    arrays = pipeline.load_development(feature['rows'], feature['cache_dir'])
    feature_by_id = {str(row['track_id']): row for row in feature['rows']}
    score_tolerance = feature['details']['score_absolute_tolerance_to_original']
    assert arrays['catalog_ids'] == frozen['catalog_ids']
    assert arrays['source_ids'] == frozen['query_source_ids']
    for tid in arrays['catalog_ids']:
        digest = pipeline.sha(feature['cache_dir']/f'{tid}.npz')
        assert digest == feature_by_id[tid]['npz_sha256']
        if feature['details']['mode'] == 'original_cache':
            assert digest == frozen['cache_checksums'][tid]
    actual = pipeline.calibrate(**arrays, lambda_grid=conf['lambda_grid'])
    for key in actual:
        tolerance = score_tolerance if key in ['thresholds', 'development_scores'] else 2e-14
        compare_values(actual[key], original[key], f'development.{key}', tolerance)
    assert actual['selected_lambda'] == frozen['selected_lambda']
    pipeline.atomic_json(output/'development_recalculation.json', actual)
    pipeline.atomic_json(output/'development_match.json', {
        'status': 'passed', 'development_catalog_count': len(arrays['catalog_ids']),
        'development_query_count': len(arrays['source_ids']),
        'evaluation_features_opened': 0, 'frozen_settings_updated': False,
        'selected_lambda': actual['selected_lambda'],
        'recomputed_thresholds': actual['thresholds'],
        'evaluation_uses_original_frozen_thresholds': frozen['thresholds'],
        'threshold_absolute_tolerance': score_tolerance, 'metric_absolute_tolerance': 2e-14,
        'feature_source': feature['details'],
    })


def compare_reference(data, output, score_tolerance=1e-7):
    expected_dir = data/'expected/evaluation'
    produced_dir = output/'evaluation'
    original = read(expected_dir/'results.json')
    result = read(produced_dir/'results.json')
    # Original code/environment hashes remain historical evidence; the portable
    # implementation receives its own new run_spec and artifact checksums.
    for filename, checksum in original['artifact_sha256'].items():
        assert pipeline.sha(expected_dir/filename) == checksum, ('reference integrity', filename)
    actual_fields = {k: v for k, v in result.items() if k != 'artifact_sha256'}
    original_fields = {k: v for k, v in original.items() if k != 'artifact_sha256'}
    compare_values(actual_fields, original_fields, 'evaluation')
    with np.load(expected_dir/'raw_arrays.npz', allow_pickle=False) as expected:
        with np.load(produced_dir/'raw_arrays.npz', allow_pickle=False) as actual:
            assert set(actual.files) == set(expected.files)
            for key in expected.files:
                if key == 'verification_scores':
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=score_tolerance)
                elif key == 'aurocs':
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=2e-15, equal_nan=True)
                else:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with np.load(expected_dir/'bootstrap_weights.npz', allow_pickle=False) as expected:
        with np.load(produced_dir/'bootstrap_weights.npz', allow_pickle=False) as actual:
            assert set(actual.files) == set(expected.files)
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    compare_values(read(produced_dir/'assignments.json'), read(expected_dir/'assignments.json'), 'assignments', 0)
    compare_values(read(produced_dir/'cases.json'), read(expected_dir/'cases.json'), 'cases', score_tolerance)
    return {
        'status': 'passed', 'all_result_fields_except_artifact_hashes_match': True,
        'all_22080_ranks_match_exactly': True, 'gates_and_assignments_match_exactly': True,
        'bootstrap_weights_match_exactly': True,
        'metric_absolute_tolerance': 2e-14, 'score_absolute_tolerance': score_tolerance,
        'auroc_absolute_tolerance': 2e-15,
        'original_results_sha256': pipeline.sha(expected_dir/'results.json'),
        'reproduced_results_sha256': pipeline.sha(produced_dir/'results.json'),
        'different_code_and_artifact_provenance_is_expected': True,
    }


def run(data_dir, output_dir, regenerated_features=None):
    started = time.perf_counter()
    data, output = separate_directories(data_dir, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_before = pipeline.sha(data/'frozen_development_settings.json')
    conf, frozen, original, active = verify_input_accounting(data)
    if regenerated_features is not None:
        separate_directories(regenerated_features, output)
    feature = feature_source(data, regenerated_features)
    # Freeze provenance and tolerances before any development/evaluation score.
    evaluate.locked_json(output/'reproduction_spec.json', {
        'feature_source': feature['details'],
        'protocol_sha256': pipeline.sha(data/'minimum_experiment.json'),
        'frozen_settings_sha256': frozen_before,
        'reproduction_script_sha256': pipeline.sha(Path(__file__)),
        'feature_source_validator_sha256': pipeline.sha(Path(__file__).parent/'feature_source.py'),
    })
    reproduce_development(data, output, conf, frozen, original, active, feature)
    verify_development.main(data, output, regenerated_features)
    evaluate.run(data, output/'evaluation', regenerated_features)
    verify_evaluation.main(data, output/'evaluation', regenerated_features)
    reference = compare_reference(data, output, feature['details']['score_absolute_tolerance_to_original'])
    assert pipeline.sha(data/'frozen_development_settings.json') == frozen_before
    pipeline.atomic_json(output/'reference_comparison.json', reference)
    report = {
        'status': 'passed', 'scope': 'cache-based development and evaluation reproduction',
        'feature_source': feature['details'],
        'new_audio_embedding_inference': False, 'gpu_used': False,
        'frozen_settings_unchanged': True, 'selected_lambda': frozen['selected_lambda'],
        'thresholds': frozen['thresholds'], 'development_tracks': 99, 'evaluation_tracks': 309,
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn']},
        'protocol_sha256': pipeline.sha(data/'minimum_experiment.json'),
        'frozen_settings_sha256': frozen_before,
        'portable_scripts_sha256': {p.name: pipeline.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'historical_pipeline_sha256': original['pipeline_sha256'],
        'elapsed_seconds': time.perf_counter()-started,
        'independent_development_verification': 'passed',
        'independent_evaluation_verification': 'passed',
        'comparison_to_original_numeric_results': 'passed',
    }
    pipeline.atomic_json(output/'reproduction_summary.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regenerated-features', type=Path,
                        help='Separate locally generated feature directory; never updates original frozen settings')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.regenerated_features)
