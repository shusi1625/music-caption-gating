"""Independent CPU verification of persisted development calibration.

Uses direct comparison counts for ranks, explicit sorted-score interpolation for
quantiles, and never opens an evaluation feature file. Does not import pipeline.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from feature_source import feature_source

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent / "artifacts/input"
OUT = SCRIPT_DIR.parent / "outputs/reproduction"


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unit(x):
    lengths = np.sqrt(np.sum(x*x, axis=-1, keepdims=True))
    if not np.isfinite(x).all() or np.any(lengths <= 1e-12):
        raise AssertionError('Invalid vector')
    return x / lengths


def main(data_dir=None, output_dir=None, regenerated_features=None):
    global HERE, OUT
    if data_dir is not None:
        HERE = Path(data_dir).resolve()
    if output_dir is not None:
        OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    result = read(HERE/'development_calibration.json')
    feature = feature_source(HERE, regenerated_features)
    feature_by_id = {str(row['track_id']): row for row in feature['rows']}
    score_tolerance = feature['details']['score_absolute_tolerance_to_original']
    conf = read(HERE/'minimum_experiment.json')
    planned = read(HERE/'data/experiment_manifest.json')
    active = read(HERE/'cache/active_manifest.json')
    excluded = read(HERE/'cache/exclusions.json')
    run = read(HERE/'cache/run_report.json')
    plan_ids = {r['track_id'] for r in planned}
    active_ids = {r['track_id'] for r in active}
    excluded_ids = {r['track_id'] for r in excluded}
    assert len(plan_ids) == len(planned)
    assert len(active_ids) == len(active)
    assert len(excluded_ids) == len(excluded)
    assert active_ids.isdisjoint(excluded_ids)
    assert active_ids | excluded_ids == plan_ids
    assert not run['errors']
    assert run['processed'] + run['reused'] == len(active)
    assert run['excluded_short_audio'] == len(excluded)
    assert run['total_requested'] == len(planned)
    assert run['manifest_sha256'] == canonical_digest(planned)
    for item in excluded:
        assert item['reason'] == 'fewer_than_twelve_complete_windows'
        assert 0 <= item['padding_samples'] <= 2400
        assert item['original_seconds'] + item['padding_samples']/48000 < 120

    for field, path in [
        ('protocol_sha256', HERE/'minimum_experiment.json'),
        ('split_sha256', HERE/'planned_split.json'),
        ('model_spec_sha256', HERE/'cache/model_spec.json'),
        ('active_manifest_sha256', HERE/'cache/active_manifest.json'),
        ('exclusions_sha256', HERE/'cache/exclusions.json'),
    ]:
        assert result[field] == digest(path), field

    by_id = {r['track_id']: r for r in planned}
    expected = sorted([r['track_id'] for r in planned
                       if r['split'] == 'development' and r['track_id'] not in excluded_ids], key=int)
    catalog = result['catalog_ids']
    assert catalog == expected
    sources = result['query_source_ids']
    assert sources == [tid for tid in expected if len(by_id[tid]['captions']) >= 2]
    assert result['threshold_input_count'] == len(expected)
    assert result['query_count'] == len(sources)
    assert result['evaluation_scores_computed'] is False
    assert set(result['length_excluded_development_ids']) == {
        r['track_id'] for r in excluded if r['split'] == 'development'}

    audio, indexed, queried, opened = [], [], [], []
    for tid in catalog:
        assert by_id[tid]['split'] == 'development'
        path = feature['cache_dir']/f'{tid}.npz'
        assert digest(path) == feature_by_id[tid]['npz_sha256']
        if regenerated_features is None:
            assert digest(path) == result['cache_checksums'][tid]
        with np.load(path, allow_pickle=False) as values:
            opened.append(tid)
            assert str(values['track_id']) == tid
            assert list(values['caption_ids']) == [r['caption_id'] for r in by_id[tid]['captions']]
            assert values['audio'].shape == (12, 512)
            assert values['text'].shape == (len(by_id[tid]['captions']), 512)
            audio.append(values['audio'])
            indexed.append(values['text'][0])
            if tid in sources:
                queried.append(values['text'][1])
    audio = unit(np.stack(audio))
    indexed = unit(np.stack(indexed))
    queried = unit(np.stack(queried))
    pooled = unit(audio.mean(axis=1))
    views = {'first_window': audio[:, 0], 'normalized_12_window_mean': pooled}
    thresholds = {}
    for key, value in views.items():
        scores = np.sum(value*indexed, axis=1)
        ordered = sorted(float(x) for x in scores)
        h = (len(ordered)-1)*.1
        lower, upper = math.floor(h), math.ceil(h)
        quantile = ordered[lower] + (ordered[upper]-ordered[lower])*(h-lower)
        thresholds[key] = quantile
        assert abs(result['thresholds'][key]-quantile) < score_tolerance
        # Small interpolation round-off must not change the retained set here.
        retained = float(np.mean(scores >= result['thresholds'][key]))
        assert retained == result['retained_fraction'][key]
        np.testing.assert_allclose(scores, result['development_scores'][key],
                                   atol=score_tolerance, rtol=0 if regenerated_features is not None else 1e-7)
        np.testing.assert_array_equal(scores >= result['thresholds'][key],
                                      np.asarray(result['development_scores'][key]) >= result['thresholds'][key])

    A = queried@pooled.T
    T = queried@indexed.T
    columns = {tid: i for i, tid in enumerate(catalog)}
    numbers = np.array([int(tid) for tid in catalog])
    exact_by_weight = {}
    reported_grid = {r['lambda']: r for r in result['lambda_grid_results']}
    assert sorted(reported_grid) == sorted(conf['lambda_grid'])
    for weight in sorted(conf['lambda_grid']):
        scores = (1-weight)*A + weight*T
        ranks = []
        for i, source in enumerate(sources):
            target = scores[i, columns[source]]
            ahead = np.count_nonzero(scores[i] > target)
            tied_before = np.count_nonzero((scores[i] == target) & (numbers < int(source)))
            ranks.append(1 + int(ahead) + int(tied_before))
        exact = sum((Fraction(1, r) for r in ranks), Fraction(0))/len(ranks)
        reported = reported_grid[weight]
        assert ranks == reported['ranks'], weight
        assert abs(float(exact)-reported['MRR']) < 1e-14
        assert str(exact) == reported['exact_mrr']
        for key, k in [('R1', 1), ('R5', 5), ('R10', 10)]:
            assert reported[key] == sum(r <= k for r in ranks)/len(ranks)
        exact_by_weight[weight] = exact
    best = max(exact_by_weight.values())
    selected = min(w for w, value in exact_by_weight.items() if value == best)
    assert selected == result['selected_lambda']

    # Evaluation metadata is inspected for accounting and duplicate identity;
    # no evaluation embeddings, score matrices, or ranks are opened/computed.
    pcm = defaultdict(list)
    for item in active:
        pcm[item['decoded_pcm_sha256']].append(item['track_id'])
    canonical_groups = lambda groups: sorted([sorted(g, key=int) for g in groups])
    groups = [ids for ids in pcm.values() if len(ids)>1]
    assert canonical_groups(groups) == canonical_groups(
        read(HERE/'cache/duplicates.json')['exact_decoded_duplicate_groups'])
    by_split = Counter(r['split'] for r in active)
    report = {
        'status': 'passed',
        'scope': 'independent development-only arithmetic and metadata accounting',
        'development_features_opened': len(opened),
        'evaluation_features_opened': 0,
        'active_tracks_by_split': dict(by_split),
        'excluded_tracks_by_split': dict(Counter(r['split'] for r in excluded)),
        'exact_decoded_duplicate_groups': groups,
        'thresholds_by_explicit_interpolation': thresholds,
        'selected_lambda': selected,
        'checks': ['active/excluded accounting', 'provenance SHA256', 'caption roles',
                   'explicit linear quantile', 'direct greater-than and tie counts',
                   'exact Fraction MRR at all lambda values', 'metadata PCM duplicates'],
        'calibration_sha256': digest(HERE/'development_calibration.json'),
        'historical_source_pipeline_sha256': result['pipeline_sha256'],
        'portable_verifier_sha256': digest(Path(__file__)),
        'feature_source': feature['details'],
        'shared_metadata_validation_only': True,
    }
    target = OUT/'development_verification.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regenerated-features', type=Path)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.regenerated_features)
