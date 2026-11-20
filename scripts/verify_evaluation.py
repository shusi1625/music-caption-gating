"""Independent CPU audit of the frozen evaluation and artist bootstrap.

No imports from evaluate/pipeline, no sklearn, no settings selection. Reads the
saved feature files directly. Ranks use strict greater-than/tied-ID counts;
TF-IDF uses explicit token counts; confidence limits use sorted interpolation.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import time
import csv

import numpy as np

from feature_source import feature_source

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent / "artifacts/input"
OUT = SCRIPT_DIR.parent / "outputs/reproduction/evaluation"
FEATURES = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unit(values):
    lengths = np.sqrt(np.sum(values * values, axis=-1, keepdims=True))
    assert np.isfinite(values).all() and np.all(lengths > 1e-12)
    return values / lengths


def load_features(rows):
    audio, text, queries, checked = [], [], [], {}
    for row in rows:
        tid = str(row['track_id'])
        path = FEATURES['cache_dir'] / (tid + '.npz')
        checksum = digest(path)
        assert checksum == row['npz_sha256'], ('cache hash', tid)
        with np.load(path, allow_pickle=False) as feature:
            assert str(feature['track_id']) == tid
            assert list(feature['caption_ids']) == [c['caption_id'] for c in row['captions']]
            assert feature['audio'].shape == (12, 512)
            assert feature['text'].shape == (len(row['captions']), 512)
            assert np.isfinite(feature['audio']).all() and np.isfinite(feature['text']).all()
            assert np.allclose(np.sum(feature['audio']**2, axis=-1), 1, atol=2e-5)
            assert np.allclose(np.sum(feature['text']**2, axis=-1), 1, atol=2e-5)
            audio.append(feature['audio'])
            text.append(feature['text'][0])
            if len(row['captions']) >= 2:
                queries.append(feature['text'][1])
        checked[tid] = checksum
    audio = unit(np.stack(audio))
    text = unit(np.stack(text))
    queries = unit(np.stack(queries))
    mean = unit(audio.mean(axis=1))
    return audio, text, queries, mean, checked


def explicit_tfidf(indexed, queried):
    """Default sklearn word tokenization/count/IDF/L2 formula, independent code."""
    tokenize = re.compile(r'(?u)\b\w\w+\b').findall
    clean_counts = [Counter(tokenize(text.lower())) for text in indexed]
    query_counts = [Counter(tokenize(text.lower())) for text in queried]
    vocabulary = sorted(set().union(*(set(c) for c in clean_counts)))
    columns = {token: i for i, token in enumerate(vocabulary)}
    n = len(indexed)
    df = Counter(token for counts in clean_counts for token in counts)
    idf = np.array([math.log((1+n)/(1+df[token])) + 1 for token in vocabulary], dtype=np.float64)

    def matrix(items):
        values = np.zeros((len(items), len(vocabulary)), dtype=np.float64)
        for i, counts in enumerate(items):
            for token, count in counts.items():
                if token in columns:
                    values[i, columns[token]] = count * idf[columns[token]]
        lengths = np.sqrt(np.sum(values*values, axis=1, keepdims=True))
        return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 0)

    return matrix(query_counts) @ matrix(clean_counts).T, len(vocabulary)


def direct_ranks(scores, catalog_ids, query_ids):
    ids = np.array([int(tid) for tid in catalog_ids])
    position = {tid: i for i, tid in enumerate(catalog_ids)}
    values = []
    for i, tid in enumerate(query_ids):
        target = scores[i, position[tid]]
        better = int(np.count_nonzero(scores[i] > target))
        tied_before = int(np.count_nonzero((scores[i] == target) & (ids < int(tid))))
        values.append(1 + better + tied_before)
    return np.array(values, dtype=np.int64)


def independent_assignment(artist_ids, fraction, seed):
    n = len(artist_ids)
    count = math.floor(fraction*n + .5)
    mapping = np.arange(n)
    selected = np.zeros(n, dtype=bool)
    if count == 0:
        return mapping, selected, 0
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(n, size=count, replace=False))
    selected[positions] = True
    for attempts in range(1, 10001):
        donors = rng.permutation(positions)
        if all(artist_ids[int(a)] != artist_ids[int(b)] for a, b in zip(positions, donors)):
            mapping[positions] = donors
            return mapping, selected, attempts
    raise AssertionError('Protocol assignment exhausted 10000 attempts')


def independent_weights(artists, repeats, seed):
    """Draw artist indices and expand each complete selected cluster explicitly."""
    labels = sorted(set(map(str, artists)))
    positions = {label: [i for i, a in enumerate(artists) if str(a) == label] for label in labels}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(repeats, len(labels)))
    weights = np.zeros((repeats, len(artists)), dtype=np.float64)
    for b, draw in enumerate(draws):
        for label_index, multiplicity in Counter(map(int, draw)).items():
            weights[b, positions[labels[label_index]]] = multiplicity
        weights[b] /= float(weights[b].sum())
    return weights


def explicit_quantile(values, p):
    ordered = sorted(float(v) for v in values)
    at = p*(len(ordered)-1)
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)


def interval(values, weights, level):
    values = np.asarray(values, dtype=np.float64)
    replicates = weights @ values
    tail = (1-level)/2
    return {
        'estimate': float(sum(values)/len(values)),
        'lower': explicit_quantile(replicates, tail),
        'upper': explicit_quantile(replicates, 1-tail),
        'level': level,
    }


def assert_interval(actual, expected, label):
    for key, value in expected.items():
        assert math.isclose(float(actual[key]), value, rel_tol=0, abs_tol=2e-14), (label, key, actual[key], value)


def audit_arrays(raw, config, frozen, rows, catalog_ids, query_ids):
    audio, text, queries, pooled, checked = load_features(rows)
    artists = [str(row['artist_id']) for row in rows]
    n = len(catalog_ids)
    rates, seeds = config['corruption_rates'], config['corruption_seeds']
    conditions = config['conditions']
    assert raw['ranks'].shape == (len(rates), len(seeds), len(conditions), len(query_ids))
    assert raw['donor_indices'].shape == (len(rates), len(seeds), n)
    assert raw['selected_mask'].shape == (len(rates), len(seeds), n)
    assert raw['verification_scores'].shape == (len(rates), len(seeds), 2, n)
    assert raw['gates'].shape == (len(rates), len(seeds), 4, n)
    assert np.issubdtype(raw['ranks'].dtype, np.integer)
    assert np.issubdtype(raw['donor_indices'].dtype, np.integer)
    assert raw['selected_mask'].dtype == np.dtype(bool)
    assert raw['gates'].dtype == np.dtype(bool)
    assert np.all((raw['ranks'] >= 1) & (raw['ranks'] <= n))
    A = queries @ pooled.T
    clean_T = queries @ text.T
    lexical, vocabulary_size = explicit_tfidf(
        [row['captions'][0]['caption'] for row in rows],
        [row['captions'][1]['caption'] for row in rows if len(row['captions']) >= 2],
    )
    weight = frozen['selected_lambda']
    threshold_first = frozen['thresholds']['first_window']
    threshold_mean = frozen['thresholds']['normalized_12_window_mean']
    ranks = np.empty_like(raw['ranks'])
    all_scores = np.empty_like(raw['verification_scores'])
    all_gates = np.empty_like(raw['gates'])
    assignments = []
    for pi, rate in enumerate(rates):
        for si, seed in enumerate(seeds):
            donor, selected, attempts = independent_assignment(artists, rate, seed)
            np.testing.assert_array_equal(raw['donor_indices'][pi, si], donor)
            np.testing.assert_array_equal(raw['selected_mask'][pi, si], selected)
            # Structural checks supplement RNG replay and are independent of it.
            assert selected.sum() == math.floor(rate*n + .5)
            assert np.array_equal(np.sort(donor), np.arange(n))
            assert np.array_equal(donor[~selected], np.arange(n)[~selected])
            assert np.array_equal(np.sort(donor[selected]), np.flatnonzero(selected))
            assert all(artists[i] != artists[int(donor[i])] for i in np.flatnonzero(selected))
            contaminated_text = text[donor]
            first_score = np.sum(audio[:, 0] * contaminated_text, axis=1)
            mean_score = np.sum(pooled * contaminated_text, axis=1)
            all_scores[pi, si] = np.stack([first_score, mean_score])
            np.testing.assert_allclose(raw['verification_scores'][pi, si],
                                       all_scores[pi, si], rtol=0, atol=1e-7)
            first = first_score.astype(np.float64) >= threshold_first
            mean = mean_score.astype(np.float64) >= threshold_mean
            order = np.random.default_rng(config['random_gate_seed_base'] + seed).permutation(n)
            random_first = np.ones(n, dtype=bool)
            random_mean = np.ones(n, dtype=bool)
            random_first[order[:np.count_nonzero(~first)]] = False
            random_mean[order[:np.count_nonzero(~mean)]] = False
            all_gates[pi, si] = np.stack([first, mean, random_first, random_mean])
            np.testing.assert_array_equal(raw['gates'][pi, si], all_gates[pi, si])
            assert np.count_nonzero(~random_first) == np.count_nonzero(~first)
            assert np.count_nonzero(~random_mean) == np.count_nonzero(~mean)
            T = clean_T[:, donor]
            gain = np.float32(weight)
            difference = T-A
            scores = {
                'audio_only': A,
                'caption_only': T,
                'ungated_fusion': A + gain*difference,
                'gate_first': A + gain*first.astype(np.float32)*difference,
                'gate_mean12': A + gain*mean.astype(np.float32)*difference,
                'random_gate_matched_first': A + gain*random_first.astype(np.float32)*difference,
                'random_gate_matched_mean12': A + gain*random_mean.astype(np.float32)*difference,
                'tfidf': lexical[:, donor],
            }
            for ci, condition in enumerate(conditions):
                ranks[pi, si, ci] = direct_ranks(scores[condition], catalog_ids, query_ids)
                np.testing.assert_array_equal(raw['ranks'][pi, si, ci], ranks[pi, si, ci],
                                              err_msg=f'p={rate} seed={seed} condition={condition}')
            assignments.append({'p': rate, 'seed': seed, 'attempts': attempts,
                                'selected_indices': np.flatnonzero(selected).tolist(),
                                'donor_indices': donor.tolist()})
    return {'ranks': ranks, 'gates': all_gates, 'verification_scores': all_scores,
            'cache_checksums': checked, 'assignments': assignments,
            'vocabulary_size': vocabulary_size}


def main(data_dir=None, output_dir=None, regenerated_features=None):
    global HERE, OUT, FEATURES
    if data_dir is not None:
        HERE = Path(data_dir).resolve()
    if output_dir is not None:
        OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = read(HERE/'minimum_experiment.json')
    frozen = read(HERE/'frozen_development_settings.json')
    calibration = read(HERE/'development_calibration.json')
    FEATURES = feature_source(HERE, regenerated_features)
    active = FEATURES['rows']
    planned = read(HERE/'data/experiment_manifest.json')
    planned_by_id = {str(row['track_id']): row for row in planned}
    result = read(OUT/'results.json')
    spec = read(OUT/'run_spec.json')
    assert spec['feature_source'] == FEATURES['details']
    assert spec['feature_source_validator_sha256'] == digest(SCRIPT_DIR/'feature_source.py')
    assignments = read(OUT/'assignments.json')
    cases = read(OUT/'cases.json')
    assert all(frozen[key] == calibration[key] for key in frozen)
    assert digest(HERE/'planned_split.json') == config['split_sha256'] == frozen['split_sha256']
    assert digest(HERE/'cache/model_spec.json') == frozen['model_spec_sha256']
    assert digest(HERE/'minimum_experiment.json') == frozen['protocol_sha256']
    assert digest(HERE/'cache/active_manifest.json') == calibration['active_manifest_sha256']
    assert digest(HERE/'cache/exclusions.json') == calibration['exclusions_sha256']
    assert not result['evaluation_settings_retuned'] and not spec['evaluation_retuning']
    assert result['status'] == 'completed'
    for filename, checksum in result['artifact_sha256'].items():
        assert digest(OUT/filename) == checksum, filename
    for field, path in {
        'protocol_sha256': HERE/'minimum_experiment.json',
        'frozen_settings_sha256': HERE/'frozen_development_settings.json',
        'active_manifest_sha256': HERE/'cache/active_manifest.json',
        'development_calibration_sha256': HERE/'development_calibration.json',
        'evaluator_sha256': SCRIPT_DIR/'evaluate.py',
        'pipeline_sha256': SCRIPT_DIR/'pipeline.py',
    }.items():
        assert spec[field] == digest(path), field
    rows = sorted((row for row in active if row['split'] == 'evaluation'), key=lambda r: int(r['track_id']))
    for row in rows:
        original = planned_by_id[row['track_id']]
        assert all(row[key] == original[key] for key in ['split', 'artist_id', 'captions'])
        assert row['captions'] == sorted(row['captions'], key=lambda r: int(r['caption_id']))
    catalog_ids = [str(row['track_id']) for row in rows]
    query_rows = [row for row in rows if len(row['captions']) >= 2]
    query_ids = [str(row['track_id']) for row in query_rows]
    artists = [str(row['artist_id']) for row in rows]
    query_artists = [str(row['artist_id']) for row in query_rows]
    assert len(catalog_ids) == len(set(catalog_ids)) == 309
    assert len(query_ids) == len(set(query_ids)) == 69
    assert len(set(artists)) == 108 and len(set(query_artists)) == 38
    assert not set(artists) & {str(row['artist_id']) for row in active if row['split'] == 'development'}
    assert not set(catalog_ids) & set(frozen['catalog_ids'])
    assert spec['catalog_ids'] == catalog_ids and spec['query_source_ids'] == query_ids
    for field, expected in [('catalog_tracks', len(catalog_ids)), ('query_tracks', len(query_ids)),
                            ('catalog_artists', len(set(artists))), ('query_artists', len(set(query_artists))),
                            ('selected_lambda', frozen['selected_lambda']), ('thresholds', frozen['thresholds']),
                            ('rates', config['corruption_rates']), ('seeds', config['corruption_seeds']),
                            ('conditions', config['conditions']), ('corruption_count_25pct', 77)]:
        assert result[field] == expected, field
    raw = dict(np.load(OUT/'raw_arrays.npz', allow_pickle=False))
    assert raw['catalog_ids'].tolist() == catalog_ids
    assert raw['query_source_ids'].tolist() == query_ids
    assert raw['catalog_artist_ids'].tolist() == artists
    assert raw['query_artist_ids'].tolist() == query_artists
    independent = audit_arrays(raw, config, frozen, rows, catalog_ids, query_ids)
    assert independent['cache_checksums'] == spec['cache_checksums']
    assert independent['vocabulary_size'] == result['tfidf_vocabulary_size']
    assert len(assignments) == len(independent['assignments'])
    for recorded, checked in zip(assignments, independent['assignments']):
        selected = checked['selected_indices']
        mapping = checked['donor_indices']
        assert recorded == {
            'rate': checked['p'], 'seed': checked['seed'], 'selected_count': len(selected),
            'permutation_attempts': checked['attempts'],
            'rejected_permutations': max(checked['attempts']-1, 0),
            'selected_track_ids': [catalog_ids[i] for i in selected],
            'target_to_donor': {catalog_ids[i]: catalog_ids[mapping[i]] for i in selected},
        }
    boot = config['bootstrap']
    for key, value in boot.items():
        assert result['bootstrap'][key] == spec['bootstrap'][key] == value
    weights = dict(np.load(OUT/'bootstrap_weights.npz', allow_pickle=False))
    q_weights = independent_weights(query_artists, boot['resamples'], boot['seed'])
    c_weights = independent_weights(artists, boot['resamples'], boot['seed'])
    np.testing.assert_array_equal(weights['query'], q_weights)
    np.testing.assert_array_equal(weights['catalog'], c_weights)
    for values in [q_weights, c_weights]:
        assert np.isfinite(values).all() and np.all(values >= 0)
        np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=1e-15)
    ranks, gates = independent['ranks'], independent['gates']
    rr = 1.0/ranks
    rates, seeds, conditions = config['corruption_rates'], config['corruption_seeds'], config['conditions']
    metrics = {'MRR': rr, 'R1': (ranks <= 1).astype(float),
               'R5': (ranks <= 5).astype(float), 'R10': (ranks <= 10).astype(float)}
    retrieval = {(row['rate'], row['condition']): row for row in result['retrieval']}
    assert len(retrieval) == len(result['retrieval']) == len(rates)*len(conditions)
    for pi, rate in enumerate(rates):
        for ci, condition in enumerate(conditions):
            for metric, values in metrics.items():
                expected = interval(values[pi, :, ci].mean(axis=0), q_weights, .95)
                assert_interval(retrieval[(rate, condition)]['metrics'][metric], expected,
                                (rate, condition, metric))
    first_ci, mean_ci = conditions.index('gate_first'), conditions.index('gate_mean12')
    primary = {
        'H1': interval((~gates[0, 0, 0]).astype(float) - (~gates[0, 0, 1]).astype(float),
                       c_weights, boot['primary_ci_level']),
        'H2': interval((rr[1, :, mean_ci] - rr[1, :, first_ci]).mean(axis=0),
                       q_weights, boot['primary_ci_level']),
    }
    for name, expected in primary.items():
        reported = result['primary_effects'][name]
        assert reported['definition'] == config['primary_effects'][name]
        assert_interval(reported, expected, name)
    for contrast in result['secondary_MRR_contrasts']:
        left, right = contrast['contrast'].split(' minus ')
        pi = rates.index(contrast['rate'])
        values = (rr[pi, :, conditions.index(left)] - rr[pi, :, conditions.index(right)]).mean(axis=0)
        assert_interval(contrast, interval(values, q_weights, .95), contrast['contrast'])
    # Pairwise concordance computes AUROC without sklearn/rank sorting.
    assert np.isnan(raw['aurocs'][0]).all()
    for vi, verifier in enumerate(result['verifiers']):
        rejected = (~gates[0, 0, vi]).astype(float)
        assert verifier['clean_reject_count'] == int(rejected.sum())
        assert_interval(verifier['clean_rejection_rate'], interval(rejected, c_weights, .95),
                        verifier['verifier'])
        accepted_swaps, aurocs = [], []
        for si, seed in enumerate(seeds):
            selected = raw['selected_mask'][1, si]
            clean = independent['verification_scores'][0, 0, vi, selected]
            swapped = independent['verification_scores'][1, si, vi, selected]
            concordance = (clean[:, None] > swapped[None, :]).astype(float)
            concordance += .5*(clean[:, None] == swapped[None, :])
            aurocs.append(float(concordance.sum()/concordance.size))
            accepted_swaps.append(float(gates[1, si, vi, selected].mean()))
        np.testing.assert_allclose(aurocs, raw['aurocs'][1, :, vi], rtol=0, atol=2e-15)
        np.testing.assert_allclose(aurocs, verifier['paired_track_source_auroc_by_seed'], rtol=0, atol=2e-15)
        np.testing.assert_array_equal(accepted_swaps, verifier['swapped_caption_acceptance_by_seed'])
        assert abs(np.mean(aurocs)-verifier['paired_track_source_auroc_mean']) < 2e-15
        assert abs(np.mean(accepted_swaps)-verifier['swapped_caption_acceptance_mean']) < 2e-15
    assert [case['track_id'] for case in cases] == config['case_track_ids']
    for case in cases:
        tid = case['track_id']
        qi, j = query_ids.index(tid), catalog_ids.index(tid)
        assert case['artist_id'] == artists[j]
        assert case['index_caption'] == rows[j]['captions'][0]
        assert case['query_caption'] == rows[j]['captions'][1]
        for vi, view in enumerate(result['view_order']):
            assert case['clean_source_scores'][view] == float(independent['verification_scores'][0, 0, vi, j])
            assert case['clean_accepted'][view] == bool(gates[0, 0, vi, j])
        assert case['swap_target_seeds'] == [seed for si, seed in enumerate(seeds) if raw['selected_mask'][1, si, j]]
        for pi, rate in enumerate(rates):
            for ci, condition in enumerate(conditions):
                assert case['rank_by_rate_condition_seed'][str(rate)][condition] == ranks[pi, :, ci, qi].tolist()
    with (OUT/'per_seed_retrieval.csv').open(encoding='utf-8-sig', newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rates)*len(seeds)*len(conditions)
    for row in csv_rows:
        pi, si, ci = rates.index(float(row['rate'])), seeds.index(int(row['seed'])), conditions.index(row['condition'])
        for metric, values in metrics.items():
            assert abs(float(row[metric])-values[pi, si, ci].mean()) < 2e-14
    n = len(catalog_ids)
    expected_random = {'MRR': math.fsum(1/i for i in range(1, n+1))/n,
                       'R1': 1/n, 'R5': 5/n, 'R10': 10/n}
    for key, value in expected_random.items():
        assert abs(result['random_ranking_expectation'][key]-value) < 1e-14
    report = {
        'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'independent frozen evaluation arithmetic, assignments, artist bootstrap and exported cases',
        'evaluation_features_opened': len(rows), 'development_features_opened': 0,
        'catalog_tracks': n, 'query_tracks': len(query_ids),
        'rank_values_compared': int(ranks.size), 'gate_values_compared': int(gates.size),
        'verification_scores_compared': int(independent['verification_scores'].size),
        'assignment_runs_compared': len(assignments),
        'bootstrap_weight_values_compared': int(q_weights.size+c_weights.size),
        'retrieval_intervals_compared': len(result['retrieval'])*len(metrics),
        'primary_effects': primary,
        'secondary_contrasts_compared': len(result['secondary_MRR_contrasts']),
        'auroc_values_compared': len(seeds)*2,
        'preselected_cases_compared': len(cases),
        'tfidf_vocabulary_size': independent['vocabulary_size'],
        'checks': ['frozen inputs and feature SHA256', 'split and caption roles',
                   'independent cross-artist permutation replay', 'random gate rejection counts and order',
                   'explicit token-count TF-IDF without sklearn', 'direct greater-than and ID-tie ranks',
                   'whole-artist resampling replay with multiplicities', 'paired CI by sorted linear interpolation',
                   'seed means before query-artist resampling', 'pairwise AUROC without sklearn',
                   'preselected cases and per-seed CSV'],
        'imports_evaluate_or_pipeline': False,
        'shared_metadata_validation_only': True,
        'feature_source': FEATURES['details'],
        'verifier_sha256': digest(Path(__file__)),
        'artifact_sha256': {filename: digest(OUT/filename) for filename in
                            ['results.json', 'run_spec.json', 'raw_arrays.npz', 'bootstrap_weights.npz',
                             'assignments.json', 'cases.json', 'per_seed_retrieval.csv']},
        'elapsed_seconds': time.perf_counter()-start,
    }
    destination = OUT/'independent_verification.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regenerated-features', type=Path)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.regenerated_features)
