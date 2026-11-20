"""Locked evaluation of caption assignment robustness; never retunes dev settings."""
from pathlib import Path
import csv
import importlib.metadata
import json
import math
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from pipeline import atomic_json, canonical_hash, normalize, rank_metrics, sha
from feature_source import feature_source

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent / "artifacts/input"
OUT = SCRIPT_DIR.parent / "outputs/reproduction/evaluation"
VIEW_NAMES = ['first_window', 'normalized_12_window_mean']
GATE_NAMES = ['first', 'mean12', 'random_first', 'random_mean12']
FEATURES = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def make_assignment(artist_ids, p, seed, max_attempts=10000):
    artists = np.asarray(artist_ids, dtype=str)
    n = len(artists)
    if not 0 <= p <= 1:
        raise ValueError('Corruption rate outside [0,1]')
    m = math.floor(p * n + .5)
    donor = np.arange(n)
    if m == 0:
        return donor, np.empty(0, dtype=int), 0
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(n, m, replace=False))
    _, counts = np.unique(artists[selected], return_counts=True)
    if 2 * counts.max() > m:
        raise ValueError('Cross-artist permutation impossible for fixed selected subset')
    for attempts in range(1, max_attempts + 1):
        proposed = rng.permutation(selected)
        if np.all(artists[proposed] != artists[selected]):
            donor[selected] = proposed
            assert np.array_equal(np.sort(donor), np.arange(n))
            return donor, selected, attempts
    raise RuntimeError('No valid permutation within fixed attempt limit; do not resample subset')


def random_gates(reject_counts, n, seed):
    ordering = np.random.default_rng(10000 + seed).permutation(n)
    gates = {}
    for name, count in reject_counts.items():
        if not 0 <= count <= n:
            raise ValueError('Invalid rejection count')
        g = np.ones(n, dtype=bool)
        g[ordering[:count]] = False
        gates[name] = g
    return gates


def cluster_weights(artist_ids, resamples, seed):
    labels, inverse = np.unique(np.asarray(artist_ids, dtype=str), return_inverse=True)
    k = len(labels)
    if not k or resamples <= 0:
        raise ValueError('Need observations and resamples')
    draws = np.random.default_rng(seed).integers(0, k, size=(resamples, k))
    multiplicities = np.stack([np.bincount(row, minlength=k) for row in draws])
    weights = multiplicities[:, inverse].astype(np.float64)
    return weights / weights.sum(axis=1, keepdims=True)


def interval(values, weights, level):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or weights.shape[1] != len(values):
        raise ValueError('Incompatible bootstrap values')
    if not np.isfinite(values).all() or not 0 < level < 1:
        raise ValueError('Invalid bootstrap inputs')
    samples = weights @ values
    alpha = (1 - level) / 2
    lower, upper = np.quantile(samples, [alpha, 1-alpha], method='linear')
    return {'estimate': float(values.mean()), 'lower': float(lower),
            'upper': float(upper), 'level': level}


def fusion_scores(audio_scores, text_scores, weight, gate=None):
    # Explicit cast avoids bool * Python float silently promoting gated scores.
    g = np.ones(audio_scores.shape[1], dtype=audio_scores.dtype) if gate is None else np.asarray(gate, dtype=audio_scores.dtype)
    return audio_scores + np.asarray(weight, dtype=audio_scores.dtype) * g[None, :] * (text_scores - audio_scores)


def locked_json(path, value):
    if path.exists():
        assert read(path) == value, f'Frozen evaluation input changed: {path.name}'
    else:
        atomic_json(path, value)


def save_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_evaluation():
    conf = read(HERE/'minimum_experiment.json')
    frozen = read(HERE/'frozen_development_settings.json')
    calibration = read(HERE/'development_calibration.json')
    assert sha(HERE/'minimum_experiment.json') == frozen['protocol_sha256']
    assert sha(HERE/'planned_split.json') == frozen['split_sha256']
    assert sha(HERE/'cache/model_spec.json') == frozen['model_spec_sha256']
    assert sha(HERE/'cache/active_manifest.json') == calibration['active_manifest_sha256']
    assert sha(HERE/'cache/exclusions.json') == calibration['exclusions_sha256']
    assert all(frozen[k] == calibration[k] for k in frozen)
    assert conf['corruption_rates'] == [0, .25]
    assert conf['corruption_seeds'] == list(range(20))
    assert conf['random_gate_seed_base'] == 10000
    source = FEATURES if FEATURES is not None else feature_source(HERE)
    active = source['rows']
    items = sorted([r for r in active if r['split'] == 'evaluation'], key=lambda r: int(r['track_id']))
    dev = [r for r in active if r['split'] == 'development']
    assert not {r['artist_id'] for r in items} & {r['artist_id'] for r in dev}
    assert {r['track_id'] for r in dev} == set(frozen['catalog_ids'])
    spec = source['model_spec']
    audio, texts, queries, sources, query_texts, checksums = [], [], [], [], [], {}
    for item in items:
        tid = item['track_id']
        path = source['cache_dir']/f'{tid}.npz'
        meta = read(path.with_suffix('.json'))
        assert meta == item
        assert meta['lineage']['spec'] == canonical_hash(spec)
        assert meta['cache_key'] == canonical_hash(meta['lineage'])
        digest = sha(path)
        assert digest == meta['npz_sha256']
        checksums[tid] = digest
        with np.load(path, allow_pickle=False) as f:
            assert str(f['track_id']) == tid
            assert list(f['caption_ids']) == [r['caption_id'] for r in item['captions']]
            assert f['audio'].shape == (12, 512)
            assert f['text'].shape == (len(item['captions']), 512)
            assert all(np.allclose(np.linalg.norm(f[k], axis=-1), 1, atol=1e-5) for k in ['audio', 'text'])
            audio.append(f['audio']); texts.append(f['text'][0])
            if len(item['captions']) >= 2:
                queries.append(f['text'][1]); sources.append(tid)
                query_texts.append(item['captions'][1]['caption'])
    assert len(items) == 309 and len(sources) == 69, 'Previously fixed valid split changed'
    return conf, frozen, items, np.stack(audio), np.stack(texts), np.stack(queries), sources, query_texts, checksums


def run(data_dir=None, output_dir=None, regenerated_features=None):
    global HERE, OUT, FEATURES
    if data_dir is not None:
        HERE = Path(data_dir).resolve()
    if output_dir is not None:
        OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    FEATURES = feature_source(HERE, regenerated_features)
    conf, frozen, items, audio, indexed, queried, sources, query_texts, hashes = load_evaluation()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [r['track_id'] for r in items]
    artists = [r['artist_id'] for r in items]
    query_columns = [ids.index(tid) for tid in sources]
    query_artists = [artists[j] for j in query_columns]
    rates, seeds, conditions = conf['corruption_rates'], conf['corruption_seeds'], conf['conditions']
    boot = conf['bootstrap']
    # Record implementation details before calculating any held-out score.
    run_spec = {
        'protocol_sha256': sha(HERE/'minimum_experiment.json'),
        'frozen_settings_sha256': sha(HERE/'frozen_development_settings.json'),
        'active_manifest_sha256': sha(HERE/'cache/active_manifest.json'),
        'development_calibration_sha256': sha(HERE/'development_calibration.json'),
        'cache_checksums': hashes, 'catalog_ids': ids, 'query_source_ids': sources,
        'feature_source': FEATURES['details'],
        'feature_source_validator_sha256': sha(SCRIPT_DIR/'feature_source.py'),
        'evaluator_sha256': sha(Path(__file__)), 'pipeline_sha256': sha(SCRIPT_DIR/'pipeline.py'),
        'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn']},
        'rng': 'numpy.default_rng PCG64',
        'selection': 'sorted choice without replacement; one subset, retry its permutation only',
        'max_permutation_attempts': 10000,
        'tfidf': {'fit': 'clean indexed captions only', 'analyzer': 'word', 'ngram_range': [1, 1],
                  'lowercase': True, 'norm': 'l2', 'smooth_idf': True, 'sublinear_tf': False,
                  'token_pattern': '(?u)\\b\\w\\w+\\b', 'stop_words': None},
        'bootstrap': {**boot, 'method': 'percentile linear quantile',
                      'artist_order': 'lexicographic str np.unique',
                      'draws': 'default_rng(seed).integers(0,K,size=(B,K)); reset same seed per population',
                      'estimand': 'track-weighted mean; fixed catalog, fixed 20 corruption seeds and dev settings'},
        'secondary_ci_level': .95,
        'auroc': 'ordinary binary AUROC, clean vs swapped scores on same selected tracks; descriptive, no CI',
        'dtype': 'float32 embedding scores matching development; float64 metrics and bootstrap',
        'evaluation_retuning': False,
    }
    locked_json(OUT/'run_spec.json', run_spec)
    audio = normalize(audio)
    pooled = normalize(audio.mean(axis=1))
    indexed, queried = normalize(indexed), normalize(queried)
    views = np.stack([audio[:, 0], pooled])
    A, clean_T = queried @ pooled.T, queried @ indexed.T
    lexical = TfidfVectorizer()
    lexical_index = lexical.fit_transform([r['captions'][0]['caption'] for r in items])
    lexical_query = lexical.transform(query_texts)
    lexical_clean = (lexical_query @ lexical_index.T).toarray()
    thresholds = np.array([frozen['thresholds'][k] for k in VIEW_NAMES])
    weight = frozen['selected_lambda']
    n, nq = len(items), len(sources)
    ranks = np.empty((len(rates), len(seeds), len(conditions), nq), dtype=np.int32)
    donor_indices = np.empty((len(rates), len(seeds), n), dtype=np.int32)
    selected_masks = np.zeros_like(donor_indices, dtype=bool)
    verification_scores = np.empty((len(rates), len(seeds), 2, n), dtype=np.float32)
    gates = np.empty((len(rates), len(seeds), 4, n), dtype=bool)
    aurocs = np.full((len(rates), len(seeds), 2), np.nan)
    assignments = []
    clean_source_scores = np.sum(views * indexed[None, :, :], axis=-1)
    for ri, rate in enumerate(rates):
        for si, seed in enumerate(seeds):
            donor, selected, attempts = make_assignment(artists, rate, seed)
            donor_indices[ri, si] = donor
            selected_masks[ri, si, selected] = True
            assigned = indexed[donor]
            scores = np.sum(views * assigned[None, :, :], axis=-1)
            accepted = scores >= thresholds[:, None]
            random = random_gates({'first': int((~accepted[0]).sum()),
                                   'mean12': int((~accepted[1]).sum())}, n, seed)
            all_gates = np.stack([*accepted, random['first'], random['mean12']])
            gates[ri, si] = all_gates
            verification_scores[ri, si] = scores
            T = clean_T[:, donor]
            scores_by_condition = {'audio_only': A, 'caption_only': T,
                                   'ungated_fusion': fusion_scores(A, T, weight),
                                   'tfidf': lexical_clean[:, donor]}
            for name, g in zip(['gate_first', 'gate_mean12', 'random_gate_matched_first', 'random_gate_matched_mean12'], all_gates):
                scores_by_condition[name] = fusion_scores(A, T, weight, g)
            for ci, condition in enumerate(conditions):
                ranks[ri, si, ci] = rank_metrics(scores_by_condition[condition], ids, sources)['ranks']
            if len(selected):
                labels = np.r_[np.ones(len(selected)), np.zeros(len(selected))]
                for vi in range(2):
                    aurocs[ri, si, vi] = roc_auc_score(labels, np.r_[clean_source_scores[vi, selected], scores[vi, selected]])
            assignments.append({'rate': rate, 'seed': seed, 'selected_count': len(selected),
                                'permutation_attempts': attempts, 'rejected_permutations': max(attempts-1, 0),
                                'selected_track_ids': [ids[j] for j in selected],
                                'target_to_donor': {ids[j]: ids[donor[j]] for j in selected}})
        print(f'Completed p={rate}, {len(seeds)} seeds, {len(conditions)} conditions', flush=True)
    q_weights = cluster_weights(query_artists, boot['resamples'], boot['seed'])
    c_weights = cluster_weights(artists, boot['resamples'], boot['seed'])
    save_npz(OUT/'raw_arrays.npz', ranks=ranks, donor_indices=donor_indices,
             selected_mask=selected_masks, verification_scores=verification_scores, gates=gates,
             aurocs=aurocs, catalog_ids=np.array(ids), query_source_ids=np.array(sources),
             catalog_artist_ids=np.array(artists), query_artist_ids=np.array(query_artists))
    save_npz(OUT/'bootstrap_weights.npz', query=q_weights, catalog=c_weights)
    atomic_json(OUT/'assignments.json', assignments)
    rr = 1.0/ranks
    metrics = {'MRR': rr, 'R1': (ranks <= 1).astype(float),
               'R5': (ranks <= 5).astype(float), 'R10': (ranks <= 10).astype(float)}
    retrieval, csv_retrieval, per_seed = [], [], []
    for ri, rate in enumerate(rates):
        for ci, condition in enumerate(conditions):
            row = {'rate': rate, 'condition': condition, 'metrics': {}}
            flat = {'rate': rate, 'condition': condition}
            for metric, values in metrics.items():
                result = interval(values[ri, :, ci].mean(axis=0), q_weights, .95)
                row['metrics'][metric] = result
                for k in ['estimate', 'lower', 'upper']:
                    flat[f'{metric}_{k}'] = result[k]
            retrieval.append(row); csv_retrieval.append(flat)
            for si, seed in enumerate(seeds):
                per_seed.append({'rate': rate, 'condition': condition, 'seed': seed,
                                 **{k: float(v[ri, si, ci].mean()) for k, v in metrics.items()}})
    ci_first, ci_mean = conditions.index('gate_first'), conditions.index('gate_mean12')
    h1 = (~gates[0, 0, 0]).astype(float) - (~gates[0, 0, 1]).astype(float)
    h2 = (rr[1, :, ci_mean] - rr[1, :, ci_first]).mean(axis=0)
    primary = {
        'H1': {'definition': conf['primary_effects']['H1'], 'unit': 'catalog track',
               'tracks': n, 'artists': len(set(artists)),
               **interval(h1, c_weights, boot['primary_ci_level'])},
        'H2': {'definition': conf['primary_effects']['H2'], 'unit': 'query track',
               'tracks': nq, 'artists': len(set(query_artists)),
               **interval(h2, q_weights, boot['primary_ci_level'])},
    }
    secondary = []
    for ri, rate in enumerate(rates):
        for left, right in [('gate_mean12', 'gate_first'), ('gate_mean12', 'random_gate_matched_mean12'),
                            ('gate_first', 'random_gate_matched_first'), ('gate_mean12', 'audio_only'),
                            ('gate_first', 'audio_only'), ('ungated_fusion', 'audio_only')]:
            diff = (rr[ri, :, conditions.index(left)] - rr[ri, :, conditions.index(right)]).mean(axis=0)
            secondary.append({'rate': rate, 'contrast': f'{left} minus {right}',
                              **interval(diff, q_weights, .95)})
    verifier_results = []
    for vi, view in enumerate(VIEW_NAMES):
        clean_reject = (~gates[0, 0, vi]).astype(float)
        swap_accept = [float(gates[1, si, vi, selected_masks[1, si]].mean()) for si in range(len(seeds))]
        verifier_results.append({'verifier': view, 'threshold': float(thresholds[vi]),
                                 'clean_reject_count': int(clean_reject.sum()),
                                 'clean_rejection_rate': interval(clean_reject, c_weights, .95),
                                 'swapped_caption_acceptance_mean': float(np.mean(swap_accept)),
                                 'swapped_caption_acceptance_by_seed': swap_accept,
                                 'paired_track_source_auroc_mean': float(aurocs[1, :, vi].mean()),
                                 'paired_track_source_auroc_by_seed': aurocs[1, :, vi].tolist()})
    cases = []
    for tid in conf['case_track_ids']:
        qi, j = sources.index(tid), ids.index(tid)
        case = {'track_id': tid, 'artist_id': artists[j], 'index_caption': items[j]['captions'][0],
                'query_caption': items[j]['captions'][1],
                'clean_source_scores': dict(zip(VIEW_NAMES, map(float, clean_source_scores[:, j]))),
                'clean_accepted': dict(zip(VIEW_NAMES, map(bool, gates[0, 0, :2, j]))),
                'swap_target_seeds': [seed for si, seed in enumerate(seeds) if selected_masks[1, si, j]],
                'rank_by_rate_condition_seed': {}}
        for ri, rate in enumerate(rates):
            case['rank_by_rate_condition_seed'][str(rate)] = {name: ranks[ri, :, ci, qi].tolist() for ci, name in enumerate(conditions)}
        cases.append(case)
    result = {
        'status': 'completed', 'scope': 'frozen held-out evaluation', 'date': '2026-11-20',
        'catalog_tracks': n, 'query_tracks': nq, 'catalog_artists': len(set(artists)),
        'query_artists': len(set(query_artists)), 'rates': rates, 'seeds': seeds,
        'conditions': conditions, 'view_order': VIEW_NAMES, 'gate_order': GATE_NAMES,
        'selected_lambda': weight, 'thresholds': frozen['thresholds'],
        'corruption_count_25pct': int(selected_masks[1, 0].sum()),
        'random_ranking_expectation': {'MRR': sum(1/i for i in range(1, n+1))/n,
                                       'R1': 1/n, 'R5': 5/n, 'R10': 10/n},
        'retrieval': retrieval, 'primary_effects': primary, 'secondary_MRR_contrasts': secondary,
        'verifiers': verifier_results, 'tfidf_vocabulary_size': len(lexical.vocabulary_),
        'bootstrap': run_spec['bootstrap'], 'secondary_ci_level': .95,
        'limitations': ['Fixed catalog, designated 20 corruption seeds, and frozen development settings.',
                        'Seed Monte Carlo, development-selection, and model-training uncertainty are not integrated.',
                        'Caption swaps are provenance errors, not verified false musical descriptions.',
                        'Source-track recovery is not listener recommendation quality.',
                        'Short audio exclusions limit the target population; one model and one dataset.',
                        'Secondary intervals are descriptive and not family-wise adjusted.'],
        'evaluation_settings_retuned': False,
        'artifact_sha256': {n: sha(OUT/n) for n in ['run_spec.json', 'raw_arrays.npz', 'bootstrap_weights.npz', 'assignments.json']},
    }
    # Deterministic result must repeat exactly; wall time is kept separately.
    locked_json(OUT/'results.json', result)
    atomic_json(OUT/'cases.json', cases)
    write_csv(OUT/'retrieval.csv', csv_retrieval)
    write_csv(OUT/'per_seed_retrieval.csv', per_seed)
    write_csv(OUT/'primary_effects.csv', [{'hypothesis': k, **v} for k, v in primary.items()])
    write_csv(OUT/'secondary_MRR_contrasts.csv', secondary)
    atomic_json(OUT/'runtime.json', {'elapsed_seconds': time.perf_counter()-start,
                                    'gpu_inference': False, 'paid_api': False,
                                    'evaluation_feature_files_opened': len(items),
                                    'development_feature_files_opened': 0})
    print(json.dumps({'primary_effects': primary, 'verifiers': verifier_results}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regenerated-features', type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.regenerated_features)
