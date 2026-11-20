"""Regenerate the original CLAP features locally from licensed source audio.

Dense features, audio, and weights are runtime outputs, not repository assets.
The original 439-track plan and 31 duration exclusions are verified unchanged.
Use reproduce.py --regenerated-features FEATURE_DIR to audit a completed cache.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

import numpy as np

from pipeline import atomic_json, canonical_hash, sha

MODEL_ID = 'laion/clap-htsat-unfused'
MODEL_REVISION = '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a'
WEIGHT_SHA256 = '1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb'
WINDOW_SAMPLES = 480000
REQUIRED_SAMPLES = 12 * WINDOW_SAMPLES
MAX_PADDING_SAMPLES = 2400


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def padding_plan(sample_count):
    """Original nearest-boundary rule, including Python's round semantics."""
    nearest = round(sample_count / WINDOW_SAMPLES) * WINDOW_SAMPLES
    shortfall = nearest - sample_count
    padding = shortfall if 0 < shortfall <= MAX_PADDING_SAMPLES else 0
    return padding, sample_count + padding >= REQUIRED_SAMPLES


def separate_output(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if data == output or data in output.parents or output in data.parents:
        raise ValueError('Runtime output must be outside the immutable input directory')
    return output


def prepare_weights(weights_dir, original_spec, allow_download):
    weight = weights_dir/'pytorch_model.bin'
    required = [weight, *(weights_dir/name for name in original_spec['settings'])]
    if not all(path.is_file() for path in required):
        if not allow_download:
            raise FileNotFoundError('Pinned model files missing; allow download or supply the complete weights directory')
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            local_dir=str(weights_dir), local_dir_use_symlinks=False,
            allow_patterns=['*.json', 'merges.txt', 'pytorch_model.bin'],
        )
    if sha(weight) != WEIGHT_SHA256:
        raise ValueError('Pinned CLAP weight SHA256 mismatch; model will not be loaded')
    for name, expected in original_spec['settings'].items():
        if sha(weights_dir/name) != expected:
            raise ValueError(f'Pinned processor/model configuration SHA256 mismatch: {name}')


def input_records(data):
    conf = read(data/'minimum_experiment.json')
    frozen = read(data/'frozen_development_settings.json')
    calibration = read(data/'development_calibration.json')
    planned = read(data/'data/experiment_manifest.json')
    active = read(data/'cache/active_manifest.json')
    excluded = read(data/'cache/exclusions.json')
    original_spec = read(data/'cache/model_spec.json')
    downloads = read(data/'download_manifest.json')
    assert conf['model'] == original_spec['model'] == MODEL_ID
    assert conf['revision'] == original_spec['revision'] == MODEL_REVISION
    assert sha(data/'minimum_experiment.json') == frozen['protocol_sha256']
    assert sha(data/'planned_split.json') == conf['split_sha256'] == frozen['split_sha256']
    assert sha(data/'cache/model_spec.json') == frozen['model_spec_sha256']
    assert sha(data/'cache/active_manifest.json') == calibration['active_manifest_sha256']
    assert sha(data/'cache/exclusions.json') == calibration['exclusions_sha256']
    by_active = {str(row['track_id']): row for row in active}
    by_excluded = {str(row['track_id']): row for row in excluded}
    by_download = {str(row['track_id']): row for row in downloads['tracks']}
    planned_ids = {str(row['track_id']) for row in planned}
    assert len(planned) == len(planned_ids) == len(by_download) == 439
    assert set(by_active).isdisjoint(by_excluded)
    assert set(by_active) | set(by_excluded) == planned_ids == set(by_download)
    assert len(by_active) == 408 and len(by_excluded) == 31
    for row in planned:
        tid = str(row['track_id'])
        assert tid.isdecimal(), 'Track IDs must be safe numeric filenames'
        historical = by_active.get(tid, by_excluded.get(tid))
        assert all(row[key] == historical[key] for key in ['split', 'artist_id', 'captions'])
        expected_audio = (historical['lineage']['audio_sha256'] if tid in by_active
                          else historical['audio_sha256'])
        assert by_download[tid]['sha256'] == expected_audio
    return sorted(planned, key=lambda r: int(r['track_id'])), by_active, by_excluded, by_download, original_spec


def save_features(path, audio, text, tid, captions):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, audio=audio, text=text, track_id=tid,
                 caption_ids=[caption['caption_id'] for caption in captions])
    temporary.replace(path)


def run(data_dir, audio_dir, feature_dir, weights_dir, device='cpu', threads=4,
        allow_download=True, limit=None):
    if not __debug__:
        raise RuntimeError('Run without Python -O; integrity assertions must remain enabled')
    data = Path(data_dir).resolve()
    audio_dir = Path(audio_dir).resolve()
    feature_dir = separate_output(data, feature_dir)
    weights_dir = separate_output(data, weights_dir)
    records, historical_active, historical_excluded, downloads, original_spec = input_records(data)
    if limit is not None and limit <= 0:
        raise ValueError('--limit must be positive')
    if threads <= 0:
        raise ValueError('--threads must be positive')
    selected = records if limit is None else records[:limit]
    # Refuse missing/unverified audio before downloading or loading the model.
    for row in selected:
        tid = str(row['track_id'])
        source = audio_dir/f'{tid}.2min.mp3'
        if not source.is_file():
            raise FileNotFoundError(f'Missing audio {source}; run download_audio.py first')
        assert source.stat().st_size == downloads[tid]['size'], ('audio size', tid)
        assert sha(source) == downloads[tid]['sha256'], ('audio SHA256', tid)
    weights_dir.mkdir(parents=True, exist_ok=True)
    prepare_weights(weights_dir, original_spec, allow_download)

    import torch
    import torchaudio
    import soundfile as sf
    import torch.nn.functional as F
    from transformers import ClapModel, ClapProcessor

    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchaudio', 'transformers', 'numpy', 'soundfile', 'huggingface-hub', 'tokenizers']}
    required = {'torch': '2.0.1', 'torchaudio': '2.0.2', 'transformers': '4.30.0',
                'numpy': '1.23.5', 'soundfile': '0.12.1', 'tokenizers': '0.13.3',
                'huggingface-hub': '0.17.2'}
    for package, version in required.items():
        if versions[package].split('+')[0] != version:
            raise ValueError(f'Frozen inference requires {package}=={version}, got {versions[package]}')
    runtime_device = torch.device(device)
    if runtime_device.type not in ['cpu', 'cuda']:
        raise ValueError('Supported devices: cpu or cuda')
    if runtime_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu')
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    torch.set_num_threads(threads)
    # Force the .bin file whose SHA256 was checked even if another format exists.
    model = ClapModel.from_pretrained(weights_dir, local_files_only=True,
                                      use_safetensors=False).eval().requires_grad_(False).to(runtime_device)
    processor = ClapProcessor.from_pretrained(weights_dir, local_files_only=True)
    spec = {
        'model': MODEL_ID, 'revision': MODEL_REVISION, 'weight_sha256': WEIGHT_SHA256,
        'cache_version': 'regenerated-v1', 'versions': versions, 'python': platform.python_version(),
        'device': str(runtime_device), 'threads': threads,
        'sample_rate': 48000, 'mono': 'channel arithmetic mean',
        'windows': 12, 'window_samples': WINDOW_SAMPLES,
        'padding': 'zero shortfall <=2400 samples to nearest ten-second boundary',
        'settings': original_spec['settings'], 'implementation_sha256': sha(Path(__file__)),
    }
    provenance = {
        'original_protocol_sha256': sha(data/'minimum_experiment.json'),
        'original_active_manifest_sha256': sha(data/'cache/active_manifest.json'),
        'original_frozen_settings_sha256': sha(data/'frozen_development_settings.json'),
        'model_weight_sha256': WEIGHT_SHA256, 'model': MODEL_ID, 'revision': MODEL_REVISION,
        'download_manifest_sha256': sha(data/'download_manifest.json'),
        'original_exclusions_sha256': sha(data/'cache/exclusions.json'),
        'model_spec_sha256': canonical_hash(spec),
        'scope': 'locally regenerated source features; no new parameter selection',
        'score_absolute_tolerance': 1e-5, 'metric_absolute_tolerance': 2e-14,
        'ranks_gates_assignments': 'exact match required',
    }
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache = feature_dir/'clap'
    cache.mkdir(parents=True, exist_ok=True)
    if (feature_dir/'model_spec.json').exists():
        assert read(feature_dir/'model_spec.json') == spec, 'Existing cache configuration changed; use a new feature directory'
    atomic_json(feature_dir/'model_spec.json', spec)
    atomic_json(feature_dir/'provenance.json', provenance)
    completed, excluded, failures = [], [], []
    reused = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(selected):
            tid = str(row['track_id'])
            try:
                audio_sha = downloads[tid]['sha256']
                lineage = {'spec': canonical_hash(spec), 'audio_sha256': audio_sha, 'captions': row['captions']}
                cache_key = canonical_hash(lineage)
                npz, metadata = cache/f'{tid}.npz', cache/f'{tid}.json'
                if npz.exists() and metadata.exists():
                    prior = read(metadata)
                    assert prior['cache_key'] == cache_key and prior['npz_sha256'] == sha(npz), ('stale cache', tid)
                    assert tid in historical_active, ('unexpected cached excluded track', tid)
                    completed.append(prior)
                    reused += 1
                    continue
                decoded, sample_rate = sf.read(audio_dir/f'{tid}.2min.mp3', dtype='float32', always_2d=True)
                assert np.isfinite(decoded).all()
                pcm_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
                wave = torchaudio.functional.resample(torch.from_numpy(decoded.mean(axis=1)), sample_rate, 48000)
                original_samples = len(wave)
                pad, complete = padding_plan(original_samples)
                if pad:
                    wave = F.pad(wave, (0, pad))
                assert complete == (len(wave) >= REQUIRED_SAMPLES)
                if not complete:
                    assert tid in historical_excluded, ('new duration exclusion; stop without changing the experiment', tid)
                    excluded.append({**row, 'reason': 'fewer_than_twelve_complete_windows',
                                     'original_seconds': original_samples/48000, 'padding_samples': pad,
                                     'audio_sha256': audio_sha, 'decoded_pcm_sha256': pcm_sha})
                    continue
                assert tid in historical_active, ('historically excluded audio now retained; stop', tid)
                text_features, token_counts = [], []
                for caption in row['captions']:
                    token_count = len(processor.tokenizer(caption['caption'])['input_ids'])
                    assert token_count <= 512, ('unexpected caption truncation', tid)
                    token_counts.append(token_count)
                    inputs = processor.tokenizer([caption['caption']], padding=True, truncation=True,
                                                 max_length=512, return_tensors='pt').to(runtime_device)
                    text_features.append(F.normalize(model.get_text_features(**inputs), dim=-1).cpu().numpy()[0])
                audio_features = []
                inference_start = time.perf_counter()
                for window in range(12):
                    segment = wave[window*WINDOW_SAMPLES:(window+1)*WINDOW_SAMPLES]
                    assert len(segment) == WINDOW_SAMPLES
                    inputs = processor(audios=segment.numpy(), sampling_rate=48000, return_tensors='pt').to(runtime_device)
                    audio_features.append(F.normalize(model.get_audio_features(**inputs), dim=-1).cpu().numpy()[0])
                audio_features, text_features = np.stack(audio_features), np.stack(text_features)
                assert audio_features.shape == (12, 512)
                assert text_features.shape == (len(row['captions']), 512)
                for features in [audio_features, text_features]:
                    assert features.dtype == np.float32 and np.isfinite(features).all()
                    assert np.allclose(np.linalg.norm(features, axis=-1), 1, atol=1e-5)
                save_features(npz, audio_features, text_features, tid, row['captions'])
                record = {**row, 'cache_key': cache_key, 'lineage': lineage, 'npz_sha256': sha(npz),
                          'decoded_pcm_sha256': pcm_sha, 'original_seconds': original_samples/48000,
                          'padding_samples': pad, 'source_sample_rate': sample_rate,
                          'source_channels': decoded.shape[1], 'token_counts': token_counts,
                          'audio_forward_seconds': time.perf_counter()-inference_start}
                atomic_json(metadata, record)
                completed.append(record)
            except Exception as error:
                failures.append({'track_id': tid, 'error_type': type(error).__name__, 'message': str(error)})
                atomic_json(feature_dir/'errors.json', failures)
                raise
            finally:
                if (index+1) % 10 == 0 or index+1 == len(selected):
                    print(f'Features: {index+1}/{len(selected)} checked, {len(completed)} retained, {len(excluded)} excluded', flush=True)
    complete_run = limit is None or limit >= len(records)
    if complete_run:
        assert {r['track_id'] for r in completed} == set(historical_active)
        assert {r['track_id'] for r in excluded} == set(historical_excluded)
        atomic_json(feature_dir/'manifest.json', sorted(completed, key=lambda r: int(r['track_id'])))
        atomic_json(feature_dir/'exclusions.json', sorted(excluded, key=lambda r: int(r['track_id'])))
    else:
        atomic_json(feature_dir/'partial_manifest.json', completed)
        atomic_json(feature_dir/'partial_exclusions.json', excluded)
    report = {'status': 'complete' if complete_run else 'partial', 'planned_tracks': len(records),
              'checked_tracks': len(selected), 'retained_tracks': len(completed), 'excluded_tracks': len(excluded),
              'reused_tracks': reused, 'errors': failures, 'device': str(runtime_device),
              'elapsed_seconds': time.perf_counter()-started, 'frozen_settings_updated': False,
              'caption_and_audio_features_regenerated': True, 'model_spec_sha256': canonical_hash(spec)}
    atomic_json(feature_dir/'run_report.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--limit', type=int, help='Smoke-test only; partial runs cannot be used for reproduction')
    args = parser.parse_args()
    run(args.data_dir, args.audio_dir, args.feature_dir, args.weights_dir, args.device,
        args.threads, not args.no_download, args.limit)
