"""Download pinned Song Describer audio directly from its official archive.

The repository distributes neither audio nor dense embeddings. This opt-in
runtime utility preserves individual source attribution in download_manifest.
HTTP ranges are bounded; every member must match its original SHA256 and CRC.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import struct
import time
import urllib.error
import urllib.request
import zlib

OFFICIAL_URL = 'https://zenodo.org/api/records/10072001/files/audio.zip/content'
ARCHIVE_BYTES = 3318019617
MAX_RANGE_BYTES = 32 * 1024**2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path, content):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_bytes(content)
    temporary.replace(path)


def read_manifest(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir/'download_manifest.json').read_text(encoding='utf8'))
    protocol = json.loads((data_dir/'minimum_experiment.json').read_text(encoding='utf8'))
    split_bytes = (data_dir/'planned_split.json').read_bytes()
    assert sha256(split_bytes) == protocol['split_sha256'] == manifest['split_sha256']
    assert manifest['schema_version'] == 1
    assert manifest['source_url'] == OFFICIAL_URL and manifest['zip_size'] == ARCHIVE_BYTES
    split = json.loads(split_bytes)
    expected = {str(tid) for part in ['development', 'evaluation'] for tid in split[part]['track_ids']}
    items = manifest['tracks']
    assert len(items) == manifest['planned_tracks'] == len(expected) == 439
    assert {item['track_id'] for item in items} == expected
    for item in items:
        tid = item['track_id']
        assert tid.isdecimal() and len(item['sha256']) == 64
        assert tid in split[item['split']]['track_ids']
        assert item['zip_member'].endswith('/'+tid+'.2min.mp3')
        assert item['license_record'].splitlines()[0] == item['original_audio_path']
        assert 0 <= item['zip']['offset'] < ARCHIVE_BYTES
        assert item['zip']['compression'] in [0, 8]
        assert 0 < item['size'] <= 20*1024**2
        assert 0 < item['zip']['compressed_size'] <= 20*1024**2
    return manifest


def fetch_range(start, end):
    if not 0 <= start <= end < ARCHIVE_BYTES or end-start+1 > MAX_RANGE_BYTES:
        raise ValueError('Unbounded or invalid HTTP range')
    expected = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
    for attempt in range(6):
        try:
            request = urllib.request.Request(OFFICIAL_URL, headers={
                'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                'User-Agent': 'music-caption-gating-reproduction/1.0',
            })
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('Server did not honor the exact bounded Range request')
                content = response.read(end-start+2)
            if len(content) != end-start+1:
                raise RuntimeError('Incomplete or oversized HTTP range response')
            return content
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            if attempt == 5:
                raise
            delay = min(60, 2**(attempt+1))
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                retry_after = error.headers.get('Retry-After', '30')
                if retry_after.isdecimal():
                    delay = max(delay, min(120, int(retry_after)))
            print(f'Range retry {attempt+1}/5: {type(error).__name__}; wait {delay}s', flush=True)
            time.sleep(delay)


def verify_audio(content, item):
    return (len(content) == item['size'] and zlib.crc32(content) == item['crc32']
            and sha256(content) == item['sha256'])


def member_end(item):
    entry = item['zip']
    # ZIP filename and extra-field lengths are uint16. Bound the possible
    # extra field rather than requesting a complete archive or a second header.
    return min(ARCHIVE_BYTES-1, entry['offset']+30+len(item['zip_member'].encode('utf8'))
               +65535+entry['compressed_size']-1)


def decode_member(ranged, start, item):
    entry = item['zip']
    offset = entry['offset']-start
    header = struct.unpack('<4s5H3L2H', ranged[offset:offset+30])
    if header[0] != b'PK\x03\x04' or header[3] != entry['compression'] or header[2] & 1:
        raise RuntimeError('Unexpected or encrypted ZIP local header')
    name = ranged[offset+30:offset+30+header[-2]].decode('utf8' if header[2]&0x800 else 'cp437')
    if name != item['zip_member']:
        raise RuntimeError('ZIP member filename mismatch')
    begin = offset+30+header[-2]+header[-1]
    compressed = ranged[begin:begin+entry['compressed_size']]
    if len(compressed) != entry['compressed_size']:
        raise RuntimeError('Truncated compressed ZIP member')
    content = zlib.decompress(compressed, -15) if entry['compression'] == 8 else compressed
    if not verify_audio(content, item):
        raise RuntimeError('ZIP size/CRC or original audio SHA256 mismatch: '+item['track_id'])
    return content


def download_batch(batch, audio_dir):
    start = batch[0]['zip']['offset']
    end = max(member_end(item) for item in batch)
    ranged = fetch_range(start, end)
    finished = []
    for item in batch:
        content = decode_member(ranged, start, item)
        path = audio_dir/(item['track_id']+'.2min.mp3')
        atomic_bytes(path, content)
        finished.append(item['track_id'])
    return finished


def download_all(data_dir, audio_dir, workers=4, track_ids=None):
    if not 1 <= workers <= 8:
        raise ValueError('workers must be in 1..8')
    manifest = read_manifest(data_dir)
    items = manifest['tracks']
    if track_ids is not None:
        requested = set(map(str, track_ids))
        if not requested or not requested <= {item['track_id'] for item in items}:
            raise ValueError('Requested track IDs are not in the frozen split')
        items = [item for item in items if item['track_id'] in requested]
    audio_dir = Path(audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    reused, remaining = [], []
    for item in items:
        path = audio_dir/(item['track_id']+'.2min.mp3')
        if path.is_file() and verify_audio(path.read_bytes(), item):
            reused.append(item['track_id'])
        else:
            remaining.append(item)
    needed = sum(item['size'] for item in remaining)
    if shutil.disk_usage(audio_dir).free < needed + 512*1024**2:
        raise RuntimeError('Insufficient free disk space for the requested audio')
    batches, batch = [], []
    for item in sorted(remaining, key=lambda row: row['zip']['offset']):
        if batch and member_end(item)-batch[0]['zip']['offset']+1 > MAX_RANGE_BYTES:
            batches.append(batch)
            batch = []
        batch.append(item)
    if batch:
        batches.append(batch)
    batches.sort(key=lambda group: (not any(row['split'] == 'development' for row in group),
                                   group[0]['zip']['offset']))
    started = time.perf_counter()
    completed, failures = list(reused), []
    print(f'Audio requested={len(items)}, verified reused={len(reused)}, bounded batches={len(batches)}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_batch, group, audio_dir): group for group in batches}
        for future in as_completed(futures):
            try:
                completed.extend(future.result())
            except Exception as error:
                failed = [item['track_id'] for item in futures[future]]
                failures.append({'track_ids': failed, 'error_type': type(error).__name__})
                print('Audio batch failed: '+','.join(failed)+' '+type(error).__name__, flush=True)
            print(f'Audio progress={len(completed)}/{len(items)}, failed_batches={len(failures)}', flush=True)
    report = {'requested': len(items), 'completed': len(completed), 'reused': len(reused),
              'completed_track_ids': sorted(completed, key=int), 'failures': failures,
              'elapsed_seconds': time.perf_counter()-started, 'source_url': OFFICIAL_URL,
              'verification': 'exact HTTP range, ZIP name/size/CRC, original audio SHA256'}
    atomic_bytes(audio_dir/'download_report.json', json.dumps(report, indent=2).encode('utf8'))
    if failures or len(completed) != len(items):
        raise RuntimeError('Audio download incomplete; rerun to reuse verified files')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--track-id', action='append', default=None)
    arguments = parser.parse_args()
    print(json.dumps(download_all(arguments.data_dir, arguments.audio_dir,
                                  arguments.workers, arguments.track_id), indent=2))


if __name__ == '__main__':
    main()
