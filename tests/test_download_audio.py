"""Bounded download checks; no network, audio files or weights required."""
from pathlib import Path
import hashlib
import io
import sys
import unittest
from unittest.mock import MagicMock, patch
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import download_audio


class DownloadChecks(unittest.TestCase):
    def fixture(self):
        content = b'synthetic bytes, not a music recording'
        memory = io.BytesIO()
        name = 'audio/80/4880.2min.mp3'
        with zipfile.ZipFile(memory, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, content)
        with zipfile.ZipFile(memory) as archive:
            entry = archive.getinfo(name)
            item = {'track_id': '4880', 'zip_member': name, 'size': len(content),
                    'crc32': zlib.crc32(content), 'sha256': hashlib.sha256(content).hexdigest(),
                    'zip': {'offset': entry.header_offset, 'compressed_size': entry.compress_size,
                            'compression': entry.compress_type}}
        return memory.getvalue(), item, content

    def test_zip_payload_matches_original_sha_crc_and_filename(self):
        payload, item, expected = self.fixture()
        self.assertEqual(download_audio.decode_member(payload, 0, item), expected)
        item['sha256'] = '0'*64
        with self.assertRaises(RuntimeError):
            download_audio.decode_member(payload, 0, item)

    def test_wrong_zip_member_and_truncated_payload_rejected(self):
        payload, item, _ = self.fixture()
        with self.assertRaises(RuntimeError):
            download_audio.decode_member(payload[:40], 0, item)
        item['zip_member'] = 'audio/80/9999.2min.mp3'
        with self.assertRaises(RuntimeError):
            download_audio.decode_member(payload, 0, item)

    def test_unbounded_ranges_rejected_before_network(self):
        with patch.object(download_audio.urllib.request, 'urlopen') as request:
            for start, end in [(-1, 10), (10, 9), (0, download_audio.MAX_RANGE_BYTES),
                               (0, download_audio.ARCHIVE_BYTES)]:
                with self.assertRaises(ValueError):
                    download_audio.fetch_range(start, end)
            request.assert_not_called()

    def test_ignored_range_never_reads_full_archive(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        with patch.object(download_audio.urllib.request, 'urlopen', return_value=response), \
                patch.object(download_audio.time, 'sleep'), patch('builtins.print'):
            with self.assertRaises(RuntimeError):
                download_audio.fetch_range(0, 3)
        response.read.assert_not_called()

    def test_exact_content_range_and_length_required(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 206
        response.headers = {'Content-Range': f'bytes 0-3/{download_audio.ARCHIVE_BYTES}'}
        response.read.return_value = b'abcd'
        with patch.object(download_audio.urllib.request, 'urlopen', return_value=response):
            self.assertEqual(download_audio.fetch_range(0, 3), b'abcd')
        response.read.assert_called_with(5)
        response.read.return_value = b'abc'
        with patch.object(download_audio.urllib.request, 'urlopen', return_value=response), \
                patch.object(download_audio.time, 'sleep'), patch('builtins.print'):
            with self.assertRaises(RuntimeError):
                download_audio.fetch_range(0, 3)


if __name__ == '__main__':
    unittest.main()
