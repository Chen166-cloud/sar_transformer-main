"""Offline prefix-fetch regression tests using tiny synthetic ZIP members only."""
from __future__ import annotations

import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zlib

import fetch_official as fetch


def zip_prefix(data: bytes, crc: int | None = None) -> tuple[bytes, int]:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(data) + compressor.flush()
    name = fetch.MODEL_MEMBER.encode("ascii")
    header = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0,
                         zlib.crc32(data) if crc is None else crc, len(compressed), len(data), len(name), 0)
    return header + name + compressed, len(compressed)


class PrefixFetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = (b"synthetic checkpoint bytes for extraction tests only\n" * 17)
        self.prefix, compressed = zip_prefix(self.data)
        self.scope = patch.multiple(fetch, ROOT=self.root, MODEL_COMPRESSED_SIZE=compressed,
                                    MODEL_SIZE=len(self.data), MODEL_CRC32=zlib.crc32(self.data),
                                    MODEL_PREFIX_SIZE=len(self.prefix), GENERIC_MODEL_SHA256=None)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.addCleanup(self.directory.cleanup)

    def write_prefix(self, data=None):
        path = self.root / "prefix.cache"
        path.write_bytes(self.prefix if data is None else data)
        return path

    def test_reader_stops_at_prefix_boundary(self):
        response = io.BytesIO(self.prefix + b"do not transfer remaining models")
        output = io.BytesIO()
        fetch.read_limited_response(response, output, len(self.prefix), "test")
        self.assertEqual(output.getvalue(), self.prefix)
        self.assertEqual(response.tell(), len(self.prefix))

    def test_reader_rejects_short_response(self):
        with self.assertRaisesRegex(ValueError, "Truncated"):
            fetch.read_limited_response(io.BytesIO(b"abc"), io.BytesIO(), 4, "test")

    def test_extract_only_first_member_and_record_honest_provenance(self):
        path = self.write_prefix(self.prefix + b"unread suffix from other ZIP members")
        record = fetch.extract_generic_from_prefix(path)
        self.assertEqual((self.root / fetch.MODEL_MEMBER).read_bytes(), self.data)
        self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertTrue(record["member_crc32_verified"])
        self.assertFalse(record["complete_archive_git_blob_verified"])
        self.assertFalse(record["checkpoint_sha256_matches_pin"])
        self.assertFalse(record["prefix_is_complete_zip_archive"])
        self.assertEqual(record["prefix_bytes_used"], len(self.prefix))

    def test_pinned_checkpoint_sha_is_enforced(self):
        with patch.object(fetch, "GENERIC_MODEL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetch.extract_generic_from_prefix(self.write_prefix())
        self.assertFalse((self.root / fetch.MODEL_MEMBER).exists())
        self.assertFalse((self.root / fetch.MODEL_MEMBER).with_suffix(".pth.extracting").exists())

    def test_final_model_reuse_does_not_need_prefix_cache(self):
        prefix = self.write_prefix()
        record = fetch.extract_generic_from_prefix(prefix)
        fetch.write_manifest({}, record)
        prefix.unlink()
        with patch.object(fetch, "ensure_source", return_value={}), patch.object(fetch.urllib.request, "urlopen") as network:
            fetch.fetch_assets()
            network.assert_not_called()
        self.assertEqual((self.root / fetch.MODEL_MEMBER).read_bytes(), self.data)

    def test_streaming_extraction_handles_highly_compressed_large_output(self):
        data = b"A" * (3 * 1024 * 1024 + 17)
        prefix, compressed = zip_prefix(data)
        with patch.multiple(fetch, MODEL_COMPRESSED_SIZE=compressed, MODEL_SIZE=len(data),
                            MODEL_CRC32=zlib.crc32(data), MODEL_PREFIX_SIZE=len(prefix)):
            fetch.extract_generic_from_prefix(self.write_prefix(prefix))
        self.assertEqual((self.root / fetch.MODEL_MEMBER).read_bytes(), data)

    def test_corrupted_member_rejected_by_crc(self):
        changed = b"X" + self.data[1:]
        modified, compressed = zip_prefix(changed, zlib.crc32(self.data))
        with patch.multiple(fetch, MODEL_COMPRESSED_SIZE=compressed, MODEL_PREFIX_SIZE=len(modified)):
            with self.assertRaisesRegex(ValueError, "CRC32"):
                fetch.extract_generic_from_prefix(self.write_prefix(modified))
        self.assertFalse((self.root / fetch.MODEL_MEMBER).exists())

    def test_truncated_prefix_preserved(self):
        path = self.write_prefix(self.prefix[:-1])
        with self.assertRaisesRegex(ValueError, "Prefix incomplete"):
            fetch.extract_generic_from_prefix(path)
        self.assertEqual(path.read_bytes(), self.prefix[:-1])

    def test_unexpected_zip_header_rejected(self):
        bad = bytearray(self.prefix)
        bad[6] = 1  # encrypted flag is never accepted
        with self.assertRaisesRegex(ValueError, "header"):
            fetch.extract_generic_from_prefix(self.write_prefix(bytes(bad)))

    def test_existing_different_checkpoint_is_preserved(self):
        target = self.root / fetch.MODEL_MEMBER
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing unrelated file")
        with self.assertRaisesRegex(ValueError, "Existing checkpoint differs"):
            fetch.extract_generic_from_prefix(self.write_prefix())
        self.assertEqual(target.read_bytes(), b"existing unrelated file")

    def test_existing_extraction_temporary_is_not_deleted(self):
        target = (self.root / fetch.MODEL_MEMBER).with_suffix(".pth.extracting")
        target.parent.mkdir(parents=True)
        target.write_bytes(b"another extraction owns this")
        with self.assertRaises(FileExistsError):
            fetch.extract_generic_from_prefix(self.write_prefix())
        self.assertEqual(target.read_bytes(), b"another extraction owns this")

    def test_incomplete_download_never_restarted_implicitly(self):
        path = self.root / "models.zip.download"
        path.write_bytes(b"existing partial progress")
        with patch.object(fetch, "ensure_source", return_value={}), patch.object(fetch.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "Incomplete prefix retained"):
                fetch.fetch_assets()
            network.assert_not_called()
        self.assertEqual(path.read_bytes(), b"existing partial progress")


if __name__ == "__main__":
    unittest.main()
