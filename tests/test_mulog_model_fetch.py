"""Offline tests for extracting only MuLoG-DRUNet's generic ZIP member.

The fixtures are small in-memory ZIPs, never actual checkpoints. No network,
PyTorch import, GPU execution, or official asset mutation is needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MEMBER = "models/generic_model.pth"
OTHER_MEMBER = "models/unused_model.pth"


def load_fetch_module():
    path = ROOT / "scripts" / "mulog_drunet" / "fetch_official.py"
    spec = importlib.util.spec_from_file_location("_test_mulog_model_fetch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fetch module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_archive(payload: bytes) -> tuple[bytes, zipfile.ZipInfo]:
    buffer = io.BytesIO()
    # Stored DEFLATE blocks make an equal-length data corruption deterministic.
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=0) as archive:
        archive.writestr(MEMBER, payload)
        archive.writestr(OTHER_MEMBER, b"This unrelated model must not be read.")
        info = archive.getinfo(MEMBER)
    return buffer.getvalue(), info


class PrefixBoundaryReader(io.BytesIO):
    """Fail if a consumer attempts to read any byte of subsequent members."""

    def __init__(self, data: bytes, boundary: int):
        super().__init__(data)
        self.boundary = boundary
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0 or self.tell() + size > self.boundary:
            raise AssertionError("Attempted to read beyond the generic member")
        data = super().read(size)
        self.bytes_read += len(data)
        return data


class GenericModelPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetch = load_fetch_module()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = bytes(range(256)) * 8 + b"fake generic checkpoint"
        self.archive, info = make_archive(self.payload)
        header = struct.unpack("<IHHHHHIIIHH", self.archive[:30])
        self.data_offset = 30 + header[-2] + header[-1]
        self.prefix_size = self.data_offset + info.compress_size
        self.prefix = self.root / "generic_model.zip-prefix"
        self.target = self.root / MEMBER
        self.extracting = self.target.with_suffix(".pth.extracting")
        constants = mock.patch.multiple(
            self.fetch,
            ROOT=self.root,
            MODEL_MEMBER=MEMBER,
            MODEL_COMPRESSED_SIZE=info.compress_size,
            MODEL_SIZE=len(self.payload),
            MODEL_CRC32=zlib.crc32(self.payload),
            MODEL_PREFIX_SIZE=self.prefix_size,
            GENERIC_MODEL_SHA256=None,
        )
        constants.start()
        self.addCleanup(constants.stop)
        network = mock.patch.object(
            self.fetch.urllib.request,
            "urlopen",
            side_effect=AssertionError("Offline extraction must not use the network"),
        )
        self.urlopen = network.start()
        self.addCleanup(network.stop)

    def assert_no_extracting_file(self):
        self.assertFalse(self.extracting.exists())

    def test_full_zip_reads_only_first_member_and_ignores_other_members(self):
        self.prefix.write_bytes(self.archive)
        reader = PrefixBoundaryReader(self.archive, self.prefix_size)
        original_open = Path.open

        def bounded_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path == self.prefix and mode == "rb":
                return reader
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", autospec=True, side_effect=bounded_open):
            result = self.fetch.extract_generic_from_prefix(self.prefix)

        self.assertEqual(reader.bytes_read, self.prefix_size)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(list((self.root / "models").iterdir()), [self.target])
        self.assertFalse((self.root / OTHER_MEMBER).exists())
        self.assertEqual(self.prefix.read_bytes(), self.archive)
        self.assertEqual(result["prefix_bytes_used"], self.prefix_size)
        self.urlopen.assert_not_called()
        self.assert_no_extracting_file()

    def test_exact_member_prefix_needs_no_central_directory(self):
        prefix_bytes = self.archive[:self.prefix_size]
        self.prefix.write_bytes(prefix_bytes)
        self.assertFalse(zipfile.is_zipfile(self.prefix))

        result = self.fetch.extract_generic_from_prefix(self.prefix)

        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(self.prefix.read_bytes(), prefix_bytes)
        self.assertEqual(result["member"], MEMBER)
        self.assertEqual(result["member_size_bytes"], len(self.payload))
        self.assertEqual(result["member_crc32"], zlib.crc32(self.payload))
        self.assertTrue(result["member_crc32_verified"])
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertFalse(result["checkpoint_sha256_matches_pin"])
        self.assertFalse(result["complete_archive_git_blob_verified"])
        self.assertFalse(result["prefix_is_complete_zip_archive"])
        self.assert_no_extracting_file()

    def test_corrupted_payload_crc_is_rejected_and_prefix_retained(self):
        changed = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        changed_archive, changed_info = make_archive(changed)
        self.assertEqual(changed_info.compress_size, self.fetch.MODEL_COMPRESSED_SIZE)
        # Preserve the pinned original local header/CRC, but swap the valid
        # DEFLATE payload: decompression succeeds and only its CRC differs.
        damaged = (self.archive[:self.data_offset]
                   + changed_archive[self.data_offset:self.prefix_size])
        self.prefix.write_bytes(damaged)

        with self.assertRaisesRegex(ValueError, "CRC32"):
            self.fetch.extract_generic_from_prefix(self.prefix)

        self.assertEqual(self.prefix.read_bytes(), damaged)
        self.assertFalse(self.target.exists())
        self.assert_no_extracting_file()

    def test_incomplete_prefix_is_rejected_without_changing_progress(self):
        for length in (0, 29, self.data_offset, self.prefix_size - 1):
            with self.subTest(length=length):
                partial = self.archive[:length]
                self.prefix.write_bytes(partial)
                with self.assertRaisesRegex(ValueError, "Prefix incomplete"):
                    self.fetch.extract_generic_from_prefix(self.prefix)
                self.assertEqual(self.prefix.read_bytes(), partial)
                self.assertFalse(self.target.exists())
                self.assert_no_extracting_file()

    def test_existing_different_checkpoint_is_not_overwritten(self):
        self.prefix.write_bytes(self.archive[:self.prefix_size])
        self.target.parent.mkdir(parents=True)
        existing = b"user-owned different checkpoint"
        self.target.write_bytes(existing)

        with self.assertRaisesRegex(ValueError, "Existing checkpoint differs"):
            self.fetch.extract_generic_from_prefix(self.prefix)

        self.assertEqual(self.target.read_bytes(), existing)
        self.assertEqual(self.prefix.read_bytes(), self.archive[:self.prefix_size])
        self.assert_no_extracting_file()

    def test_existing_identical_checkpoint_can_be_extracted_repeatedly(self):
        self.prefix.write_bytes(self.archive[:self.prefix_size])
        first = self.fetch.extract_generic_from_prefix(self.prefix)
        # A fixed historical timestamp detects unnecessary replacement, even
        # if both calls would otherwise finish within one filesystem time tick.
        os.utime(self.target, ns=(1_600_000_000_000_000_000,) * 2)
        retained_mtime = self.target.stat().st_mtime_ns

        second = self.fetch.extract_generic_from_prefix(self.prefix)
        third = self.fetch.extract_generic_from_prefix(self.prefix)

        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(self.target.stat().st_mtime_ns, retained_mtime)
        self.assert_no_extracting_file()

    def test_known_sha_pin_is_checked_when_supplied(self):
        self.prefix.write_bytes(self.archive[:self.prefix_size])
        expected = hashlib.sha256(self.payload).hexdigest()
        with mock.patch.object(self.fetch, "GENERIC_MODEL_SHA256", expected):
            result = self.fetch.extract_generic_from_prefix(self.prefix)
        self.assertTrue(result["checkpoint_sha256_matches_pin"])
        self.assertEqual(result["checkpoint_sha256"], expected)
        self.assert_no_extracting_file()

    def test_wrong_sha_pin_is_rejected_without_publishing_checkpoint(self):
        self.prefix.write_bytes(self.archive[:self.prefix_size])
        with mock.patch.object(self.fetch, "GENERIC_MODEL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.fetch.extract_generic_from_prefix(self.prefix)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.prefix.read_bytes(), self.archive[:self.prefix_size])
        self.assert_no_extracting_file()


if __name__ == "__main__":
    unittest.main()
