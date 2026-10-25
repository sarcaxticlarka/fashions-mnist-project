"""Regression checks for IDX parsing; no dataset download or training needed."""
import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_loader import _read_images, _read_labels


class IDXReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sample.gz"

    def write_payload(self, payload):
        with gzip.open(self.path, "wb") as stream:
            stream.write(payload)

    def test_valid_images_preserve_shape_dtype_and_pixels(self):
        images = np.arange(2 * 28 * 28, dtype=np.uint16).astype(np.uint8).reshape(2, 28, 28)
        self.write_payload(struct.pack(">IIII", 2051, 2, 28, 28) + images.tobytes())
        actual = _read_images(self.path)
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual, images)

    def test_valid_labels_preserve_values(self):
        self.write_payload(struct.pack(">II", 2049, 3) + bytes([0, 6, 9]))
        actual = _read_labels(self.path)
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_array_equal(actual, [0, 6, 9])

    def test_incomplete_headers_are_rejected(self):
        for reader, size in ((_read_images, 16), (_read_labels, 8)):
            with self.subTest(reader=reader.__name__):
                self.write_payload(bytes(size - 1))
                with self.assertRaisesRegex(ValueError, "incomplete IDX"):
                    reader(self.path)

    def test_wrong_magic_numbers_are_rejected(self):
        for reader, header in (
            (_read_images, struct.pack(">IIII", 2049, 1, 28, 28)),
            (_read_labels, struct.pack(">II", 2051, 1)),
        ):
            with self.subTest(reader=reader.__name__):
                self.write_payload(header)
                with self.assertRaisesRegex(ValueError, "magic"):
                    reader(self.path)

    def test_wrong_image_dimensions_are_rejected(self):
        self.write_payload(struct.pack(">IIII", 2051, 1, 14, 56) + bytes(784))
        with self.assertRaisesRegex(ValueError, "expected 28x28"):
            _read_images(self.path)

    def test_truncated_and_extra_payloads_are_rejected(self):
        for reader, header, size in (
            (_read_images, struct.pack(">IIII", 2051, 2, 28, 28), 1568),
            (_read_labels, struct.pack(">II", 2049, 2), 2),
        ):
            for delta in (-1, 1):
                with self.subTest(reader=reader.__name__, delta=delta):
                    self.write_payload(header + bytes(size + delta))
                    with self.assertRaisesRegex(ValueError, "payload size"):
                        reader(self.path)


if __name__ == "__main__":
    unittest.main()
