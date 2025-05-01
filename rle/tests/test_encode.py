"""Tests for RLE encoding."""

import numpy as np
import pytest
import sys


try:
    from pydicom import dcmread
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import generate_frames
    from pydicom.pixels.utils import pixel_dtype, reshape_pixel_array
    from pydicom.uid import RLELossless

    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False

try:
    from pydicom.pixels.decoders.native import _rle_decode_frame
except ImportError:
    from pydicom.pixels.decoders.rle import _rle_decode_frame


from rle.data import get_indexed_datasets
from rle.rle import (
    encode_row,
    encode_segment,
    decode_segment,
    encode_frame,
    decode_frame,
)


INDEX_RLE = get_indexed_datasets("1.2.840.10008.1.2.5")
INDEX_LEI = get_indexed_datasets("1.2.840.10008.1.2")
INDEX_LEE = get_indexed_datasets("1.2.840.10008.1.2.1")


# Tests for RLE encoding
REFERENCE_ENCODE_ROW = [
    # Input, output
    pytest.param([], b"", id="0"),
    # Replicate run tests
    # 2 (min) replicate, could also be a 2 (min) literal 0x00 0x00
    pytest.param([0] * 2, b"\xff\x00", id="1"),
    pytest.param([255] * 2, b"\xff\xFF", id="1b"),
    pytest.param([0] * 3, b"\xfe\x00", id="2"),
    pytest.param([0] * 64, b"\xc1\x00", id="3"),
    pytest.param([0] * 127, b"\x82\x00", id="4"),
    # 128 (max) replicate
    pytest.param([0] * 128, b"\x81\x00", id="5"),
    # 128 (max) replicate, 1 (min) literal
    pytest.param([0] * 129, b"\x81\x00\x00\x00", id="6"),
    # 128 (max) replicate, 2 (min) replicate
    pytest.param([0] * 130, b"\x81\x00\xff\x00", id="7"),
    # 128 (max) x 5 replicates
    pytest.param([0] * 128 * 5, b"\x81\x00" * 5, id="8"),
    # Literal run tests
    # 1 (min) literal
    pytest.param([0], b"\x00\x00", id="9"),
    pytest.param([255], b"\x00\xff", id="9b"),
    pytest.param([0, 1], b"\x01\x00\x01", id="10"),
    pytest.param([0, 1, 2], b"\x02\x00\x01\x02", id="11"),
    pytest.param([0, 1] * 32, b"\x3f" + b"\x00\x01" * 32, id="12"),
    # 127 literal
    pytest.param([0, 1] * 63 + [2], b"\x7e" + b"\x00\x01" * 63 + b"\x02", id="13"),
    # 128 literal (max)
    pytest.param([0, 1] * 64, b"\x7f" + b"\x00\x01" * 64, id="14"),
    # 128 (max) literal, 1 (min) literal
    pytest.param([0, 1] * 64 + [2], b"\x7f" + b"\x00\x01" * 64 + b"\x00\x02", id="15"),
    # 128 (max) x 5 literals
    pytest.param([0, 1] * 64 * 5, (b"\x7f" + b"\x00\x01" * 64) * 5, id="16"),
    # Combination run tests
    # 2 literal, 1(min) literal
    # pytest.param([0, 1, 1], b"\x01\x00\x01\x00\x01", id="17"),
    # or 1 (min) literal, 1 (min) replicate b'\x00\x00\xff\x01' <-- this
    pytest.param([0, 1, 1], b"\x00\x00\xff\x01", id="17"),
    # 2 literal, 127 replicate
    # pytest.param([0] + [1] * 128, b"\x01\x00\x01\x82\x01", id="18"),
    # or 1 (min) literal, 128 (max) replicate  <-- this
    pytest.param([0] + [1] * 128, b"\x00\x00\x81\x01", id="18"),
    # 2 literal, 128 (max) replicate
    # pytest.param([0] + [1] * 129, b"\x01\x00\x01\x81\x01", id="18b"),
    # or 1 (min) literal, 128 (max) replicate, 1 (min) literal  <-- this
    pytest.param([0] + [1] * 129, b"\x00\x00\x81\x01\x00\x01", id="18b"),
    # 128 (max) literal, 2 (min) replicate
    # 128 (max literal)
    pytest.param(
        [0, 1] * 64 + [2] * 2, b"\x7f" + b"\x00\x01" * 64 + b"\xff\x02", id="19"
    ),
    # 128 (max) literal, 128 (max) replicate
    pytest.param(
        [0, 1] * 64 + [2] * 128, b"\x7f" + b"\x00\x01" * 64 + b"\x81\x02", id="20"
    ),
    # 2 (min) replicate, 1 (min) literal
    pytest.param([0, 0, 1], b"\xff\x00\x00\x01", id="21"),
    # 2 (min) replicate, 128 (max) literal
    pytest.param([0, 0] + [1, 2] * 64, b"\xff\x00\x7f" + b"\x01\x02" * 64, id="22"),
    # 128 (max) replicate, 1 (min) literal
    pytest.param([0] * 128 + [1], b"\x81\x00\x00\x01", id="23"),
    # 128 (max) replicate, 128 (max) literal
    pytest.param([0] * 128 + [1, 2] * 64, b"\x81\x00\x7f" + b"\x01\x02" * 64, id="24"),
]


class TestEncodeRow:
    """Tests for _rle.encode_row."""

    @pytest.mark.parametrize("row, ref", REFERENCE_ENCODE_ROW)
    def test_encode(self, row, ref):
        """Test encoding an empty row."""
        row = np.asarray(row, dtype="uint8")
        assert ref == encode_row(row.tobytes())


class TestEncodeSegment:
    """Tests for _rle.encode_segment."""

    def test_one_row(self):
        """Test encoding data that contains only a single row."""
        ds = INDEX_RLE["OBXXXX1A_rle.dcm"]["ds"]
        pixel_data = b"".join(generate_frames(ds.PixelData))
        decoded = decode_segment(pixel_data[64:])
        assert ds.Rows * ds.Columns == len(decoded)
        arr = np.frombuffer(decoded, "uint8").reshape(ds.Rows, ds.Columns)

        # Re-encode a single row of the decoded data
        row = arr[0]
        assert (ds.Columns,) == row.shape
        encoded = encode_segment(row.tobytes(), ds.Columns)

        # Decode the re-encoded data and check that it's the same
        redecoded = decode_segment(encoded)
        assert ds.Columns == len(redecoded)
        assert decoded[: ds.Columns] == redecoded

    def test_cycle(self):
        """Test the decoded data remains the same after encoding/decoding."""
        ds = INDEX_RLE["OBXXXX1A_rle.dcm"]["ds"]
        pixel_data = b"".join(generate_frames(ds.PixelData))
        decoded = decode_segment(pixel_data[64:])
        assert ds.Rows * ds.Columns == len(decoded)
        arr = np.frombuffer(decoded, "uint8").reshape(ds.Rows, ds.Columns)
        # Re-encode the decoded data
        encoded = encode_segment(arr.tobytes(), ds.Columns)

        # Decode the re-encoded data and check that it's the same
        redecoded = decode_segment(encoded)
        assert ds.Rows * ds.Columns == len(redecoded)
        assert decoded == redecoded


RLE_MATCHING_DATASETS = [
    # (compressed, reference, nr frames)
    pytest.param(
        INDEX_RLE["OBXXXX1A_rle.dcm"]["ds"], INDEX_LEE["OBXXXX1A.dcm"]["ds"], 1
    ),
    pytest.param(
        INDEX_RLE["OBXXXX1A_rle_2frame.dcm"]["ds"],
        INDEX_LEE["OBXXXX1A_2frame.dcm"]["ds"],
        2,
    ),
    pytest.param(INDEX_RLE["SC_rgb_rle.dcm"]["ds"], INDEX_LEE["SC_rgb.dcm"]["ds"], 1),
    pytest.param(
        INDEX_RLE["SC_rgb_rle_2frame.dcm"]["ds"],
        INDEX_LEE["SC_rgb_2frame.dcm"]["ds"],
        2,
    ),
    pytest.param(
        INDEX_RLE["MR_small_RLE.dcm"]["ds"], INDEX_LEE["MR_small.dcm"]["ds"], 1
    ),
    pytest.param(
        INDEX_RLE["emri_small_RLE.dcm"]["ds"], INDEX_LEE["emri_small.dcm"]["ds"], 10
    ),
    pytest.param(
        INDEX_RLE["SC_rgb_rle_16bit.dcm"]["ds"], INDEX_LEE["SC_rgb_16bit.dcm"]["ds"], 1
    ),
    pytest.param(
        INDEX_RLE["SC_rgb_rle_16bit_2frame.dcm"]["ds"],
        INDEX_LEE["SC_rgb_16bit_2frame.dcm"]["ds"],
        2,
    ),
    pytest.param(
        INDEX_RLE["rtdose_rle_1frame.dcm"]["ds"],
        INDEX_LEI["rtdose_1frame.dcm"]["ds"],
        1,
    ),
    pytest.param(INDEX_RLE["rtdose_rle.dcm"]["ds"], INDEX_LEI["rtdose.dcm"]["ds"], 15),
    pytest.param(
        INDEX_RLE["SC_rgb_rle_32bit.dcm"]["ds"], INDEX_LEE["SC_rgb_32bit.dcm"]["ds"], 1
    ),
    pytest.param(
        INDEX_RLE["SC_rgb_rle_32bit_2frame.dcm"]["ds"],
        INDEX_LEE["SC_rgb_32bit_2frame.dcm"]["ds"],
        2,
    ),
]


class TestEncodeFrame:
    """Tests for _rle.encode_frame."""

    def setup_method(self):
        """Setup the tests."""
        # Create a dataset skeleton for use in the cycle tests
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        ds.Rows = 2
        ds.Columns = 4
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 1
        self.ds = ds

    @pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
    @pytest.mark.parametrize("_, ds, nr_frames", RLE_MATCHING_DATASETS)
    def test_cycle(self, _, ds, nr_frames):
        """Encode pixel data, then decode it and compare against original"""
        if nr_frames > 1:
            return

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        encoded = encode_frame(ds.PixelData, *params, "<")
        decoded = _rle_decode_frame(encoded, *params)

        arr = np.frombuffer(decoded, pixel_dtype(ds))

        if ds.SamplesPerPixel == 1:
            arr = arr.reshape(ds.Rows, ds.Columns)
        else:
            # RLE is planar configuration 1
            arr = np.reshape(arr, (ds.SamplesPerPixel, ds.Rows, ds.Columns))
            arr = arr.transpose(1, 2, 0)

        assert np.array_equal(ds.pixel_array, arr)

    def test_16_segments_raises(self):
        """Test that trying to encode 16-segments raises exception."""
        arr = np.asarray([[[1, 2, 3]]], dtype="uint64")
        assert (1, 1, 3) == arr.shape
        assert 8 == arr.dtype.itemsize

        msg = (
            r"Unable to encode as the DICOM Standard only allows "
            r"a maximum of 15 segments in RLE encoded data"
        )
        with pytest.raises(ValueError, match=msg):
            encode_frame(arr.tobytes(), 1, 1, 3, 64, "<")

    def test_12_segment(self):
        """Test encoding 12-segments works as expected."""
        # Most that can be done w/ -rle
        arr = np.asarray([[[1, 2, 3]]], dtype="uint32")
        # 00 00 00 01 | 00 00 00 02 | 00 00 00 03
        assert (1, 1, 3) == arr.shape
        assert 4 == arr.dtype.itemsize

        encoded = encode_frame(arr.tobytes(), 1, 1, 3, 32, "<")
        header = (
            b"\x0c\x00\x00\x00"
            b"\x40\x00\x00\x00"
            b"\x42\x00\x00\x00"
            b"\x44\x00\x00\x00"
            b"\x46\x00\x00\x00"
            b"\x48\x00\x00\x00"
            b"\x4a\x00\x00\x00"
            b"\x4c\x00\x00\x00"
            b"\x4e\x00\x00\x00"
            b"\x50\x00\x00\x00"
            b"\x52\x00\x00\x00"
            b"\x54\x00\x00\x00"
            b"\x56\x00\x00\x00"
            b"\x00\x00\x00\x00"
            b"\x00\x00\x00\x00"
            b"\x00\x00\x00\x00"
        )
        assert header == encoded[:64]
        assert (
            b"\x00\x00\x00\x00\x00\x00\x00\x01"
            b"\x00\x00\x00\x00\x00\x00\x00\x02"
            b"\x00\x00\x00\x00\x00\x00\x00\x03"
        ) == encoded[64:]

    @pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
    def test_encoding_multiple_frames_raises(self):
        """Test encoding multiple framed pixel data raises exception."""
        # Note: only works with multi-sample data
        ds = INDEX_LEE["SC_rgb_2frame.dcm"]["ds"]
        arr = ds.pixel_array
        assert ds.NumberOfFrames > 1
        assert len(arr.shape) == 4
        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)

        msg = (
            r"The length of the data to be encoded is not consistent with the "
            r"the values of the dataset's 'Rows', 'Columns', 'Samples per "
            r"Pixel' and 'Bits Allocated' elements"
        )
        with pytest.raises(ValueError, match=msg):
            encode_frame(arr.tobytes(), *params, "<")

    def test_single_row_1sample(self):
        """Test encoding a single row of 1 sample/pixel data."""
        # Rows 1, Columns 5, SamplesPerPixel 1
        arr = np.asarray([[0, 1, 2, 3, 4]], dtype="uint8")
        assert (1, 5) == arr.shape
        encoded = encode_frame(arr.tobytes(), 1, 5, 1, 8, "<")
        header = b"\x01\x00\x00\x00\x40\x00\x00\x00" + b"\x00" * 56
        assert header == encoded[:64]
        assert b"\x04\x00\x01\x02\x03\x04" == encoded[64:]

    def test_single_row_3sample(self):
        """Test encoding a single row of 3 samples/pixel data."""
        # Rows 1, Columns 5, SamplesPerPixel 3
        arr = np.asarray(
            [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]], dtype="uint8"
        )
        assert (1, 5, 3) == arr.shape
        encoded = encode_frame(arr.tobytes(), 1, 5, 3, 8, "<")
        header = (
            b"\x03\x00\x00\x00"
            b"\x40\x00\x00\x00"
            b"\x46\x00\x00\x00"
            b"\x4c\x00\x00\x00"
        )
        header += b"\x00" * (64 - len(header))
        assert header == encoded[:64]
        assert (
            b"\x04\x00\x01\x02\x03\x04"
            b"\x04\x00\x01\x02\x03\x04"
            b"\x04\x00\x01\x02\x03\x04"
        ) == encoded[64:]

    def test_padding(self):
        """Test that odd length encoded segments are padded."""
        data = b"\x00\x04\x01\x15"
        out = encode_frame(data, 1, 4, 1, 8, "<")

        # The segment should start with a literal run marker of 0x03
        #   then 4 bytes of RLE encoded data, then 0x00 padding
        assert b"\x03\x00\x04\x01\x15\x00" == out[64:]

    def test_16bit_segment_order(self):
        """Test that the segment order per 16-bit sample is correct."""
        # Native byte ordering
        data = b"\x00\x00\x01\xFF\xFE\x00\xFF\xFF\x10\x12"

        # Test little endian input
        out = encode_frame(data, 1, 5, 1, 16, "<")
        assert b"\x04\x00\xFF\x00\xFF\x12" == out[64:70]
        assert b"\x04\x00\x01\xFE\xFF\x10" == out[70:76]

        # Test big endian input
        out = encode_frame(data, 1, 5, 1, 16, ">")
        assert b"\x04\x00\x01\xFE\xFF\x10" == out[64:70]
        assert b"\x04\x00\xFF\x00\xFF\x12" == out[70:76]

    def test_32bit_segment_order(self):
        """Test that the segment order per 32-bit sample is correct."""
        data = b"\x00\x00\x00\x00\x01\xFF\xFE\x0A\xFF\xFC\x10\x12"

        # Test little endian input
        out = encode_frame(data, 1, 3, 1, 32, "<")
        assert b"\x02\x00\x0A\x12" == out[64:68]
        assert b"\x02\x00\xFE\x10" == out[68:72]
        assert b"\x02\x00\xFF\xFC" == out[72:76]
        assert b"\x02\x00\x01\xFF" == out[76:80]

        # Test big endian input
        out = encode_frame(data, 1, 3, 1, 32, ">")
        assert b"\x02\x00\x01\xFF" == out[64:68]
        assert b"\x02\x00\xFF\xFC" == out[68:72]
        assert b"\x02\x00\xFE\x10" == out[72:76]
        assert b"\x02\x00\x0A\x12" == out[76:80]

    def test_invalid_samples_per_pixel_raises(self):
        """Test exception raised if samples per pixel not valid."""
        msg = r"The \(0028,0002\) 'Samples per Pixel' must be 1 or 3"
        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 0, 1, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 2, 1, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 4, 1, "<")

    def test_invalid_bits_per_pixel_raises(self):
        """Test exception raised if bits per pixel not valid."""
        msg = r"The \(0028,0100\) 'Bits Allocated' value must be 8, 16, 32 or 64"
        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 0, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 1, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 7, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 9, "<")

        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 65, "<")

    def test_invalid_byteorder_raises(self):
        """Test exception raised if byteorder not valid."""
        msg = r"'byteorder' must be '>' or '<'"
        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 16, "=")

    def test_invalid_parameters_raises(self):
        """Test exception raised if parameters not valid."""
        msg = (
            r"The length of the data to be encoded is not consistent with the "
            r"the values of the dataset's 'Rows', 'Columns', 'Samples per "
            r"Pixel' and 'Bits Allocated' elements"
        )
        with pytest.raises(ValueError, match=msg):
            encode_frame(b"", 1, 1, 1, 16, "<")

    def test_invalid_nr_segments_raises(self):
        """Test exception raised if too many segments required."""
        msg = (
            r"Unable to encode as the DICOM Standard only allows "
            r"a maximum of 15 segments in RLE encoded data"
        )
        with pytest.raises(ValueError, match=msg):
            encode_frame(b"\x00" * 24, 1, 1, 3, 64, "<")
