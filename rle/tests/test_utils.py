"""Tests for the utils module."""

from struct import pack, unpack

import numpy as np
import pytest

try:
    from pydicom import dcmread
    from pydicom.pixels.utils import pixel_dtype
    from pydicom.uid import RLELossless

    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False
try:
    from pydicom.pixels.decoders.native import _rle_decode_frame
except ImportError:
    from pydicom.pixels.decoders.rle import _rle_decode_frame

from rle.data import get_indexed_datasets
from rle.utils import (
    decode_pixel_data,
    encode_pixel_data,
    encode_array,
    pixel_data,
    pixel_array,
)
from rle.rle import pack_bits, unpack_bits


INDEX_LEE = get_indexed_datasets("1.2.840.10008.1.2.1")


class TestDecodePixelData:
    """Tests for decode_pixel_data()"""

    def test_u8_1s_ba1(self):
        """Tests bits allocated 1"""
        header = b"\x01\x00\x00\x00\x40\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        # 0 0 0 0 0 1 0 1 0 1 1 0 1 1 1 1
        data = b"\xFC\x00\x07\x01\x00\x01\x00\x01\x01\x00\x01\xFD\x01\x00"
        src = header + data
        opts = {
            "rows": 1,
            "columns": 16,
            "bits_allocated": 1,
        }

        frame = decode_pixel_data(src, version=2, **opts)
        assert frame == (
            b"\x00\x00\x00\x00\x00\x01\x00\x01\x00\x01\x01\x00\x01\x01\x01\x01"
        )
        opts["pack_bits"] = True
        frame = decode_pixel_data(src, version=2, **opts)
        assert frame == b"\xA0\xF6"


@pytest.mark.skipif(not HAVE_PYDICOM, reason="no pydicom")
class TestEncodeArray:
    """Tests for utils.encode_array()."""

    def into_array(self, out, ds):
        arr = np.frombuffer(out, pixel_dtype(ds))

        if ds.SamplesPerPixel == 1:
            arr = arr.reshape(ds.Rows, ds.Columns)
        else:
            # RLE is planar configuration 1
            arr = np.reshape(arr, (ds.SamplesPerPixel, ds.Rows, ds.Columns))
            arr = arr.transpose(1, 2, 0)

        return arr

    def test_u8_1s_1f(self):
        """Test encoding 8-bit, 1 sample/px, 1 frame."""
        ds = INDEX_LEE["OBXXXX1A.dcm"]["ds"]
        ref = ds.pixel_array
        gen = encode_array(ref, ds)
        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref, arr)

        with pytest.raises(StopIteration):
            next(gen)

    def test_u8_1s_1f_by_kwargs(self):
        """Test encoding 8-bit, 1 sample/px, 1 frame by passing kwargs."""
        ds = INDEX_LEE["OBXXXX1A.dcm"]["ds"]

        kwargs = {}
        kwargs["rows"] = ds.Rows
        kwargs["columns"] = ds.Columns
        kwargs["samples_per_pixel"] = ds.SamplesPerPixel
        kwargs["bits_allocated"] = ds.BitsAllocated
        kwargs["number_of_frames"] = int(getattr(ds, "NumberOfFrames", 1))

        ref = ds.pixel_array
        gen = encode_array(ref, **kwargs)
        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref, arr)

        with pytest.raises(StopIteration):
            next(gen)

    def test_u32_3s_2f(self):
        """Test encoding LE-ordered 32-bit, 3 sample/px, 2 frame."""
        ds = INDEX_LEE["SC_rgb_32bit_2frame.dcm"]["ds"]

        ref = ds.pixel_array
        gen = encode_array(ref, ds)
        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref[0], arr)

        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref[1], arr)

        with pytest.raises(StopIteration):
            next(gen)

    def test_u32_3s_2f_by_kwargs(self):
        """Test encoding 32-bit, 3 sample/px, 2 frame by passing kwargs."""
        ds = INDEX_LEE["SC_rgb_32bit_2frame.dcm"]["ds"]

        kwargs = {}
        kwargs["rows"] = ds.Rows
        kwargs["columns"] = ds.Columns
        kwargs["samples_per_pixel"] = ds.SamplesPerPixel
        kwargs["bits_allocated"] = ds.BitsAllocated
        kwargs["number_of_frames"] = int(getattr(ds, "NumberOfFrames", 1))

        ref = ds.pixel_array
        gen = encode_array(ref, **kwargs)
        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref[0], arr)

        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref[1], arr)

        with pytest.raises(StopIteration):
            next(gen)


@pytest.mark.skipif(not HAVE_PYDICOM, reason="no pydicom")
class TestEncodePixelData:
    """Tests for utils.encode_pixel_data()."""

    def test_bad_byteorder_raises(self):
        """Test exception raised if invalid byteorder."""
        kwargs = {
            "rows": 0,
            "columns": 0,
            "samples_per_pixel": 1,
            "bits_allocated": 16,
            "byteorder": "=",
        }

        msg = (
            r"A valid 'byteorder' is required when the number of bits per "
            r"pixel is greater than 8"
        )
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

        kwargs["byteorder"] = None
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

    def test_encode_using_dataset(self):
        """Test encoding using a dataset"""
        ds = INDEX_LEE["SC_rgb_32bit_2frame.dcm"]["ds"]
        src = ds.pixel_array[0].tobytes()
        enc = encode_pixel_data(src, ds, "<")
        assert enc[:10] == b"\x0C\x00\x00\x00\x40\x00\x00\x00\x08\x01"

    def test_no_byteorder_u8(self):
        """Test exception raised if invalid byteorder."""
        kwargs = {
            "rows": 1,
            "columns": 1,
            "samples_per_pixel": 1,
            "bits_allocated": 8,
            "byteorder": None,
        }

        assert b"\x00\x01" == encode_pixel_data(b"\x01", **kwargs)[64:]

    def test_bad_samples_raises(self):
        """Test exception raised if invalid samples per pixel."""
        kwargs = {
            "rows": 0,
            "columns": 0,
            "samples_per_pixel": 0,
            "bits_allocated": 0,
            "byteorder": "<",
        }

        msg = r"'samples_per_pixel' must be 1 or 3"
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

    def test_bad_bits_allocated_raises(self):
        """Test exception raised if invalid bits allocated."""
        kwargs = {
            "rows": 0,
            "columns": 0,
            "samples_per_pixel": 1,
            "bits_allocated": 2,
            "byteorder": "<",
        }

        msg = r"'bits_allocated' must be 1, 8, 16, 32 or 64"
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

    def test_bad_length_raises(self):
        """Test exception raised if invalid parameter values."""
        kwargs = {
            "rows": 1,
            "columns": 1,
            "samples_per_pixel": 1,
            "bits_allocated": 8,
            "byteorder": "<",
        }

        msg = r"The length of the data doesn't match the image parameters"
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

    def test_too_many_segments_raises(self):
        """Test exception raised if too many segments."""
        kwargs = {
            "rows": 1,
            "columns": 1,
            "samples_per_pixel": 3,
            "bits_allocated": 64,
            "byteorder": "<",
        }

        msg = (
            r"Unable to encode the data as the RLE format used by the DICOM "
            r"Standard only allows a maximum of 15 segments"
        )
        with pytest.raises(ValueError, match=msg):
            encode_pixel_data(b"", **kwargs)

    def test_u8_1s_ba1(self):
        """Tests bits allocated 1"""
        opts = {
            "rows": 1,
            "columns": 16,
            "bits_allocated": 1,
            "samples_per_pixel": 1,
        }
        # 0 0 0 0 0 1 0 1 0 1 1 0 1 1 1 1
        enc = encode_pixel_data(b"\xA0\xF6", **opts)

        header = b"\x01\x00\x00\x00\x40\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        data = b"\xFC\x00\x03\x01\x00\x01\x00\xFF\x01\x00\x00\xFD\x01\x00"
        assert enc == header + data
        assert decode_pixel_data(enc, version=2, **opts) == (
            b"\x00\x00\x00\x00\x00\x01\x00\x01\x00\x01\x01\x00\x01\x01\x01\x01"
        )

        opts = {
            "rows": 1,
            "columns": 12,
            "bits_allocated": 1,
            "samples_per_pixel": 1,
        }
        # 0 0 0 0 0 1 0 1 0 1 1 0
        enc = encode_pixel_data(b"\xA0\xF6", **opts)

        header = b"\x01\x00\x00\x00\x40\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        data = b"\xFC\x00\x03\x01\x00\x01\x00\xFF\x01\x00\x00\x00"
        assert enc == header + data
        assert decode_pixel_data(enc, version=2, **opts) == (
            b"\x00\x00\x00\x00\x00\x01\x00\x01\x00\x01\x01\x00"
        )


@pytest.mark.skipif(not HAVE_PYDICOM, reason="no pydicom")
class TestPixelData:
    """Tests for utils.pixel_data()."""

    def test_pixel_data(self):
        """Test that data is encoded and encapsulated."""
        ds = INDEX_LEE["SC_rgb_32bit_2frame.dcm"]["ds"]
        ref = ds.pixel_array

        data = pixel_data(ref, ds)
        ds.file_meta.TransferSyntaxUID = RLELossless
        ds.PixelData = data

        assert np.array_equal(ref, pixel_array(ds))


REFERENCE_PACK_UNPACK = [
    # src, little, big
    (b"", [], []),
    (b"\x00", [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]),
    (b"\x01", [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x02", [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
    (b"\x04", [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
    (b"\x08", [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]),
    (b"\x10", [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
    (b"\x20", [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]),
    (b"\x40", [0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
    (b"\x80", [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]),
    (b"\xaa", [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]),
    (b"\xf0", [0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]),
    (b"\x0f", [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]),
    (b"\xff", [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    (
        b"\x00\x00",
        #| 1st byte              | 2nd byte
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ),
    (
        b"\x00\x01",
        #| 1st byte              | 2nd byte
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ),
    (
        b"\x00\x80",
        #| 1st byte              | 2nd byte
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
    (
        b"\x00\xff",
        #| 1st byte              | 2nd byte
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
    (
        b"\x01\x80",
        #| 1st byte              | 2nd byte
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
    (
        b"\x80\x80",
        #| 1st byte              | 2nd byte
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
    (
        b"\xff\x80",
        #| 1st byte              | 2nd byte
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
]


class TestUnpackBits:
    """Tests for unpack_bits()."""

    @pytest.mark.parametrize("src, little, big", REFERENCE_PACK_UNPACK)
    def test_unpack_bytes(self, src, little, big):
        """Test unpacking data without numpy."""
        as_bytes = pack(f"{len(little)}B", *little)
        assert unpack_bits(src, 0, "<") == as_bytes
        assert unpack_bits(src, 32, "<") == as_bytes
        as_bytes = pack(f"{len(big)}B", *big)
        assert unpack_bits(src, 0, ">") == as_bytes
        assert unpack_bits(src, 32, ">") == as_bytes

    def test_count_little(self):
        """Test the `count` parameter for little endian unpacking."""
        assert unpack_bits(b"\x00", 1, "<") == b"\x00"
        assert unpack_bits(b"\xff", 1, "<") == b"\x01"
        assert unpack_bits(b"\xff", 2, "<") == b"\x01" * 2
        assert unpack_bits(b"\xff", 3, "<") == b"\x01" * 3
        assert unpack_bits(b"\xff", 4, "<") == b"\x01" * 4
        assert unpack_bits(b"\xff", 5, "<") == b"\x01" * 5
        assert unpack_bits(b"\xff", 6, "<") == b"\x01" * 6
        assert unpack_bits(b"\xff", 7, "<") == b"\x01" * 7
        assert unpack_bits(b"\xff", 8, "<") == b"\x01" * 8
        assert unpack_bits(b"\xff\xAA", 9, "<") == b"\x01" * 8 + b"\x00"
        assert unpack_bits(b"\xff\xAA", 10, "<") == b"\x01" * 8 + b"\x00\x01"
        assert unpack_bits(b"\xff\xAA", 11, "<") == b"\x01" * 8 + b"\x00\x01\x00"
        assert unpack_bits(b"\xff\xAA", 12, "<") == b"\x01" * 8 + b"\x00\x01" * 2
        assert unpack_bits(b"\xff\xAA", 13, "<") == b"\x01" * 8 + b"\x00\x01" * 2 + b"\x00"
        assert unpack_bits(b"\xff\xAA", 14, "<") == b"\x01" * 8 + b"\x00\x01" * 3
        assert unpack_bits(b"\xff\xAA", 15, "<") == b"\x01" * 8 + b"\x00\x01" * 3 + b"\x00"
        assert unpack_bits(b"\xff\xAA", 16, "<") == b"\x01" * 8 + b"\x00\x01" * 4

    def test_count_big(self):
        """Test the `count` parameter for big endian unpacking."""
        assert unpack_bits(b"\x00", 1, ">") == b"\x00"
        assert unpack_bits(b"\xff", 1, ">") == b"\x01"
        assert unpack_bits(b"\xff", 2, ">") == b"\x01" * 2
        assert unpack_bits(b"\xff", 3, ">") == b"\x01" * 3
        assert unpack_bits(b"\xff", 4, ">") == b"\x01" * 4
        assert unpack_bits(b"\xff", 5, ">") == b"\x01" * 5
        assert unpack_bits(b"\xff", 6, ">") == b"\x01" * 6
        assert unpack_bits(b"\xff", 7, ">") == b"\x01" * 7
        assert unpack_bits(b"\xff", 8, ">") == b"\x01" * 8
        assert unpack_bits(b"\xff\xAA", 9, ">") == b"\x01" * 8 + b"\x01"
        assert unpack_bits(b"\xff\xAA", 10, ">") == b"\x01" * 8 + b"\x01\x00"
        assert unpack_bits(b"\xff\xAA", 11, ">") == b"\x01" * 8 + b"\x01\x00\x01"
        assert unpack_bits(b"\xff\xAA", 12, ">") == b"\x01" * 8 + b"\x01\x00" * 2
        assert unpack_bits(b"\xff\xAA", 13, ">") == b"\x01" * 8 + b"\x01\x00" * 2 + b"\x01"
        assert unpack_bits(b"\xff\xAA", 14, ">") == b"\x01" * 8 + b"\x01\x00" * 3
        assert unpack_bits(b"\xff\xAA", 15, ">") == b"\x01" * 8 + b"\x01\x00" * 3 + b"\x01"
        assert unpack_bits(b"\xff\xAA", 16, ">") == b"\x01" * 8 + b"\x01\x00" * 4

    @pytest.mark.parametrize("src, little, big", REFERENCE_PACK_UNPACK)
    def test_unpack_bytearray(self, src, little, big):
        """Test unpacking data without numpy."""
        as_bytes = pack(f"{len(little)}B", *little)
        assert unpack_bits(bytearray(src), 0, "<") == as_bytes
        as_bytes = pack(f"{len(big)}B", *big)
        assert unpack_bits(bytearray(src), 0, ">") == as_bytes


REFERENCE_PACK_PARTIAL_LITTLE = [
    #              | 1st byte              | 2nd byte
    (b"\x00\x40", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 15-bits
    (b"\x00\x20", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x10", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x08", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x04", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x02", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x01", [0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 9-bits
    (b"\x80", [0, 0, 0, 0, 0, 0, 0, 1]),  # 8-bits
    (b"\x40", [0, 0, 0, 0, 0, 0, 1]),
    (b"\x20", [0, 0, 0, 0, 0, 1]),
    (b"\x10", [0, 0, 0, 0, 1]),
    (b"\x08", [0, 0, 0, 1]),
    (b"\x04", [0, 0, 1]),
    (b"\x02", [0, 1]),
    (b"\x01", [1]),
    (b"", []),
]
REFERENCE_PACK_PARTIAL_BIG = [
    #              | 1st byte              | 2nd byte
    (b"\x00\x02", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 15-bits
    (b"\x00\x04", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x08", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x10", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x20", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x40", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    (b"\x00\x80", [0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 9-bits
    (b"\x01", [0, 0, 0, 0, 0, 0, 0, 1]),  # 8-bits
    (b"\x02", [0, 0, 0, 0, 0, 0, 1]),
    (b"\x04", [0, 0, 0, 0, 0, 1]),
    (b"\x08", [0, 0, 0, 0, 1]),
    (b"\x10", [0, 0, 0, 1]),
    (b"\x20", [0, 0, 1]),
    (b"\x40", [0, 1]),
    (b"\x80", [1]),
    (b"", []),
]


class TestPackBits:
    """Tests for pack_bits()."""

    @pytest.mark.parametrize("output, little, big", REFERENCE_PACK_UNPACK)
    def test_pack_bytes(self, output, little, big):
        """Test packing data."""
        assert output == pack_bits(bytes(little), "<")
        assert output == pack_bits(bytes(big), ">")

    @pytest.mark.parametrize("output, little, big", REFERENCE_PACK_UNPACK)
    def test_pack_bytearray(self, output, little, big):
        """Test packing data."""
        assert output == pack_bits(bytearray(little), "<")
        assert output == pack_bits(bytearray(big), ">")

    def test_non_binary_input(self):
        """Test non-binary input raises exception."""
        msg = r"Only binary input \(containing zeros or ones\) can be packed"
        with pytest.raises(ValueError, match=msg):
            pack_bits(b"\x00\x00\x02\x00\x00\x00\x00\x00", "<")

    def test_bytes_input(self):
        """Repeat above test with bytes input."""
        # fmt: off
        src = bytes(
            [
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
            ]
        )
        # fmt: on
        assert b"\x00\x55\xff" == pack_bits(src, "<")
        assert b"\x00\xAA\xff" == pack_bits(src, ">")

    def test_bytearry_input(self):
        """Repeat above test with bytearray input."""
        # fmt: off
        src = bytearray(
            [
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
            ]
        )
        # fmt: on
        assert b"\x00\x55\xff" == pack_bits(src, "<")
        assert b"\x00\xAA\xff" == pack_bits(src, ">")

    @pytest.mark.parametrize("output, src", REFERENCE_PACK_PARTIAL_LITTLE)
    def test_pack_partial_bytes(self, src, output):
        """Test packing data that isn't a full byte long."""
        assert output == pack_bits(bytes(src), "<")

    @pytest.mark.parametrize("output, src", REFERENCE_PACK_PARTIAL_LITTLE)
    def test_pack_partial_bytearray(self, src, output):
        """Test packing data that isn't a full byte long."""
        assert output == pack_bits(bytearray(src), "<")

    @pytest.mark.parametrize("output, src", REFERENCE_PACK_PARTIAL_BIG)
    def test_pack_partial_bytes_big(self, src, output):
        """Test packing data that isn't a full byte long."""
        assert output == pack_bits(bytes(src), ">")

    @pytest.mark.parametrize("output, src", REFERENCE_PACK_PARTIAL_BIG)
    def test_pack_partial_bytearray_big(self, src, output):
        """Test packing data that isn't a full byte long."""
        assert output == pack_bits(bytearray(src), ">")
