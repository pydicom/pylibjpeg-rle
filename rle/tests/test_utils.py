"""Tests for the utils module."""

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
from rle.utils import encode_pixel_data, encode_array, pixel_data, pixel_array


INDEX_LEE = get_indexed_datasets("1.2.840.10008.1.2.1")


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

        msg = r"'bits_allocated' must be 8, 16, 32 or 64"
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
