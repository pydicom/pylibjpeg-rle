"""Tests for the utils module."""

import numpy as np
import pytest

try:
    from pydicom import dcmread
    from pydicom.pixel_data_handlers.rle_handler import _rle_decode_frame
    from pydicom.pixel_data_handlers.util import (
        pixel_dtype, reshape_pixel_array
    )
    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False

from rle.data import get_indexed_datasets
from rle.utils import encode_pixel_data, encode_array


INDEX_LEE = get_indexed_datasets('1.2.840.10008.1.2.1')


class TestEncodeArray:
    """Tests for utils.encode_array()."""
    def into_array(self, out, ds):
        dtype = pixel_dtype(ds).newbyteorder('>')
        arr = np.frombuffer(out, dtype)

        if ds.SamplesPerPixel == 1:
            arr = arr.reshape(ds.Rows, ds.Columns)
        else:
            # RLE is planar configuration 1
            arr = np.reshape(arr, (ds.SamplesPerPixel, ds.Rows, ds.Columns))
            arr = arr.transpose(1, 2, 0)

        return arr

    def test_u8_1s_1f(self):
        """Test encoding 8-bit, 1 sample/px, 1 frame."""
        ds = INDEX_LEE["OBXXXX1A.dcm"]['ds']
        ref = ds.pixel_array
        gen = encode_array(ref, **{'ds': ds})
        b = next(gen)

        params = (ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.BitsAllocated)
        arr = self.into_array(_rle_decode_frame(b, *params), ds)

        assert np.array_equal(ref, arr)

        with pytest.raises(StopIteration):
            next(gen)

    def test_u32_3s_2f(self):
        """Test encoding 32-bit, 3 sample/px, 2 frame."""
        ds = INDEX_LEE["SC_rgb_32bit_2frame.dcm"]['ds']

        ref = ds.pixel_array
        gen = encode_array(ref, **{'ds': ds})
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

    def test_byteorder(self):
        pass

    def test_missing_params(self):
        pass


class TestEncodePixelData:
    """Tests for utils.encode_pixel_data()."""
    def test_missing_params(self):
        pass

    def test_byteorder(self):
        pass

    def test_bad_samples_raises(self):
        pass

    def test_bad_bits_allocated_raises(self):
        pass

    def test_bad_length_raises(self):
        pass

    def test_too_many_segments_raises(self):
        pass
