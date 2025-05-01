"""Tests for the pylibjpeg pixel data handler."""

import pytest
import numpy as np

try:
    from pydicom import dcmread
    from pydicom.encaps import generate_fragmented_frames
    from pydicom.uid import RLELossless

    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False

from rle.data import get_indexed_datasets
from rle.utils import decode_pixel_data

INDEX = get_indexed_datasets("1.2.840.10008.1.2.5")


@pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
class TestDecodePixelData:
    """Tests for the plugin's decoder interface."""

    def test_no_dataset_kwargs_raises(self):
        """Test plugin decoder with no dataset or kwargs raises"""
        ds = INDEX["OBXXXX1A_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 600 == ds.Rows
        assert 800 == ds.Columns
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        frame = next(generate_fragmented_frames(ds.PixelData))
        msg = r"Either `ds` or `\*\*kwargs` must be used"
        with pytest.raises(ValueError, match=msg):
            decode_pixel_data(frame)

    def test_u8_1s_1f(self):
        """Test plugin decoder for 8 bit, 1 sample, 1 frame data."""
        ds = INDEX["OBXXXX1A_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 600 == ds.Rows
        assert 800 == ds.Columns
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        frame = b"".join(next(generate_fragmented_frames(ds.PixelData)))
        arr = decode_pixel_data(frame, ds)
        assert (480000,) == arr.shape
        assert arr.flags.writeable
        assert "uint8" == arr.dtype

    def test_u32_3s_1f(self):
        """Test plugin decoder for 32 bit, 3 sample, 1 frame data."""
        ds = INDEX["SC_rgb_rle_32bit.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 32 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 100 == ds.Rows
        assert 100 == ds.Columns
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        frame = b"".join(next(generate_fragmented_frames(ds.PixelData)))
        arr = decode_pixel_data(frame, ds)
        assert (120000,) == arr.shape
        assert arr.flags.writeable
        assert "uint8" == arr.dtype

    def test_v2(self):
        """Test v2 plugin decoder for 8 bit, 1 sample, 1 frame data."""
        ds = INDEX["OBXXXX1A_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        opts = {
            "bits_allocated": ds.BitsAllocated,
            "rows": ds.Rows,
            "columns": ds.Columns,
        }
        frame = b"".join(next(generate_fragmented_frames(ds.PixelData)))
        buffer = decode_pixel_data(frame, version=2, **opts)
        assert isinstance(buffer, bytearray)
        arr = np.frombuffer(buffer, dtype="u1")
        assert (480000,) == arr.shape
        assert arr.flags.writeable

    def test_v2_missing_kwarg_raises(self):
        """Test v2 raises exception if required argument missing"""
        ds = INDEX["OBXXXX1A_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        opts = {
            "bits_allocated": ds.BitsAllocated,
            "rows": ds.Rows,
            # "columns": ds.Columns,
        }
        frame = next(generate_fragmented_frames(ds.PixelData))
        msg = "Missing expected keyword arguments: columns"
        with pytest.raises(AttributeError, match=msg):
            buffer = decode_pixel_data(frame, version=2, **opts)
