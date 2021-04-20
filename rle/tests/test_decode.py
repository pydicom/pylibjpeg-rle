"""Tests for decoding RLE data."""


import pytest

try:
    from pydicom import dcmread
    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.rle_handler import (
        _parse_rle_header, _rle_decode_frame, _rle_decode_segment
    )
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from rle.data import get_indexed_datasets
from rle._rle import decode_segment, decode_frame
#from rle.utils import parse_header


INDEX = get_indexed_datasets('1.2.840.10008.1.2.5')

REF = [
    ('MR_small_RLE.dcm', 2, (64, 1948)),
]


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestParseHeader:
    def test_one(self):
        # 2 segments, (64, 1948)
        ds = INDEX['MR_small_RLE.dcm']['ds']
        frame_gen = generate_pixel_data_frame(ds.PixelData)
        frame = next(frame_gen)

        offsets = _parse_rle_header(frame[:64])
        print(offsets)

        result = decode_segment(frame[offsets[0]:offsets[1]])
        print(result[:20], len(result))

        px_per_sample = ds.Rows * ds.Columns
        bits_per_px = ds.BitsAllocated
        print(ds.BitsAllocated)
        #print(type(frame))
        #frame = b'\x00'
        frame = decode_frame(frame, px_per_sample, bits_per_px)
        print(type(frame), len(frame))
