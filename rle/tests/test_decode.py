"""Tests for decoding RLE data."""


import pytest

try:
    from pydicom import dcmread
    from pydicom.encaps import generate_pixel_data_frame
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from rle.data import get_indexed_datasets
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

        header = frame[:64]
        b = [f"0x{ii:02x}" for ii in header]
        print(" ".join(b))

        from librle import parse_header, decode_segment
        assert 64 == len(parse_header(frame))


        offsets = parse_header(header)
        print(offsets)
        result = decode_segment(frame[offsets[0]:offsets[1]])
        print(len(result))
