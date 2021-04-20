"""Tests for decoding RLE data."""

from struct import pack

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
from rle._rle import decode_segment, decode_frame, parse_header


INDEX = get_indexed_datasets('1.2.840.10008.1.2.5')

REF = [
    ('MR_small_RLE.dcm', 2, (64, 1948)),
]


HEADER_DATA = [
    # (Number of segments, offsets)
    (0, []),
    (1, [64]),
    (2, [64, 16]),
    (8, [64, 16, 31, 55, 62, 110, 142, 551]),
    (14, [64, 16, 31, 55, 62, 110, 142, 551, 641, 456, 43, 11, 6, 55]),
    (15, [64, 16, 31, 55, 62, 110, 142, 551, 641, 456, 43, 11, 6, 55, 9821]),
]


class TestParseHeader:
    """Tests for rle._rle.parse_header()."""
    def test_invalid_header_length(self):
        """Test exception raised if header is not 64 bytes long."""
        msg = r"The RLE header must be 64 bytes long"
        for length in [0, 1, 63, 65]:
            with pytest.raises(ValueError, match=msg):
                parse_header(b'\x00' * length)

    @pytest.mark.parametrize('nr_segments, offsets', HEADER_DATA)
    def test_parse_header(self, nr_segments, offsets):
        """Test parsing header data."""
        # Encode the header
        header = bytearray()
        header.extend(pack('<L', nr_segments))
        header.extend(pack('<{}L'.format(len(offsets)), *offsets))
        # Add padding
        header.extend(b'\x00' * (64 - len(header)))

        offsets += [0] * (15 - len(offsets))

        assert len(header) == 64
        assert offsets == parse_header(bytes(header))


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestDecodeFrame:
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


class TestDecodeSegment:
    """Tests for rle._rle.decode_segment().

    References
    ----------
    DICOM Standard, Part 5, Annex G.3.2
    """
    def test_noop(self):
        """Test no-operation output."""
        # For n == 128, do nothing
        # data is only noop, 0x80 = 128
        assert b'' == decode_segment(b'\x80\x80\x80')

        # noop at start, data after
        data = (
            b'\x80\x80'  # No operation
            b'\x05\x01\x02\x03\x04\x05\x06'  # Literal
            b'\xFE\x01'  # Copy
            b'\x80'
        )
        assert (
            b'\x01\x02\x03\x04\x05\x06'
            b'\x01\x01\x01'
        ) == decode_segment(data)

        # data at start, noop middle, data at end
        data = (
            b'\x05\x01\x02\x03\x04\x05\x06'  # Literal
            b'\x80'  # No operation
            b'\xFE\x01'  # Copy
            b'\x80'
        )
        assert (
            b'\x01\x02\x03\x04\x05\x06'
            b'\x01\x01\x01'
        ) == decode_segment(data)

        # data at start, noop end
        # Copy 6 bytes literally, then 3 x 0x01
        data = (
            b'\x05\x01\x02\x03\x04\x05\x06'
            b'\xFE\x01'
            b'\x80'
        )
        assert (
            b'\x01\x02\x03\x04\x05\x06'
            b'\x01\x01\x01'
        ) == decode_segment(data)

    def test_literal(self):
        """Test literal output."""
        # For n < 128, read the next (n + 1) bytes literally
        # n = 0 (0x80 is 128 -> no operation)
        assert b'\x02' == decode_segment(b'\x00\x02\x80')
        # n = 1
        assert b'\x02\x03' == decode_segment(b'\x01\x02\x03\x80')
        # n = 127
        data = b'\x7f' + b'\x40' * 128 + b'\x80'
        assert b'\x40' * 128 == decode_segment(data)

    def test_copy(self):
        """Test copy output."""
        # For n > 128, copy the next byte (257 - n) times
        # n = 255, copy x2 (0x80 is 128 -> no operation)
        assert b'\x02\x02' == decode_segment(b'\xFF\x02\x80')
        # n = 254, copy x3
        assert b'\x02\x02\x02' == decode_segment(b'\xFE\x02\x80')
        # n = 129, copy x128
        assert b'\x02' * 128 == decode_segment(b'\x81\x02\x80')

    def test_invalid_copy_raises(self):
        """Test an invalid repeat sequence raises an exceptions."""
        msg = (
            r"The end of the data was reached before the segment "
            r"was completely decoded"
        )
        with pytest.raises(ValueError, match=msg):
            decode_segment(b"\x02")

    def test_invalid_literal_raises(self):
        """Test an invalid literal sequence raises an exceptions."""
        msg = (
            r"The end of the data was reached before the segment "
            r"was completely decoded"
        )
        with pytest.raises(ValueError, match=msg):
            decode_segment(b'\x01\x02')
