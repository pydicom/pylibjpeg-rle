"""Tests for decoding RLE data."""

from copy import deepcopy
from struct import pack

import numpy as np
import pytest

try:
    from pydicom import dcmread
    from pydicom.uid import RLELossless

    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False

from rle.data import get_indexed_datasets
from rle.rle import decode_segment, decode_frame, parse_header
from rle.utils import generate_frames, pixel_array


INDEX = get_indexed_datasets("1.2.840.10008.1.2.5")


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
                parse_header(b"\x00" * length)

    @pytest.mark.parametrize("nr_segments, offsets", HEADER_DATA)
    def test_parse_header(self, nr_segments, offsets):
        """Test parsing header data."""
        # Encode the header
        header = bytearray()
        header.extend(pack("<L", nr_segments))
        header.extend(pack(f"<{len(offsets)}L", *offsets))
        # Add padding
        header.extend(b"\x00" * (64 - len(header)))

        offsets += [0] * (15 - len(offsets))

        assert len(header) == 64
        assert offsets == parse_header(bytes(header))


class TestDecodeFrame:
    """Tests for rle._rle.decode_frame()."""

    def as_bytes(self, offsets):
        d = [len(offsets)] + offsets
        d += [0] * (16 - len(d))
        return pack("<16l", *d)

    def test_bits_allocated_zero_raises(self):
        """Test exception raised for BitsAllocated 0."""
        msg = r"The \(0028,0100\) 'Bits Allocated' value must be 8, 16, 32 or 64"
        with pytest.raises(ValueError, match=msg):
            decode_frame(b"\x00\x00\x00\x00", 1, 0, "<")

    def test_bits_allocated_not_octal_raises(self):
        """Test exception raised for BitsAllocated not a multiple of 8."""
        msg = r"The \(0028,0100\) 'Bits Allocated' value must be 8, 16, 32 or 64"
        with pytest.raises(ValueError, match=msg):
            decode_frame(b"\x00\x00\x00\x00", 1, 12, "<")

    def test_bits_allocated_large_raises(self):
        """Test exception raised for BitsAllocated greater than 64."""
        msg = r"The \(0028,0100\) 'Bits Allocated' value must be 8, 16, 32 or 64"
        with pytest.raises(ValueError, match=msg):
            decode_frame(b"\x00\x00\x00\x00", 1, 72, "<")

    def test_insufficient_data_for_header_raises(self):
        """Test exception raised if insufficient data."""
        msg = r"Frame is not long enough to contain RLE encoded data"
        with pytest.raises(ValueError, match=msg):
            decode_frame(b"\x00\x00\x00\x00", 1, 8, "<")

    def test_no_data_raises(self):
        """Test exception raised if no data."""
        msg = r"Frame is not long enough to contain RLE encoded data"
        with pytest.raises(ValueError, match=msg):
            decode_frame(b"", 1, 8, "<")

    def test_invalid_first_offset_raises(self):
        """Test exception if invalid first offset."""
        msg = r"Invalid segment offset found in the RLE header"
        d = self.as_bytes([0])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d, 1, 8, "<")

    def test_insufficient_data_for_offsets_raises(self):
        """Test exception if invalid first offset."""
        msg = r"Invalid segment offset found in the RLE header"
        # Offset 64 with length 64
        d = self.as_bytes([64])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d, 1, 8, "<")

    def test_non_increasing_offsets_raises(self):
        """Test exception if offsets not in increasing order."""
        msg = r"Invalid segment offset found in the RLE header"
        d = self.as_bytes([64, 70, 68])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d, 1, 8, "<")

    def test_invalid_samples_px_raises(self):
        """Test exception if samples per px not 1 or 3."""
        msg = r"The \(0028,0002\) 'Samples per Pixel' must be 1 or 3"
        d = self.as_bytes([64, 70])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d + b"\x00" * 8, 1, 8, "<")

    def test_insufficient_frame_literal(self):
        """Test segment with excess padding on lit."""
        d = self.as_bytes([64])
        assert decode_frame(d + b"\x00" * 8, 1, 8, "<") == b"\x00"

    def test_insufficient_frame_copy(self):
        """Test segment withe excess padding on copy."""
        d = self.as_bytes([64])
        assert decode_frame(d + b"\xff\x00\x00", 1, 8, "<") == b"\x00"

    def test_insufficient_segment_copy_raises(self):
        """Test exception if insufficient segment data on copy."""
        msg = (
            r"The end of the data was reached before the segment was "
            r"completely decoded"
        )
        d = self.as_bytes([64])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d + b"\xff", 8, 8, "<")

    def test_insufficient_segment_literal_raises(self):
        """Test exception if insufficient segment data on literal."""
        msg = (
            r"The end of the data was reached before the segment was "
            r"completely decoded"
        )
        d = self.as_bytes([64])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d + b"\x0a" * 8, 12, 8, "<")

    def test_invalid_byteorder_raises(self):
        """Test exception if invalid byteorder."""
        header = b"\x01\x00\x00\x00" b"\x40\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        # 0, 64, 128, 160, 192, 255
        data = b"\x05\x00\x40\x80\xA0\xC0\xFF"

        # Ok with u8
        decode_frame(header + data, 2 * 3, 8, "=")

        msg = r"'byteorder' must be '>' or '<'"
        with pytest.raises(ValueError, match=msg):
            decode_frame(header + data, 1 * 3, 16, "=")

    def test_decoded_segment_length_short(self):
        """Test exception if decoded segment length invalid."""
        msg = r"The decoded segment length does not match the expected length"
        d = self.as_bytes([64])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d + b"\x00" * 8, 12, 8, "<")

    def test_decoded_segment_length_long(self):
        """Test exception if decoded segment length invalid."""
        msg = r"The decoded segment length does not match the expected length"
        d = self.as_bytes([64, 72])
        with pytest.raises(ValueError, match=msg):
            decode_frame(d + b"\x00" * 20, 8, 16, "<")

    def test_u8_1s(self):
        """Test decoding 8-bit, 1 sample/pixel."""
        header = b"\x01\x00\x00\x00" b"\x40\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        # 0, 64, 128, 160, 192, 255
        data = b"\x05\x00\x40\x80\xA0\xC0\xFF"
        # Big endian
        decoded = decode_frame(header + data, 2 * 3, 8, ">")
        arr = np.frombuffer(decoded, np.dtype("uint8"))
        assert [0, 64, 128, 160, 192, 255] == arr.tolist()

        # Little-endian
        decoded = decode_frame(header + data, 2 * 3, 8, "<")
        arr = np.frombuffer(decoded, np.dtype("uint8"))
        assert [0, 64, 128, 160, 192, 255] == arr.tolist()

    def test_u8_3s(self):
        """Test decoding 8-bit, 3 sample/pixel."""
        header = (
            b"\x03\x00\x00\x00"  # 3 segments
            b"\x40\x00\x00\x00"  # 64
            b"\x47\x00\x00\x00"  # 71
            b"\x4E\x00\x00\x00"  # 78
        )
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        # 0, 64, 128, 160, 192, 255
        data = (
            b"\x05\x00\x40\x80\xA0\xC0\xFF"  # R
            b"\x05\xFF\xC0\x80\x40\x00\xFF"  # B
            b"\x05\x01\x40\x80\xA0\xC0\xFE"  # G
        )
        decoded = decode_frame(header + data, 2 * 3, 8, "<")
        arr = np.frombuffer(decoded, np.dtype("uint8"))
        # Ordered all R, all G, all B
        assert [0, 64, 128, 160, 192, 255] == arr[:6].tolist()
        assert [255, 192, 128, 64, 0, 255] == arr[6:12].tolist()
        assert [1, 64, 128, 160, 192, 254] == arr[12:].tolist()

    def test_u16_1s(self):
        """Test decoding 16-bit, 1 sample/pixel."""
        header = b"\x02\x00\x00\x00" b"\x40\x00\x00\x00" b"\x47\x00\x00\x00"
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        data = (
            # 0, 1, 256, 255, 65280, 65535
            b"\x05\x00\x00\x01\x00\xFF\xFF"  # MSB
            b"\x05\x00\x01\x00\xFF\x00\xFF"  # LSB
        )
        # Big-endian output
        decoded = decode_frame(header + data, 2 * 3, 16, ">")
        arr = np.frombuffer(decoded, np.dtype(">u2"))
        assert [0, 1, 256, 255, 65280, 65535] == arr.tolist()

        # Little-endian output
        decoded = decode_frame(header + data, 2 * 3, 16, "<")
        arr = np.frombuffer(decoded, np.dtype("<u2"))
        assert [0, 1, 256, 255, 65280, 65535] == arr.tolist()

    def test_u16_3s(self):
        """Test decoding 16-bit, 3 sample/pixel."""
        header = (
            b"\x06\x00\x00\x00"  # 6 segments
            b"\x40\x00\x00\x00"  # 64
            b"\x47\x00\x00\x00"  # 71
            b"\x4E\x00\x00\x00"  # 78
            b"\x55\x00\x00\x00"  # 85
            b"\x5C\x00\x00\x00"  # 92
            b"\x63\x00\x00\x00"  # 99
        )
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        data = (
            # 0, 1, 256, 255, 65280, 65535
            b"\x05\x00\x00\x01\x00\xFF\xFF"  # MSB
            b"\x05\x00\x01\x00\xFF\x00\xFF"  # LSB
            b"\x05\xFF\x00\x01\x00\xFF\x00"  # MSB
            b"\x05\xFF\x01\x00\xFF\x00\x00"  # LSB
            b"\x05\x00\x00\x01\x00\xFF\xFF"  # MSB
            b"\x05\x01\x01\x00\xFF\x00\xFE"  # LSB
        )
        # Big-endian output
        decoded = decode_frame(header + data, 2 * 3, 16, ">")
        arr = np.frombuffer(decoded, np.dtype(">u2"))
        assert [0, 1, 256, 255, 65280, 65535] == arr[:6].tolist()
        assert [65535, 1, 256, 255, 65280, 0] == arr[6:12].tolist()
        assert [1, 1, 256, 255, 65280, 65534] == arr[12:].tolist()

        # Little-endian output
        decoded = decode_frame(header + data, 2 * 3, 16, "<")
        arr = np.frombuffer(decoded, np.dtype("<u2"))
        assert [0, 1, 256, 255, 65280, 65535] == arr[:6].tolist()
        assert [65535, 1, 256, 255, 65280, 0] == arr[6:12].tolist()
        assert [1, 1, 256, 255, 65280, 65534] == arr[12:].tolist()

    def test_u32_1s(self):
        """Test decoding 32-bit, 1 sample/pixel."""
        header = (
            b"\x04\x00\x00\x00"  # 4 segments
            b"\x40\x00\x00\x00"  # 64 offset
            b"\x47\x00\x00\x00"  # 71 offset
            b"\x4E\x00\x00\x00"  # 78 offset
            b"\x55\x00\x00\x00"  # 85 offset
        )
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        data = (
            # 0, 16777216, 65536, 256, 4294967295
            b"\x05\x00\x01\x00\x00\x00\xFF"  # MSB
            b"\x05\x00\x00\x01\x00\x00\xFF"
            b"\x05\x00\x00\x00\x01\x00\xFF"
            b"\x05\x00\x00\x00\x00\x01\xFF"  # LSB
        )
        # Big-endian output
        decoded = decode_frame(header + data, 2 * 3, 32, ">")
        arr = np.frombuffer(decoded, np.dtype(">u4"))
        assert [0, 16777216, 65536, 256, 1, 4294967295] == arr.tolist()

        # Little-endian output
        decoded = decode_frame(header + data, 2 * 3, 32, "<")
        arr = np.frombuffer(decoded, np.dtype("<u4"))
        assert [0, 16777216, 65536, 256, 1, 4294967295] == arr.tolist()

    def test_u32_3s(self):
        """Test decoding 32-bit, 3 sample/pixel."""
        header = (
            b"\x0C\x00\x00\x00"  # 12 segments
            b"\x40\x00\x00\x00"  # 64
            b"\x47\x00\x00\x00"  # 71
            b"\x4E\x00\x00\x00"  # 78
            b"\x55\x00\x00\x00"  # 85
            b"\x5C\x00\x00\x00"  # 92
            b"\x63\x00\x00\x00"  # 99
            b"\x6A\x00\x00\x00"  # 106
            b"\x71\x00\x00\x00"  # 113
            b"\x78\x00\x00\x00"  # 120
            b"\x7F\x00\x00\x00"  # 127
            b"\x86\x00\x00\x00"  # 134
            b"\x8D\x00\x00\x00"  # 141
        )
        header += (64 - len(header)) * b"\x00"
        # 2 x 3 data
        data = (
            # 0, 16777216, 65536, 256, 4294967295
            b"\x05\x00\x01\x00\x00\x00\xFF"  # MSB
            b"\x05\x00\x00\x01\x00\x00\xFF"
            b"\x05\x00\x00\x00\x01\x00\xFF"
            b"\x05\x00\x00\x00\x00\x01\xFF"  # LSB
            b"\x05\xFF\x01\x00\x00\x00\x00"  # MSB
            b"\x05\xFF\x00\x01\x00\x00\x00"
            b"\x05\xFF\x00\x00\x01\x00\x00"
            b"\x05\xFF\x00\x00\x00\x01\x00"  # LSB
            b"\x05\x00\x01\x00\x00\x00\xFF"  # MSB
            b"\x05\x00\x00\x01\x00\x00\xFF"
            b"\x05\x00\x00\x00\x01\x00\xFF"
            b"\x05\x01\x00\x00\x00\x01\xFE"  # LSB
        )
        # Big-endian output
        decoded = decode_frame(header + data, 2 * 3, 32, ">")
        arr = np.frombuffer(decoded, np.dtype(">u4"))
        assert [0, 16777216, 65536, 256, 1, 4294967295] == arr[:6].tolist()
        assert [4294967295, 16777216, 65536, 256, 1, 0] == arr[6:12].tolist()
        assert [1, 16777216, 65536, 256, 1, 4294967294] == arr[12:].tolist()

        # Little-endian output
        decoded = decode_frame(header + data, 2 * 3, 32, "<")
        arr = np.frombuffer(decoded, np.dtype("<u4"))
        assert [0, 16777216, 65536, 256, 1, 4294967295] == arr[:6].tolist()
        assert [4294967295, 16777216, 65536, 256, 1, 0] == arr[6:12].tolist()
        assert [1, 16777216, 65536, 256, 1, 4294967294] == arr[12:].tolist()


@pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
class TestDecodeFrame_Datasets:
    """Test DICOM dataset decoding."""

    def test_u8_1s_1f(self):
        """Test unsigned 8-bit, 1 sample/px, 1 frame."""
        ds = INDEX["OBXXXX1A_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (600, 800) == arr.shape
        assert ">u1" == arr.dtype

        assert 244 == arr[0].min() == arr[0].max()
        assert (1, 246, 1) == tuple(arr[300, 491:494])
        assert 0 == arr[-1].min() == arr[-1].max()

    def test_u8_1s_2f(self):
        """Test unsigned 8-bit, 1 sample/px, 2 frame."""
        ds = INDEX["OBXXXX1A_rle_2frame.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 2 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (2, 600, 800) == arr.shape
        assert ">u1" == arr.dtype

        assert 244 == arr[0, 0].min() == arr[0, 0].max()
        assert (1, 246, 1) == tuple(arr[0, 300, 491:494])
        assert 0 == arr[0, -1].min() == arr[0, -1].max()

        # Frame 2 is frame 1 inverted
        assert np.array_equal((2**ds.BitsAllocated - 1) - arr[1], arr[0])

    def test_u8_3s_1f(self):
        """Test unsigned 8-bit, 3 sample/px, 1 frame."""
        ds = INDEX["SC_rgb_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (100, 100, 3) == arr.shape
        assert ">u1" == arr.dtype

        assert (255, 0, 0) == tuple(arr[5, 50, :])
        assert (255, 128, 128) == tuple(arr[15, 50, :])
        assert (0, 255, 0) == tuple(arr[25, 50, :])
        assert (128, 255, 128) == tuple(arr[35, 50, :])
        assert (0, 0, 255) == tuple(arr[45, 50, :])
        assert (128, 128, 255) == tuple(arr[55, 50, :])
        assert (0, 0, 0) == tuple(arr[65, 50, :])
        assert (64, 64, 64) == tuple(arr[75, 50, :])
        assert (192, 192, 192) == tuple(arr[85, 50, :])
        assert (255, 255, 255) == tuple(arr[95, 50, :])

    def test_u8_3s_2f(self):
        """Test unsigned 8-bit, 3 sample/px, 2 frame."""
        ds = INDEX["SC_rgb_rle_2frame.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 8 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 2 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (2, 100, 100, 3) == arr.shape
        assert ">u1" == arr.dtype

        # Frame 1
        frame = arr[0]
        assert (255, 0, 0) == tuple(frame[5, 50, :])
        assert (255, 128, 128) == tuple(frame[15, 50, :])
        assert (0, 255, 0) == tuple(frame[25, 50, :])
        assert (128, 255, 128) == tuple(frame[35, 50, :])
        assert (0, 0, 255) == tuple(frame[45, 50, :])
        assert (128, 128, 255) == tuple(frame[55, 50, :])
        assert (0, 0, 0) == tuple(frame[65, 50, :])
        assert (64, 64, 64) == tuple(frame[75, 50, :])
        assert (192, 192, 192) == tuple(frame[85, 50, :])
        assert (255, 255, 255) == tuple(frame[95, 50, :])

        # Frame 2 is frame 1 inverted
        assert np.array_equal((2**ds.BitsAllocated - 1) - arr[1], arr[0])

    def test_i16_1s_1f(self):
        """Test signed 16-bit, 1 sample/px, 1 frame."""
        ds = INDEX["MR_small_RLE.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 16 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 1 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (64, 64) == arr.shape
        assert "<i2" == arr.dtype

        assert (422, 319, 361) == tuple(arr[0, 31:34])
        assert (366, 363, 322) == tuple(arr[31, :3])
        assert (1369, 1129, 862) == tuple(arr[-1, -3:])

    def test_u16_1s_10f(self):
        """Test unsigned 16-bit, 1 sample/px, 10 frame."""
        ds = INDEX["emri_small_RLE.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 16 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 10 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (10, 64, 64) == arr.shape
        assert "<u2" == arr.dtype

        # Frame 1
        assert (206, 197, 159) == tuple(arr[0, 0, 31:34])
        assert (49, 78, 128) == tuple(arr[0, 31, :3])
        assert (362, 219, 135) == tuple(arr[0, -1, -3:])

        # Frame 5
        assert (67, 82, 44) == tuple(arr[4, 0, 31:34])
        assert (37, 41, 17) == tuple(arr[4, 31, :3])
        assert (225, 380, 355) == tuple(arr[4, -1, -3:])

        # Frame 10
        assert (72, 86, 69) == tuple(arr[-1, 0, 31:34])
        assert (25, 4, 9) == tuple(arr[-1, 31, :3])
        assert (227, 300, 147) == tuple(arr[-1, -1, -3:])

    def test_u16_3s_1f(self):
        """Test unsigned 16-bit, 3 sample/px, 1 frame."""
        ds = INDEX["SC_rgb_rle_16bit.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 16 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(ds.pixel_array, ref)
        assert (100, 100, 3) == arr.shape
        assert "<u2" == arr.dtype

        assert (65535, 0, 0) == tuple(arr[5, 50, :])
        assert (65535, 32896, 32896) == tuple(arr[15, 50, :])
        assert (0, 65535, 0) == tuple(arr[25, 50, :])
        assert (32896, 65535, 32896) == tuple(arr[35, 50, :])
        assert (0, 0, 65535) == tuple(arr[45, 50, :])
        assert (32896, 32896, 65535) == tuple(arr[55, 50, :])
        assert (0, 0, 0) == tuple(arr[65, 50, :])
        assert (16448, 16448, 16448) == tuple(arr[75, 50, :])
        assert (49344, 49344, 49344) == tuple(arr[85, 50, :])
        assert (65535, 65535, 65535) == tuple(arr[95, 50, :])

    def test_u16_3s_2f(self):
        """Test unsigned 16-bit, 3 sample/px, 2 frame."""
        ds = INDEX["SC_rgb_rle_16bit_2frame.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 16 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 2 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(ds.pixel_array, ref)
        assert (2, 100, 100, 3) == arr.shape
        assert "<u2" == arr.dtype

        # Frame 1
        frame = arr[0]
        assert (65535, 0, 0) == tuple(frame[5, 50, :])
        assert (65535, 32896, 32896) == tuple(frame[15, 50, :])
        assert (0, 65535, 0) == tuple(frame[25, 50, :])
        assert (32896, 65535, 32896) == tuple(frame[35, 50, :])
        assert (0, 0, 65535) == tuple(frame[45, 50, :])
        assert (32896, 32896, 65535) == tuple(frame[55, 50, :])
        assert (0, 0, 0) == tuple(frame[65, 50, :])
        assert (16448, 16448, 16448) == tuple(frame[75, 50, :])
        assert (49344, 49344, 49344) == tuple(frame[85, 50, :])
        assert (65535, 65535, 65535) == tuple(frame[95, 50, :])

        # Frame 2 is frame 1 inverted
        assert np.array_equal((2**ds.BitsAllocated - 1) - arr[1], arr[0])

    def test_u32_1s_1f(self):
        """Test unsigned 32-bit, 1 sample/px, 1 frame."""
        ds = INDEX["rtdose_rle_1frame.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 32 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)
        assert (10, 10) == arr.shape
        assert "<u4" == arr.dtype

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (1249000, 1249000, 1250000) == tuple(arr[0, :3])
        assert (1031000, 1029000, 1027000) == tuple(arr[4, 3:6])
        assert (803000, 801000, 798000) == tuple(arr[-1, -3:])

    def test_u32_1s_15f(self):
        """Test unsigned 32-bit, 1 sample/px, 15 frame."""
        ds = INDEX["rtdose_rle.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 32 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 15 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)

        assert arr.flags.writeable
        assert np.array_equal(arr, ref)
        assert (15, 10, 10) == arr.shape
        assert "<u4" == arr.dtype

        # Frame 1
        assert (1249000, 1249000, 1250000) == tuple(arr[0, 0, :3])
        assert (1031000, 1029000, 1027000) == tuple(arr[0, 4, 3:6])
        assert (803000, 801000, 798000) == tuple(arr[0, -1, -3:])

        # Frame 8
        assert (1253000, 1253000, 1249000) == tuple(arr[7, 0, :3])
        assert (1026000, 1023000, 1022000) == tuple(arr[7, 4, 3:6])
        assert (803000, 803000, 803000) == tuple(arr[7, -1, -3:])

        # Frame 15
        assert (1249000, 1250000, 1251000) == tuple(arr[-1, 0, :3])
        assert (1031000, 1031000, 1031000) == tuple(arr[-1, 4, 3:6])
        assert (801000, 800000, 799000) == tuple(arr[-1, -1, -3:])

    def test_u32_3s_1f(self):
        """Test unsigned 32-bit, 3 sample/px, 1 frame."""
        ds = INDEX["SC_rgb_rle_32bit.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 32 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 1 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)
        assert arr.flags.writeable
        assert (100, 100, 3) == arr.shape
        assert "<u4" == arr.dtype

        assert np.array_equal(ds.pixel_array, ref)
        assert (4294967295, 0, 0) == tuple(arr[5, 50, :])
        assert (4294967295, 2155905152, 2155905152) == tuple(arr[15, 50, :])
        assert (0, 4294967295, 0) == tuple(arr[25, 50, :])
        assert (2155905152, 4294967295, 2155905152) == tuple(arr[35, 50, :])
        assert (0, 0, 4294967295) == tuple(arr[45, 50, :])
        assert (2155905152, 2155905152, 4294967295) == tuple(arr[55, 50, :])
        assert (0, 0, 0) == tuple(arr[65, 50, :])
        assert (1077952576, 1077952576, 1077952576) == tuple(arr[75, 50, :])
        assert (3233857728, 3233857728, 3233857728) == tuple(arr[85, 50, :])
        assert (4294967295, 4294967295, 4294967295) == tuple(arr[95, 50, :])

    def test_u32_3s_2f(self):
        """Test unsigned 32-bit, 3 sample/px, 2 frame."""
        ds = INDEX["SC_rgb_rle_32bit_2frame.dcm"]["ds"]
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert 32 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel
        assert 0 == ds.PixelRepresentation
        assert 2 == getattr(ds, "NumberOfFrames", 1)

        ref = ds.pixel_array
        arr = pixel_array(ds)
        assert arr.flags.writeable
        assert (2, 100, 100, 3) == arr.shape
        assert "<u4" == arr.dtype

        assert np.array_equal(ds.pixel_array, ref)

        # Frame 1
        assert (4294967295, 0, 0) == tuple(arr[0, 5, 50, :])
        assert (4294967295, 2155905152, 2155905152) == tuple(arr[0, 15, 50, :])
        assert (0, 4294967295, 0) == tuple(arr[0, 25, 50, :])
        assert (2155905152, 4294967295, 2155905152) == tuple(arr[0, 35, 50, :])
        assert (0, 0, 4294967295) == tuple(arr[0, 45, 50, :])
        assert (2155905152, 2155905152, 4294967295) == tuple(arr[0, 55, 50, :])
        assert (0, 0, 0) == tuple(arr[0, 65, 50, :])
        assert (1077952576, 1077952576, 1077952576) == tuple(arr[0, 75, 50, :])
        assert (3233857728, 3233857728, 3233857728) == tuple(arr[0, 85, 50, :])
        assert (4294967295, 4294967295, 4294967295) == tuple(arr[0, 95, 50, :])

        # Frame 2 is frame 1 inverted
        assert np.array_equal((2**ds.BitsAllocated - 1) - arr[1], arr[0])


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
        assert b"" == decode_segment(b"\x80\x80\x80")

        # noop at start, data after
        data = (
            b"\x80\x80"  # No operation
            b"\x05\x01\x02\x03\x04\x05\x06"  # Literal
            b"\xFE\x01"  # Copy
            b"\x80"
        )
        assert (b"\x01\x02\x03\x04\x05\x06" b"\x01\x01\x01") == decode_segment(data)

        # data at start, noop middle, data at end
        data = (
            b"\x05\x01\x02\x03\x04\x05\x06"  # Literal
            b"\x80"  # No operation
            b"\xFE\x01"  # Copy
            b"\x80"
        )
        assert (b"\x01\x02\x03\x04\x05\x06" b"\x01\x01\x01") == decode_segment(data)

        # data at start, noop end
        # Copy 6 bytes literally, then 3 x 0x01
        data = b"\x05\x01\x02\x03\x04\x05\x06" b"\xFE\x01" b"\x80"
        assert (b"\x01\x02\x03\x04\x05\x06" b"\x01\x01\x01") == decode_segment(data)

    def test_literal(self):
        """Test literal output."""
        # For n < 128, read the next (n + 1) bytes literally
        # n = 0 (0x80 is 128 -> no operation)
        assert b"\x02" == decode_segment(b"\x00\x02\x80")
        # n = 1
        assert b"\x02\x03" == decode_segment(b"\x01\x02\x03\x80")
        # n = 127
        data = b"\x7f" + b"\x40" * 128 + b"\x80"
        assert b"\x40" * 128 == decode_segment(data)

    def test_copy(self):
        """Test copy output."""
        # For n > 128, copy the next byte (257 - n) times
        # n = 255, copy x2 (0x80 is 128 -> no operation)
        assert b"\x02\x02" == decode_segment(b"\xFF\x02\x80")
        # n = 254, copy x3
        assert b"\x02\x02\x02" == decode_segment(b"\xFE\x02\x80")
        # n = 129, copy x128
        assert b"\x02" * 128 == decode_segment(b"\x81\x02\x80")

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
            decode_segment(b"\x01\x02")


@pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
class TestGenerateFrames:
    """Tests for utils.generate_frames()."""

    def test_invalid_uid(self):
        index = get_indexed_datasets("1.2.840.10008.1.2.1")
        ds = index["CT_small.dcm"]["ds"]

        msg = r"Only RLE Lossless encoded pixel data encoded is supported"
        gen = generate_frames(ds)
        with pytest.raises(NotImplementedError, match=msg):
            next(gen)

    def test_missing_required(self):
        ds = deepcopy(INDEX["OBXXXX1A_rle.dcm"]["ds"])
        del ds.Rows

        msg = (
            "Unable to convert the pixel data as the following required "
            "elements are missing from the dataset: Rows"
        )
        gen = generate_frames(ds)
        with pytest.raises(AttributeError, match=msg):
            next(gen)

    def test_generator(self):
        ds = deepcopy(INDEX["OBXXXX1A_rle.dcm"]["ds"])

        gen = generate_frames(ds, reshape=True)
        arr = next(gen)
        assert (600, 800) == arr.shape
        with pytest.raises(StopIteration):
            next(gen)

    def test_multi_sample(self):
        ds = deepcopy(INDEX["SC_rgb_rle_16bit.dcm"]["ds"])

        gen = generate_frames(ds, reshape=True)
        arr = next(gen)
        assert (100, 100, 3) == arr.shape
        with pytest.raises(StopIteration):
            next(gen)
