"""Tests for decoding RLE data."""

from struct import pack
import timeit

import pytest

import numpy as np

try:
    from pydicom import dcmread
    import pydicom.config
    from pydicom.data import get_testdata_file
    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.rle_handler import (
        _parse_rle_header, _rle_decode_frame, _rle_decode_segment
    )
    from pydicom.pixel_data_handlers.util import pixel_dtype, reshape_pixel_array
    from pydicom.uid import RLELossless
    HAVE_PYDICOM = True
except ImportError:
    HAVE_PYDICOM = False

from rle.data import get_indexed_datasets
from rle._rle import decode_segment, decode_frame, parse_header
from rle.utils import generate_frames


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
        header.extend(pack(f'<{len(offsets)}L', *offsets))
        # Add padding
        header.extend(b'\x00' * (64 - len(header)))

        offsets += [0] * (15 - len(offsets))

        assert len(header) == 64
        assert offsets == parse_header(bytes(header))


if HAVE_PYDICOM:
    # Paths to the test datasets
    # EXPL: Explicit VR Little Endian
    # RLE: RLE Lossless
    # 8/8-bit, 1 sample/pixel, 1 frame
    EXPL_8_1_1F = get_testdata_file("OBXXXX1A.dcm")
    RLE_8_1_1F = get_testdata_file("OBXXXX1A_rle.dcm")
    # 8/8-bit, 1 sample/pixel, 2 frame
    EXPL_8_1_2F = get_testdata_file("OBXXXX1A_2frame.dcm")
    RLE_8_1_2F = get_testdata_file("OBXXXX1A_rle_2frame.dcm")
    # 8/8-bit, 3 sample/pixel, 1 frame
    EXPL_8_3_1F = get_testdata_file("SC_rgb.dcm")
    RLE_8_3_1F = get_testdata_file("SC_rgb_rle.dcm")
    # 8/8-bit, 3 sample/pixel, 2 frame
    EXPL_8_3_2F = get_testdata_file("SC_rgb_2frame.dcm")
    RLE_8_3_2F = get_testdata_file("SC_rgb_rle_2frame.dcm")
    # 16/16-bit, 1 sample/pixel, 1 frame
    EXPL_16_1_1F = get_testdata_file("MR_small.dcm")
    RLE_16_1_1F = get_testdata_file("MR_small_RLE.dcm")
    # 16/12-bit, 1 sample/pixel, 10 frame
    EXPL_16_1_10F = get_testdata_file("emri_small.dcm")
    RLE_16_1_10F = get_testdata_file("emri_small_RLE.dcm")
    # 16/16-bit, 3 sample/pixel, 1 frame
    EXPL_16_3_1F = get_testdata_file("SC_rgb_16bit.dcm")
    RLE_16_3_1F = get_testdata_file("SC_rgb_rle_16bit.dcm")
    # 16/16-bit, 3 sample/pixel, 2 frame
    EXPL_16_3_2F = get_testdata_file("SC_rgb_16bit_2frame.dcm")
    RLE_16_3_2F = get_testdata_file("SC_rgb_rle_16bit_2frame.dcm")
    # 32/32-bit, 1 sample/pixel, 1 frame
    EXPL_32_1_1F = get_testdata_file("rtdose_1frame.dcm")
    RLE_32_1_1F = get_testdata_file("rtdose_rle_1frame.dcm")
    # 32/32-bit, 1 sample/pixel, 15 frame
    EXPL_32_1_15F = get_testdata_file("rtdose.dcm")
    RLE_32_1_15F = get_testdata_file("rtdose_rle.dcm")
    # 32/32-bit, 3 sample/pixel, 1 frame
    EXPL_32_3_1F = get_testdata_file("SC_rgb_32bit.dcm")
    RLE_32_3_1F = get_testdata_file("SC_rgb_rle_32bit.dcm")
    # 32/32-bit, 3 sample/pixel, 2 frame
    EXPL_32_3_2F = get_testdata_file("SC_rgb_32bit_2frame.dcm")
    RLE_32_3_2F = get_testdata_file("SC_rgb_rle_32bit_2frame.dcm")


@pytest.mark.skip()
class TestNumpy_RLEHandler:
    """Tests for handling datasets with the handler."""
    def setup(self):
        """Setup the environment."""
        self.original_handlers = pydicom.config.pixel_data_handlers
        #pydicom.config.pixel_data_handlers = [RLE_HANDLER]

    def teardown(self):
        """Restore the environment."""
        #pydicom.config.pixel_data_handlers = self.original_handlers
        pass

    def test_pixel_array_signed(self):
        """Test pixel_array for unsigned -> signed data."""
        ds = dcmread(RLE_8_1_1F)
        # 0 is unsigned int, 1 is 2's complement
        assert ds.PixelRepresentation == 0
        ds.PixelRepresentation = 1
        ref = _get_pixel_array(EXPL_8_1_1F)
        arr = ds.pixel_array

        assert not np.array_equal(arr, ref)
        assert (600, 800) == arr.shape
        assert -12 == arr[0].min() == arr[0].max()
        assert (1, -10, 1) == tuple(arr[300, 491:494])
        assert 0 == arr[-1].min() == arr[-1].max()

    def test_pixel_array_1bit_raises(self):
        """Test pixel_array for 1-bit raises exception."""
        ds = dcmread(RLE_8_3_1F)
        ds.BitsAllocated = 1
        with pytest.raises(NotImplementedError,
                           match="Bits Allocated' value of 1"):
            ds.pixel_array

    def test_pixel_array_8bit_1sample_1f(self):
        """Test pixel_array for 8-bit, 1 sample/pixel, 1 frame."""
        ds = dcmread(RLE_8_1_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 8
        assert ds.SamplesPerPixel == 1
        assert 'NumberOfFrames' not in ds
        ref = _get_pixel_array(EXPL_8_1_1F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (600, 800) == arr.shape
        assert 244 == arr[0].min() == arr[0].max()
        assert (1, 246, 1) == tuple(arr[300, 491:494])
        assert 0 == arr[-1].min() == arr[-1].max()

    def test_decompress_with_handler(self):
        """Test that decompress works with the correct handler."""
        ds = dcmread(RLE_8_1_1F)
        with pytest.raises(ValueError,
                           match="'zip' is not a known handler name"):
            ds.decompress(handler_name='zip')
        with pytest.raises(NotImplementedError, match='Unable to decode*'):
            ds.decompress(handler_name='numpy')
        ds.decompress(handler_name='rle')
        assert hasattr(ds, '_pixel_array')
        arr = ds.pixel_array
        assert (600, 800) == arr.shape
        assert 244 == arr[0].min() == arr[0].max()
        assert (1, 246, 1) == tuple(arr[300, 491:494])
        assert 0 == arr[-1].min() == arr[-1].max()

    def test_pixel_array_8bit_1sample_2f(self):
        """Test pixel_array for 8-bit, 1 sample/pixel, 2 frame."""
        ds = dcmread(RLE_8_1_2F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 8
        assert ds.SamplesPerPixel == 1
        assert ds.NumberOfFrames == 2
        ref = _get_pixel_array(EXPL_8_1_2F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (2, 600, 800) == arr.shape
        assert 244 == arr[0, 0].min() == arr[0, 0].max()
        assert (1, 246, 1) == tuple(arr[0, 300, 491:494])
        assert 0 == arr[0, -1].min() == arr[0, -1].max()

        # Frame 2 is frame 1 inverted
        assert np.array_equal((2**ds.BitsAllocated - 1) - arr[1], arr[0])

    def test_pixel_array_8bit_3sample_1f(self):
        """Test pixel_array for 8-bit, 3 sample/pixel, 1 frame."""
        ds = dcmread(RLE_8_3_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 8
        assert ds.SamplesPerPixel == 3
        assert 'NumberOfFrames' not in ds
        ref = _get_pixel_array(EXPL_8_3_1F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
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

    def test_pixel_array_8bit_3sample_2f(self):
        """Test pixel_array for 8-bit, 3 sample/pixel, 2 frame."""
        ds = dcmread(RLE_8_3_2F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 8
        assert ds.SamplesPerPixel == 3
        assert ds.NumberOfFrames == 2
        ref = _get_pixel_array(EXPL_8_3_2F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)

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

    def test_pixel_array_16bit_1sample_1f(self):
        """Test pixel_array for 16-bit, 1 sample/pixel, 1 frame."""
        ds = dcmread(RLE_16_1_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 16
        assert ds.SamplesPerPixel == 1
        assert 'NumberOfFrames' not in ds
        assert ds.PixelRepresentation == 1
        ref = _get_pixel_array(EXPL_16_1_1F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (64, 64) == arr.shape

        assert (422, 319, 361) == tuple(arr[0, 31:34])
        assert (366, 363, 322) == tuple(arr[31, :3])
        assert (1369, 1129, 862) == tuple(arr[-1, -3:])

    def test_pixel_array_16bit_1sample_10f(self):
        """Test pixel_array for 16-bit, 1, sample/pixel, 10 frame."""
        ds = dcmread(RLE_16_1_10F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 16
        assert ds.SamplesPerPixel == 1
        assert ds.NumberOfFrames == 10
        ref = _get_pixel_array(EXPL_16_1_10F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (10, 64, 64) == arr.shape

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

    def test_pixel_array_16bit_3sample_1f(self):
        """Test pixel_array for 16-bit, 3 sample/pixel, 1 frame."""
        ds = dcmread(RLE_16_3_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 16
        assert ds.SamplesPerPixel == 3
        assert 'NumberOfFrames' not in ds
        arr = ds.pixel_array
        ref = _get_pixel_array(EXPL_16_3_1F)

        assert arr.flags.writeable

        assert np.array_equal(ds.pixel_array, ref)

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

    def test_pixel_array_16bit_3sample_2f(self):
        """Test pixel_array for 16-bit, 3, sample/pixel, 10 frame."""
        ds = dcmread(RLE_16_3_2F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 16
        assert ds.SamplesPerPixel == 3
        assert ds.NumberOfFrames == 2
        arr = ds.pixel_array
        ref = _get_pixel_array(EXPL_16_3_2F)

        assert arr.flags.writeable

        assert np.array_equal(ds.pixel_array, ref)

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

    def test_pixel_array_32bit_1sample_1f(self):
        """Test pixel_array for 32-bit, 1 sample/pixel, 1 frame."""
        ds = dcmread(RLE_32_1_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 32
        assert ds.SamplesPerPixel == 1
        assert 'NumberOfFrames' not in ds
        ref = _get_pixel_array(EXPL_32_1_1F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (10, 10) == arr.shape
        assert (1249000, 1249000, 1250000) == tuple(arr[0, :3])
        assert (1031000, 1029000, 1027000) == tuple(arr[4, 3:6])
        assert (803000, 801000, 798000) == tuple(arr[-1, -3:])

    def test_pixel_array_32bit_1sample_15f(self):
        """Test pixel_array for 32-bit, 1, sample/pixel, 15 frame."""
        ds = dcmread(RLE_32_1_15F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 32
        assert ds.SamplesPerPixel == 1
        assert ds.NumberOfFrames == 15
        ref = _get_pixel_array(EXPL_32_1_15F)
        arr = ds.pixel_array

        assert arr.flags.writeable

        assert np.array_equal(arr, ref)
        assert (15, 10, 10) == arr.shape

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

    def test_pixel_array_32bit_3sample_1f(self):
        """Test pixel_array for 32-bit, 3 sample/pixel, 1 frame."""
        ds = dcmread(RLE_32_3_1F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 32
        assert ds.SamplesPerPixel == 3
        assert 'NumberOfFrames' not in ds
        arr = ds.pixel_array
        ref = _get_pixel_array(EXPL_32_3_1F)

        assert arr.flags.writeable

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

    def test_pixel_array_32bit_3sample_2f(self):
        """Test pixel_array for 32-bit, 3, sample/pixel, 2 frame."""
        ds = dcmread(RLE_32_3_2F)
        assert ds.file_meta.TransferSyntaxUID == RLELossless
        assert ds.BitsAllocated == 32
        assert ds.SamplesPerPixel == 3
        assert ds.NumberOfFrames == 2
        arr = ds.pixel_array
        ref = _get_pixel_array(EXPL_32_3_2F)

        assert arr.flags.writeable

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


@pytest.mark.skipif(not HAVE_PYDICOM, reason="No pydicom")
class TestDecodeFrame:
    def test_one(self):
        # 600 x 800, 8 bit, pr0
        #ds = INDEX['OBXXXX1A_rle.dcm']['ds']
        # 64 x 64, 16 bit, pr1
        #ds = INDEX['MR_small_RLE.dcm']['ds']
        # 100 x 100, 32 bit, pr0
        #ds = INDEX["SC_rgb_rle_32bit.dcm"]['ds']
        ds = INDEX["SC_rgb_rle.dcm"]['ds']
        #print(ds[0x00280000:0x00300000])
        ref = ds.pixel_array

        frame_gen = generate_pixel_data_frame(ds.PixelData)
        frame = next(frame_gen)

        #offsets = _parse_rle_header(frame[:64])
        #print(offsets)

        #result = decode_segment(frame[offsets[0]:offsets[1]])
        #print(result[:20], len(result))

        #px_per_sample = ds.Rows * ds.Columns
        #bits_per_px = ds.BitsAllocated
        #frame = decode_frame(frame, px_per_sample, bits_per_px)

        #dtype = pixel_dtype(ds).newbyteorder('>')
        #arr = np.frombuffer(frame, dtype=dtype)
        #arr = reshape_pixel_array(ds, arr)
        #print(arr, arr.shape)

        gen = generate_frames(ds)
        arr = next(gen)

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(ref)
        ax2.imshow(arr)
        plt.show()


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
