"""Tests for RLE encoding."""

import numpy as np
import pytest

from rle._rle import encode_row


# Tests for RLE encoding
REFERENCE_ENCODE_ROW = [
    # Input, output
    pytest.param([], b'', id='1'),
    # Replicate run tests
    # 2 (min) replicate
    pytest.param([0] * 2, b'\xff\x00', id='1'),
    pytest.param([0] * 3, b'\xfe\x00', id='2'),
    pytest.param([0] * 64, b'\xc1\x00', id='3'),
    pytest.param([0] * 127, b'\x82\x00', id='4'),
    # 128 (max) replicate
    pytest.param([0] * 128, b'\x81\x00', id='5'),
    # 128 (max) replicate, 1 (min) literal
    pytest.param([0] * 129, b'\x81\x00\x00\x00', id='6'),
    # 128 (max) replicate, 2 (min) replicate
    pytest.param([0] * 130, b'\x81\x00\xff\x00', id='7'),
    # 128 (max) x 5 replicates
    pytest.param([0] * 128 * 5, b'\x81\x00' * 5, id='8'),
    # Literal run tests
    # 1 (min) literal
    pytest.param([0], b'\x00\x00', id='9'),
    pytest.param([0, 1], b'\x01\x00\x01', id='10'),
    pytest.param([0, 1, 2], b'\x02\x00\x01\x02', id='11'),
    pytest.param([0, 1] * 32, b'\x3f' + b'\x00\x01' * 32, id='12'),
    # 127 literal
    pytest.param([0, 1] * 63 + [2], b'\x7e' + b'\x00\x01' * 63 + b'\x02', id='13'),
    # 128 literal (max)
    pytest.param([0, 1] * 64, b'\x7f' + b'\x00\x01' * 64, id='14'),
    # 128 (max) literal, 1 (min) literal
    pytest.param([0, 1] * 64 + [2], b'\x7f' + b'\x00\x01' * 64 + b'\x00\x02', id='15'),
    # 128 (max) x 5 literals
    pytest.param([0, 1] * 64 * 5, (b'\x7f' + b'\x00\x01' * 64) * 5, id='16'),
    # Combination run tests
    # 1 (min) literal, 1 (min) replicate
    pytest.param([0, 1, 1], b'\x00\x00\xff\x01', id='17'),
    # 1 (min) literal, 128 (max) replicate
    pytest.param([0] + [1] * 128, b'\x00\x00\x81\x01', id='18'),
    # 128 (max) literal, 2 (min) replicate
    pytest.param([0, 1] * 64 + [2] * 2, b'\x7f' + b'\x00\x01' * 64 + b'\xff\x02', id='19'),
    # 128 (max) literal, 128 (max) replicate
    pytest.param([0, 1] * 64 + [2] * 128, b'\x7f' + b'\x00\x01' * 64 + b'\x81\x02', id='20'),
    # 2 (min) replicate, 1 (min) literal
    pytest.param([0, 0, 1], b'\xff\x00\x00\x01', id='21'),
    # 2 (min) replicate, 128 (max) literal
    pytest.param([0, 0] + [1, 2] * 64, b'\xff\x00\x7f' + b'\x01\x02' * 64, id='22'),
    # 128 (max) replicate, 1 (min) literal
    pytest.param([0] * 128 + [1], b'\x81\x00\x00\x01', id='23'),
    # 128 (max) replicate, 128 (max) literal
    pytest.param([0] * 128 + [1, 2] * 64, b'\x81\x00\x7f' + b'\x01\x02' * 64, id='24'),
]


class TestEncodeRow:
    """Tests for _rle.encode_row."""
    @pytest.mark.parametrize('row, ref', REFERENCE_ENCODE_ROW)
    def test_encode(self, row, ref):
        """Test encoding an empty row."""
        row = np.asarray(row)
        assert ref == encode_row(row.tobytes())


@pytest.mark.skip()
class TestEncodeSegment:
    """Tests for rle_handler._rle_encode_segment."""
    def test_one_row(self):
        """Test encoding data that contains only a single row."""
        ds = dcmread(RLE_8_1_1F)
        pixel_data = defragment_data(ds.PixelData)
        decoded = _rle_decode_segment(pixel_data[64:])
        assert ds.Rows * ds.Columns == len(decoded)
        arr = np.frombuffer(decoded, 'uint8').reshape(ds.Rows, ds.Columns)

        # Re-encode a single row of the decoded data
        row = arr[0]
        assert (ds.Columns,) == row.shape
        encoded = _rle_encode_segment(row)

        # Decode the re-encoded data and check that it's the same
        redecoded = _rle_decode_segment(encoded)
        assert ds.Columns == len(redecoded)
        assert decoded[:ds.Columns] == redecoded

    def test_cycle(self):
        """Test the decoded data remains the same after encoding/decoding."""
        ds = dcmread(RLE_8_1_1F)
        pixel_data = defragment_data(ds.PixelData)
        decoded = _rle_decode_segment(pixel_data[64:])
        assert ds.Rows * ds.Columns == len(decoded)
        arr = np.frombuffer(decoded, 'uint8').reshape(ds.Rows, ds.Columns)
        # Re-encode the decoded data
        encoded = _rle_encode_segment(arr)

        # Decode the re-encoded data and check that it's the same
        redecoded = _rle_decode_segment(encoded)
        assert ds.Rows * ds.Columns == len(redecoded)
        assert decoded == redecoded


@pytest.mark.skip()
class TestEncodePlane:
    """Tests for rle_handler._rle_encode_plane."""
    def test_8bit(self):
        """Test encoding an 8-bit plane into 1 segment."""
        ds = dcmread(RLE_8_1_1F)
        pixel_data = defragment_data(ds.PixelData)
        decoded = _rle_decode_frame(pixel_data, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.BitsAllocated // 8 == len(decoded)
        arr = np.frombuffer(decoded, 'uint8').reshape(ds.Rows, ds.Columns)
        # Re-encode the decoded data
        encoded = bytearray()
        nr_segments = 0
        for segment in _rle_encode_plane(arr):
            encoded.extend(segment)
            nr_segments += 1

        # Add header
        header = b'\x01\x00\x00\x00\x40\x00\x00\x00'
        header += b'\x00' * (64 - len(header))

        assert 1 == nr_segments

        # Decode the re-encoded data and check that it's the same
        redecoded = _rle_decode_frame(header + encoded,
                                      ds.Rows, ds.Columns,
                                      ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.SamplesPerPixel == len(redecoded)
        assert decoded == redecoded

    def test_16bit(self):
        """Test encoding a 16-bit plane into 2 segments."""
        ds = dcmread(RLE_16_1_1F)
        pixel_data = defragment_data(ds.PixelData)
        decoded = _rle_decode_frame(pixel_data, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.BitsAllocated // 8 == len(decoded)

        # `decoded` is in big endian byte ordering
        dtype = np.dtype('uint16').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype).reshape(ds.Rows, ds.Columns)

        # Re-encode the decoded data
        encoded = bytearray()
        nr_segments = 0
        offsets = [64]
        for segment in _rle_encode_plane(arr):
            offsets.append(offsets[nr_segments] + len(segment))
            encoded.extend(segment)
            nr_segments += 1

        assert 2 == nr_segments

        # Add header
        header = b'\x02\x00\x00\x00'
        header += pack('<2L', *offsets[:-1])
        header += b'\x00' * (64 - len(header))

        # Decode the re-encoded data and check that it's the same
        redecoded = _rle_decode_frame(header + encoded,
                                      ds.Rows, ds.Columns,
                                      ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.BitsAllocated // 8 == len(redecoded)
        assert decoded == redecoded

    def test_16bit_segment_order(self):
        """Test that the segment order per 16-bit sample is correct."""
        # Native byte ordering
        data = b'\x00\x00\x01\xFF\xFE\x00\xFF\xFF\x10\x12'
        dtype = np.dtype('uint16')
        arr = np.frombuffer(data, dtype)

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 2 == len(segments)

        # Each segment should start with a literal run marker of 0x04
        # and MSB should be first segment, then LSB in second
        if sys.byteorder == 'little':
            assert b'\x04\x00\xFF\x00\xFF\x12' == segments[0]
            assert b'\x04\x00\x01\xFE\xFF\x10' == segments[1]
        else:
            assert b'\x04\x00\x01\xFE\xFF\x10' == segments[0]
            assert b'\x04\x00\xFF\x00\xFF\x12' == segments[1]

        # Little endian
        arr = np.frombuffer(data, dtype.newbyteorder('<'))
        assert [0, 65281, 254, 65535, 4624] == arr.tolist()

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 2 == len(segments)
        assert b'\x04\x00\xFF\x00\xFF\x12' == segments[0]
        assert b'\x04\x00\x01\xFE\xFF\x10' == segments[1]

        # Big endian
        arr = np.frombuffer(data, dtype.newbyteorder('>'))
        assert [0, 511, 65024, 65535, 4114] == arr.tolist()

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 2 == len(segments)
        assert b'\x04\x00\x01\xFE\xFF\x10' == segments[0]
        assert b'\x04\x00\xFF\x00\xFF\x12' == segments[1]

    def test_32bit(self):
        """Test encoding a 32-bit plane into 4 segments."""
        ds = dcmread(RLE_32_1_1F)
        pixel_data = defragment_data(ds.PixelData)
        decoded = _rle_decode_frame(pixel_data, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.BitsAllocated // 8 == len(decoded)

        # `decoded` is in big endian byte ordering
        dtype = np.dtype('uint32').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype).reshape(ds.Rows, ds.Columns)

        # Re-encode the decoded data
        encoded = bytearray()
        nr_segments = 0
        offsets = [64]
        for segment in _rle_encode_plane(arr):
            offsets.append(offsets[nr_segments] + len(segment))
            encoded.extend(segment)
            nr_segments += 1

        assert 4 == nr_segments

        # Add header
        header = b'\x04\x00\x00\x00'
        header += pack('<4L', *offsets[:-1])
        header += b'\x00' * (64 - len(header))

        # Decode the re-encoded data and check that it's the same
        redecoded = _rle_decode_frame(header + encoded,
                                      ds.Rows, ds.Columns,
                                      ds.SamplesPerPixel, ds.BitsAllocated)
        assert ds.Rows * ds.Columns * ds.BitsAllocated // 8 == len(redecoded)
        assert decoded == redecoded

    def test_32bit_segment_order(self):
        """Test that the segment order per 32-bit sample is correct."""
        # Native byte ordering
        data = b'\x00\x00\x00\x00\x01\xFF\xFE\x0A\xFF\xFC\x10\x12'
        dtype = np.dtype('uint32')
        arr = np.frombuffer(data, dtype)

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 4 == len(segments)

        # Each segment should start with a literal run marker of 0x02
        if sys.byteorder == 'little':
            assert b'\x02\x00\x0A\x12' == segments[0]
            assert b'\x02\x00\xFE\x10' == segments[1]
            assert b'\x02\x00\xFF\xFC' == segments[2]
            assert b'\x02\x00\x01\xFF' == segments[3]
        else:
            assert b'\x02\x00\x01\xFF' == segments[0]
            assert b'\x02\x00\xFF\xFC' == segments[1]
            assert b'\x02\x00\xFE\x10' == segments[2]
            assert b'\x02\x00\x0A\x12' == segments[3]

        # Little endian
        arr = np.frombuffer(data, dtype.newbyteorder('<'))
        assert [0, 184483585, 303103231] == arr.tolist()

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 4 == len(segments)
        assert b'\x02\x00\x0A\x12' == segments[0]
        assert b'\x02\x00\xFE\x10' == segments[1]
        assert b'\x02\x00\xFF\xFC' == segments[2]
        assert b'\x02\x00\x01\xFF' == segments[3]

        # Big endian
        arr = np.frombuffer(data, dtype.newbyteorder('>'))
        assert [0, 33553930, 4294709266] == arr.tolist()

        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        assert 4 == len(segments)
        assert b'\x02\x00\x01\xFF' == segments[0]
        assert b'\x02\x00\xFF\xFC' == segments[1]
        assert b'\x02\x00\xFE\x10' == segments[2]
        assert b'\x02\x00\x0A\x12' == segments[3]

    def test_padding(self):
        """Test that odd length encoded segments are padded."""
        data = b'\x00\x04\x01\x15'
        arr = np.frombuffer(data, 'uint8')
        segments = []
        for segment in _rle_encode_plane(arr):
            segments.append(segment)

        # The segment should start with a literal run marker of 0x03
        #   then 4 bytes of RLE encoded data, then 0x00 padding
        assert b'\x03\x00\x04\x01\x15\x00' == segments[0]


@pytest.mark.skip()
class TestNumpy_RLEEncodeFrame:
    """Tests for rle_handler.rle_encode_frame."""
    def setup(self):
        """Setup the tests."""
        # Create a dataset skeleton for use in the cycle tests
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
        ds.Rows = 2
        ds.Columns = 4
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 1
        self.ds = ds

    def test_cycle_8bit_1sample(self):
        """Test an encode/decode cycle for 8-bit 1 sample/pixel."""
        ds = dcmread(EXPL_8_1_1F)
        ref = ds.pixel_array
        assert 8 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        dtype = np.dtype('uint8').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype)
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_cycle_8bit_3sample(self):
        """Test an encode/decode cycle for 8-bit 3 sample/pixel."""
        ds = dcmread(EXPL_8_3_1F)
        ref = ds.pixel_array
        assert 8 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        arr = np.frombuffer(decoded, 'uint8')
        # The decoded data is planar configuration 1
        ds.PlanarConfiguration = 1
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_cycle_16bit_1sample(self):
        """Test an encode/decode cycle for 16-bit 1 sample/pixel."""
        ds = dcmread(EXPL_16_1_1F)
        ref = ds.pixel_array
        assert 16 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        dtype = np.dtype('uint16').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype)
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_cycle_16bit_3sample(self):
        """Test an encode/decode cycle for 16-bit 3 sample/pixel."""
        ds = dcmread(EXPL_16_3_1F)
        ref = ds.pixel_array
        assert 16 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        dtype = np.dtype('uint16').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype)
        # The decoded data is planar configuration 1
        ds.PlanarConfiguration = 1
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_cycle_32bit_1sample(self):
        """Test an encode/decode cycle for 32-bit 1 sample/pixel."""
        ds = dcmread(EXPL_32_1_1F)
        ref = ds.pixel_array
        assert 32 == ds.BitsAllocated
        assert 1 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        dtype = np.dtype('uint32').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype)
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_cycle_32bit_3sample(self):
        """Test an encode/decode cycle for 32-bit 3 sample/pixel."""
        ds = dcmread(EXPL_32_3_1F)
        ref = ds.pixel_array
        assert 32 == ds.BitsAllocated
        assert 3 == ds.SamplesPerPixel

        encoded = rle_encode_frame(ref)
        decoded = _rle_decode_frame(encoded, ds.Rows, ds.Columns,
                                    ds.SamplesPerPixel, ds.BitsAllocated)
        dtype = np.dtype('uint32').newbyteorder('>')
        arr = np.frombuffer(decoded, dtype)
        # The decoded data is planar configuration 1
        ds.PlanarConfiguration = 1
        arr = reshape_pixel_array(ds, arr)

        assert np.array_equal(ref, arr)

    def test_16_segments_raises(self):
        """Test that trying to encode 16-segments raises exception."""
        arr = np.asarray([[[1, 2, 3, 4]]], dtype='uint32')
        assert (1, 1, 4) == arr.shape
        assert 4 == arr.dtype.itemsize

        msg = (
            r"Unable to encode as the DICOM standard only allows "
            r"a maximum of 15 segments in RLE encoded data"
        )
        with pytest.raises(ValueError, match=msg):
            rle_encode_frame(arr)

    def test_15_segment(self):
        """Test encoding 15-segments works as expected."""
        arr = np.asarray(
            [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]],
            dtype='uint8'
        )
        assert (1, 1, 15) == arr.shape
        assert 1 == arr.dtype.itemsize

        encoded = rle_encode_frame(arr)
        header = (
            b'\x0f\x00\x00\x00'
            b'\x40\x00\x00\x00'
            b'\x42\x00\x00\x00'
            b'\x44\x00\x00\x00'
            b'\x46\x00\x00\x00'
            b'\x48\x00\x00\x00'
            b'\x4a\x00\x00\x00'
            b'\x4c\x00\x00\x00'
            b'\x4e\x00\x00\x00'
            b'\x50\x00\x00\x00'
            b'\x52\x00\x00\x00'
            b'\x54\x00\x00\x00'
            b'\x56\x00\x00\x00'
            b'\x58\x00\x00\x00'
            b'\x5a\x00\x00\x00'
            b'\x5c\x00\x00\x00'
        )
        assert header == encoded[:64]
        assert (
            b'\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06'
            b'\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c'
            b'\x00\x0d\x00\x0e\x00\x0f'
        ) == encoded[64:]

    def test_encoding_multiple_frames_raises(self):
        """Test encoding multiple framed pixel data raises exception."""
        # Note: only works with multi-sample data
        ds = dcmread(EXPL_8_3_2F)
        arr = ds.pixel_array
        assert ds.NumberOfFrames > 1
        assert len(arr.shape) == 4
        msg = (
            r"Unable to encode multiple frames at once, please encode one "
            r"frame at a time"
        )
        with pytest.raises(ValueError, match=msg):
            rle_encode_frame(arr)

    def test_single_row_1sample(self):
        """Test encoding a single row of 1 sample/pixel data."""
        # Rows 1, Columns 5, SamplesPerPixel 1
        arr = np.asarray([[0, 1, 2, 3, 4]], dtype='uint8')
        assert (1, 5) == arr.shape
        encoded = rle_encode_frame(arr)
        header = b'\x01\x00\x00\x00\x40\x00\x00\x00' + b'\x00' * 56
        assert header == encoded[:64]
        assert b'\x04\x00\x01\x02\x03\x04' == encoded[64:]

    def test_single_row_3sample(self):
        """Test encoding a single row of 3 samples/pixel data."""
        # Rows 1, Columns 5, SamplesPerPixel 3
        arr = np.asarray(
            [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]],
            dtype='uint8'
        )
        assert (1, 5, 3) == arr.shape
        encoded = rle_encode_frame(arr)
        header = (
            b'\x03\x00\x00\x00'
            b'\x40\x00\x00\x00'
            b'\x46\x00\x00\x00'
            b'\x4c\x00\x00\x00'
        )
        header += b'\x00' * (64 - len(header))
        assert header == encoded[:64]
        assert (
            b'\x04\x00\x01\x02\x03\x04'
            b'\x04\x00\x01\x02\x03\x04'
            b'\x04\x00\x01\x02\x03\x04'
        ) == encoded[64:]
