import asv
import timeit

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.pixels.encoders.native import _rle_encode_row, rle_encode_frame
from pydicom.pixels.utils import reshape_pixel_array
from pydicom.uid import RLELossless

from ljdata import get_indexed_datasets
from rle.utils import pixel_array, decode_frame
from rle._rle import encode_row, encode_frame

INDEX = get_indexed_datasets(RLELossless)
# 8/8-bit, 1 sample/pixel, 1 frame
EXPL_8_1_1F = get_testdata_file("OBXXXX1A.dcm")
# 8/8-bit, 3 sample/pixel, 1 frame
EXPL_8_3_1F = get_testdata_file("SC_rgb.dcm")
# 16/16-bit, 1 sample/pixel, 1 frame
EXPL_16_1_1F = get_testdata_file("MR_small.dcm")
# 16/16-bit, 3 sample/pixel, 1 frame
EXPL_16_3_1F = get_testdata_file("SC_rgb_16bit.dcm")
# 32/32-bit, 1 sample/pixel, 1 frame
EXPL_32_1_1F = get_testdata_file("rtdose_1frame.dcm")
# 32/32-bit, 3 sample/pixel, 1 frame
EXPL_32_3_1F = get_testdata_file("SC_rgb_32bit.dcm")


class TimeEncodeRow:
    def setup(self):
        self.no_runs = 100000
        ds = u8_1s_1f_rle
        arr = ds.pixel_array
        self.row_arr = arr[0, :].ravel()
        self.row = self.row_arr.tobytes()

    def time_rle(self):
        for _ in range(self.no_runs):
            encode_row(self.row)

    def time_default(self):
        for _ in range(self.no_runs):
            _rle_encode_row(self.row_arr)


class TimePYDEncodeFrame:
    """Time tests for rle_handler.rle_encode_frame."""

    def setup(self):
        ds = dcmread(EXPL_8_1_1F)
        self.arr8_1 = ds.pixel_array
        ds = dcmread(EXPL_8_3_1F)
        self.arr8_3 = ds.pixel_array
        ds = dcmread(EXPL_16_1_1F)
        self.arr16_1 = ds.pixel_array
        ds = dcmread(EXPL_16_3_1F)
        self.arr16_3 = ds.pixel_array
        ds = dcmread(EXPL_32_1_1F)
        self.arr32_1 = ds.pixel_array
        ds = dcmread(EXPL_32_3_1F)
        self.arr32_3 = ds.pixel_array

        self.no_runs = 1000

    def time_08_1(self):
        """Time encoding 8 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr8_1)

    def time_08_3(self):
        """Time encoding 8 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr8_3)

    def time_16_1(self):
        """Time encoding 16 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr16_1)

    def time_16_3(self):
        """Time encoding 16 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr16_3)

    def time_32_1(self):
        """Time encoding 32 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr32_1)

    def time_32_3(self):
        """Time encoding 32 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            rle_encode_frame(self.arr32_3)


class TimeRLEEncodeFrame:
    def setup(self):
        ds = dcmread(EXPL_8_1_1F)
        self.ds8_1 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )
        ds = dcmread(EXPL_8_3_1F)
        self.ds8_3 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )
        ds = dcmread(EXPL_16_1_1F)
        self.ds16_1 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )
        ds = dcmread(EXPL_16_3_1F)
        self.ds16_3 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )
        ds = dcmread(EXPL_32_1_1F)
        self.ds32_1 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )
        ds = dcmread(EXPL_32_3_1F)
        self.ds32_3 = (
            ds.PixelData,
            ds.Rows,
            ds.Columns,
            ds.SamplesPerPixel,
            ds.BitsAllocated,
        )

        self.no_runs = 1000

    def time_08_1(self):
        """Time encoding 8 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds8_1, "<")

    def time_08_3(self):
        """Time encoding 8 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds8_3, "<")

    def time_16_1(self):
        """Time encoding 16 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds16_1, "<")

    def time_16_3(self):
        """Time encoding 16 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds16_3, "<")

    def time_32_1(self):
        """Time encoding 32 bit 1 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds32_1, "<")

    def time_32_3(self):
        """Time encoding 32 bit 3 sample/pixel."""
        for ii in range(self.no_runs):
            encode_frame(*self.ds32_3, "<")
