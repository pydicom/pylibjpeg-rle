
import asv
import timeit

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.rle_handler import (
    get_pixeldata, _rle_decode_frame, _rle_encode_row, rle_encode_frame
)
from pydicom.pixel_data_handlers.util import reshape_pixel_array
from pydicom.uid import RLELossless

from ljdata import get_indexed_datasets
from rle.utils import pixel_array, decode_frame
from rle._rle import encode_row, encode_frame

INDEX = get_indexed_datasets(RLELossless)


u8_1s_1f_rle = INDEX["OBXXXX1A_rle.dcm"]['ds']
u8_1s_1f = dcmread(get_testdata_file("OBXXXX1A.dcm"))
u8_1s_2f_rle = INDEX["OBXXXX1A_rle_2frame.dcm"]['ds']


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

class TimeEncodeFrame:
    def setup(self):
        self.arr_u8_1s_1f = u8_1s_1f.pixel_array
        self.bytes_u8_1s_1f = u8_1s_1f.PixelData
        self.parms = (
            u8_1s_1f.Rows,
            u8_1s_1f.Columns,
            u8_1s_1f.SamplesPerPixel,
            u8_1s_1f.BitsAllocated,
            '<'
        )
        self.no_runs = 100

    def time_default(self):
        for _ in range(self.no_runs):
            rle_encode_frame(self.arr_u8_1s_1f)

    def time_rle(self):
        for _ in range(self.no_runs):
            encode_frame(self.bytes_u8_1s_1f, *self.parms)
