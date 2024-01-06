import asv
import timeit

from pydicom import dcmread
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.rle_handler import get_pixeldata, _rle_decode_frame
from pydicom.pixel_data_handlers.util import reshape_pixel_array
from pydicom.uid import RLELossless

from ljdata import get_indexed_datasets
from rle.utils import pixel_array, decode_frame

INDEX = get_indexed_datasets(RLELossless)


u8_1s_1f = INDEX["OBXXXX1A_rle.dcm"]["ds"]
u8_1s_2f = INDEX["OBXXXX1A_rle_2frame.dcm"]["ds"]
u8_3s_1f = INDEX["SC_rgb_rle.dcm"]["ds"]
u8_3s_2f = INDEX["SC_rgb_rle_2frame.dcm"]["ds"]
i16_1s_1f = INDEX["MR_small_RLE.dcm"]["ds"]
u16_1s_10f = INDEX["emri_small_RLE.dcm"]["ds"]
u16_3s_1f = INDEX["SC_rgb_rle_16bit.dcm"]["ds"]
u16_3s_2f = INDEX["SC_rgb_rle_16bit_2frame.dcm"]["ds"]
u32_1s_1f = INDEX["rtdose_rle_1frame.dcm"]["ds"]
u32_1s_2f = INDEX["rtdose_rle.dcm"]["ds"]
u32_3s_1f = INDEX["SC_rgb_rle_32bit.dcm"]["ds"]
u32_3s_2f = INDEX["SC_rgb_rle_32bit_2frame.dcm"]["ds"]


class TimeDecodePixelData_NumpyHandler:
    def setup(self):
        self.no_runs = 1000

    def time_u08_1s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u8_1s_1f, get_pixeldata(u8_1s_1f))

    def time_u08_1s_2f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u8_1s_2f, get_pixeldata(u8_1s_2f))

    def time_u08_3s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u8_3s_1f, get_pixeldata(u8_3s_1f))

    def time_u08_3s_2f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u8_3s_2f, get_pixeldata(u8_3s_2f))

    def time_i16_1s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(i16_1s_1f, get_pixeldata(i16_1s_1f))

    def time_u16_1s_10f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u16_1s_10f, get_pixeldata(u16_1s_10f))

    def time_u16_3s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u16_3s_1f, get_pixeldata(u16_3s_1f))

    def time_u16_3s_2f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u16_3s_2f, get_pixeldata(u16_3s_2f))

    def time_u32_1s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u32_1s_1f, get_pixeldata(u32_1s_1f))

    def time_u32_1s_2f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u32_1s_2f, get_pixeldata(u32_1s_2f))

    def time_u32_3s_1f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u32_3s_1f, get_pixeldata(u32_3s_1f))

    def time_u32_3s_2f(self):
        for _ in range(self.no_runs):
            reshape_pixel_array(u32_3s_2f, get_pixeldata(u32_3s_2f))


class TimeDecodePixelData_PyLJ:
    def setup(self):
        self.no_runs = 1000

    def time_u08_1s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(u8_1s_1f)

    def time_u08_1s_2f(self):
        for _ in range(self.no_runs):
            pixel_array(u8_1s_2f)

    def time_u08_3s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(u8_3s_1f)

    def time_u08_3s_2f(self):
        for _ in range(self.no_runs):
            pixel_array(u8_3s_2f)

    def time_i16_1s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(i16_1s_1f)

    def time_u16_1s_10f(self):
        for _ in range(self.no_runs):
            pixel_array(u16_1s_10f)

    def time_u16_3s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(u16_3s_1f)

    def time_u16_3s_2f(self):
        for _ in range(self.no_runs):
            pixel_array(u16_3s_2f)

    def time_u32_1s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(u32_1s_1f)

    def time_u32_1s_2f(self):
        for _ in range(self.no_runs):
            pixel_array(u32_1s_2f)

    def time_u32_3s_1f(self):
        for _ in range(self.no_runs):
            pixel_array(u32_3s_1f)

    def time_u32_3s_2f(self):
        for _ in range(self.no_runs):
            pixel_array(u32_3s_2f)


if __name__ == "__main__":
    ds = u8_1s_2f
    nr_runs = 1000

    nr_frames = getattr(ds, "NumberOfFrames", 1)
    frame_gen = generate_pixel_data_frame(ds.PixelData, nr_frames)
    frame = next(frame_gen)

    r = ds.Rows
    c = ds.Columns
    n = ds.SamplesPerPixel
    b = ds.BitsAllocated

    print(
        timeit.timeit(
            "_rle_decode_frame(frame, r, c, n, b)", number=nr_runs, globals=globals()
        )
    )

    p = r * c
    print(timeit.timeit("decode_frame(frame, p, b)", number=nr_runs, globals=globals()))
