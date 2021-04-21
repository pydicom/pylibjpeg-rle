from struct import pack
import timeit

import pytest

import numpy as np

from pydicom import dcmread
import pydicom.config
from pydicom.data import get_testdata_file
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.rle_handler import (
    _parse_rle_header, _rle_decode_frame, _rle_decode_segment
)
from pydicom.pixel_data_handlers.util import pixel_dtype, reshape_pixel_array
from pydicom.uid import RLELossless

from ljdata import get_indexed_datasets
from rle._rle import decode_segment, decode_frame, parse_header


if __name__ == "__main__":
    INDEX = get_indexed_datasets('1.2.840.10008.1.2.5')
    #ds = INDEX['OBXXXX1A_rle.dcm']['ds']
    ds = INDEX["SC_rgb_rle_32bit.dcm"]['ds']

    nr_runs = 1000

    frame_gen = generate_pixel_data_frame(ds.PixelData)
    frame = next(frame_gen)

    r = ds.Rows
    c = ds.Columns
    n = ds.SamplesPerPixel
    b = ds.BitsAllocated

    print(
        timeit.timeit(
            "_rle_decode_frame(frame, r, c, n, b)",
            number=nr_runs,
            globals=globals()
        )
    )

    p = r * c
    print(
        timeit.timeit(
            "decode_frame(frame, p, b)",
            number=nr_runs,
            globals=globals()
        )
    )
