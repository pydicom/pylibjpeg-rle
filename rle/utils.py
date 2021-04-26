"""Utility functions."""

import sys
from typing import Generator

import numpy as np

from rle._rle import decode_frame, decode_segment, encode_frame, encode_segment


def decode_pixel_data(stream: bytes, ds: "Dataset") -> "np.ndarray":
    """Return the decoded RLE Lossless data as a :class:`numpy.ndarray`.

    Intended for use with *pydicom* ``Dataset`` objects.

    Parameters
    ----------
    stream : bytes
        The image frame to be decoded.
    ds : pydicom.dataset.Dataset
        A :class:`~pydicom.dataset.Dataset` containing the group ``0x0028``
        elements corresponding to the *Pixel Data*.

    Returns
    -------
    numpy.ndarray
        A 1D array of ``numpy.uint8`` containing the decoded frame data,
        with little-endian encoding and planar configuration 1.

    Raises
    ------
    ValueError
        If the decoding failed.
    """
    return np.frombuffer(
        decode_frame(stream, ds.Rows * ds.Columns, ds.BitsAllocated, '<'),
        dtype='uint8'
    )


def encode_array(arr: "np.ndarray", **kwargs) -> Generator[bytes, None, None]:
    """Yield RLE encoded frames from `arr`.

    Parameters
    ----------
    arr : numpy.ndarray
        The array of data to be RLE encoded
    kwargs : dict
        A dictionary containing keyword arguments. Required keys are either:

        * ``{'ds': pydicom.dataset.Dataset}``, which is the corresponding
          dataset, or
        * ``{'rows': int, 'columns': int, samples_per_px': int,
          'bits_per_px': int, 'nr_frames': int}``.

    Yields
    ------
    bytes
        An RLE encoded frame from `arr`.
    """
    byteorder = arr.dtype.byteorder
    if byteorder == '=':
        byteorder = '<' if sys.byteorder == "little" else '>'

    kwargs['byteorder'] = byteorder

    if 'ds' in kwargs:
        nr_frames = getattr(kwargs['ds'], "NumberOfFrames", 1)
    else:
        nr_frames = kwargs['nr_frames']

    if nr_frames > 1:
        for frame in arr:
            yield encode_pixel_data(frame.tobytes(), **kwargs)
    else:
        yield encode_pixel_data(arr.tobytes(), **kwargs)


def encode_pixel_data(stream: bytes, **kwargs) -> bytes:
    """Return `stream` encoded using the DICOM RLE (PackBits) algorithm.

    .. warning::

        *Samples per Pixel* x *Bits Allocated* must be less than or equal
        to 15 in order to meet the requirements of the *RLE Lossless*
        transfer syntax.

    Parameters
    ----------
    stream : bytes
        The image frame data to be RLE encoded. Only 1 frame can be encoded
        at a time.
    kwargs : dict
        A dictionary containing keyword arguments. Required keys are:

        * ``{'byteorder': str}``, required if the number of samples per
          pixel is greater than 1. If `stream` is in little-endian byte order
          then ``'<'``, otherwise ``'>'`` for big-endian.

        And either:

        * ``{'ds': pydicom.dataset.Dataset}``, which is the dataset
          corresponding to `stream` with matching values for *Rows*, *Columns*,
          *Samples per Pixel* and *Bits Allocated*, or
        * ``{'rows': int, 'columns': int, samples_per_px': int,
          'bits_per_px': int}``.

    Returns
    -------
    bytes
        The RLE encoded frame.
    """
    if 'ds' in kwargs:
        ds = kwargs['ds']
        r, c = ds.Rows, ds.Columns
        bpp = ds.BitsAllocated
        spp = ds.SamplesPerPixel
    else:
        r, c = kwargs['rows'], kwargs['columns']
        bpp = kwargs['bits_per_pixel']
        spp = kwargs['samples_per_px']

    # Validate input
    if bpp not in [8, 16, 32, 64]:
        raise NotImplementedError("'Bits Allocated' must be 8, 16, 32 or 64")

    if spp not in [1, 3]:
        raise ValueError("'Samples per Pixel' must be 1 or 3")

    if bpp / 8 * spp > 15:
        raise ValueError(
            "Unable to encode the data as the DICOM Standard blah blah"
        )

    if len(stream) != (r * c * bpp / 8 * spp):
        raise ValueError(
            "The length of the data doesn't not match"
        )

    return encode_frame(stream, r, c, spp, bpp, kwargs['byteorder'])


def generate_frames(ds: "Dataset", reshape: bool = True) -> "np.ndarray":
    """Yield a *Pixel Data* frame from `ds` as an :class:`~numpy.ndarray`.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing an :dcm:`Image Pixel
        <part03/sect_C.7.6.3.html>` module and the *Pixel Data* to be
        converted.
    reshape : bool, optional
        If ``True`` (default), then the returned :class:`~numpy.ndarray` will
        be reshaped to the correct dimensions. If ``False`` then no reshaping
        will be performed.

    Yields
    -------
    numpy.ndarray
        A single frame of (7FE0,0010) *Pixel Data* as a little-endian ordered
        :class:`~numpy.ndarray` with an appropriate dtype for the data.

    Raises
    ------
    AttributeError
        If `ds` is missing a required element.
    NotImplementedError
        If the dataset's *Transfer Syntax UID* is not *RLE Lossless*.
    """
    import numpy as np

    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.util import pixel_dtype
    from pydicom.uid import RLELossless

    if ds.file_meta.TransferSyntaxUID != RLELossless:
        raise NotImplementedError(
            "Only RLE Lossless encoded pixel data encoded is supported"
        )

    # Check required elements
    required_elements = [
        "BitsAllocated", "Rows", "Columns", "PixelRepresentation",
        "SamplesPerPixel", "PixelData",
    ]
    missing = [elem for elem in required_elements if elem not in ds]
    if missing:
        raise AttributeError(
            "Unable to convert the pixel data as the following required "
            "elements are missing from the dataset: " + ", ".join(missing)
        )

    nr_frames = getattr(ds, "NumberOfFrames", 1)
    r, c = ds.Rows, ds.Columns
    bpp = ds.BitsAllocated

    dtype = pixel_dtype(ds)
    for frame in generate_pixel_data_frame(ds.PixelData, nr_frames):
        arr = np.frombuffer(decode_frame(frame, r * c, bpp, '<'), dtype=dtype)

        if not reshape:
            yield arr
            continue

        if ds.SamplesPerPixel == 1:
            yield arr.reshape(ds.Rows, ds.Columns)
        else:
            # RLE is planar configuration 1
            arr = np.reshape(arr, (ds.SamplesPerPixel, ds.Rows, ds.Columns))
            yield arr.transpose(1, 2, 0)


def pixel_array(ds: "Dataset") -> "np.ndarray":
    """Return the entire *Pixel Data* as an :class:`~numpy.ndarray`.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing an :dcm:`Image Pixel
        <part03/sect_C.7.6.3.html>` module and the *Pixel Data* to be
        converted.

    Returns
    -------
    numpy.ndarray
        The contents of (7FE0,0010) *Pixel Data* as a little-endian ordered
        :class:`~numpy.ndarray` with shape (rows, columns), (rows, columns,
        components), (frames, rows, columns), or (frames, rows, columns,
        components) depending on the dataset.
    """
    from pydicom.pixel_data_handlers.util import (
        get_expected_length, reshape_pixel_array, pixel_dtype
    )

    expected_len = get_expected_length(ds, 'pixels')
    frame_len = expected_len // getattr(ds, "NumberOfFrames", 1)
    # Empty destination array for our decoded pixel data
    arr = np.empty(expected_len, pixel_dtype(ds))

    generate_offsets = range(0, expected_len, frame_len)
    for frame, offset in zip(generate_frames(ds, False), generate_offsets):
        arr[offset:offset + frame_len] = frame

    return reshape_pixel_array(ds, arr)
