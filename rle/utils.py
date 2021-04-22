"""Utility functions."""


import numpy as np

from rle._rle import decode_frame, decode_segment, parse_header


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
        with big-endian encoding and planar configuration 1.

    Raises
    ------
    ValueError
        If the decoding failed.
    """
    return np.frombuffer(
        decode_frame(stream, ds.Rows * ds.Columns, ds.BitsAllocated),
        dtype='uint8'
    )


def generate_frames(
    ds: "Dataset", reshape: bool = True, rle_segment_order: str = '>'
) -> "np.ndarray":
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
    rle_segment_order : str
        The order of segments used by the RLE decoder when dealing with *Bits
        Allocated* > 8. Each RLE segment contains 8-bits of the pixel data,
        and segments are supposed to be ordered from MSB to LSB. A value of
        ``'>'`` means interpret the segments as being in big endian order
        (default) while a value of ``'<'`` means interpret the segments as
        being in little endian order which may be possible if the encoded data
        is non-conformant.

    Yields
    -------
    numpy.ndarray
        A single frame of (7FE0,0010) *Pixel Data* as an
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

    dtype = pixel_dtype(ds).newbyteorder(rle_segment_order)
    for frame in generate_pixel_data_frame(ds.PixelData, nr_frames):
        arr = np.frombuffer(decode_frame(frame, r * c, bpp), dtype=dtype)

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
        The contents of (7FE0,0010) *Pixel Data* as an :class:`~numpy.ndarray`
        with shape (rows, columns), (rows, columns, components), (frames,
        rows, columns), or (frames, rows, columns, components) depending on
        the dataset.
    """
    from pydicom.pixel_data_handlers.util import (
        get_expected_length, reshape_pixel_array, pixel_dtype
    )

    expected_len = get_expected_length(ds, 'pixels')
    frame_len = expected_len // getattr(ds, "NumberOfFrames", 1)
    # Empty destination array for our decoded pixel data
    dtype = pixel_dtype(ds).newbyteorder('>')
    arr = np.empty(expected_len, dtype)

    generate_offsets = range(0, expected_len, frame_len)
    for frame, offset in zip(generate_frames(ds, False), generate_offsets):
        arr[offset:offset + frame_len] = frame

    return reshape_pixel_array(ds, arr)
