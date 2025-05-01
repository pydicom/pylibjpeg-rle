"""Utility functions."""

import enum
import sys
from typing import Iterator, Optional, Any, TYPE_CHECKING, cast, Union

import numpy as np

from rle.rle import decode_frame, decode_segment, encode_frame, encode_segment


if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


class Version(enum.IntEnum):
    v1 = 1
    v2 = 2


def decode_pixel_data(
    src: bytes,
    ds: Optional["Dataset"] = None,
    version: int = Version.v1,
    **kwargs: Any,
) -> Union[np.ndarray, bytearray]:
    """Return the decoded RLE Lossless data as a :class:`numpy.ndarray`.

    Intended for use with *pydicom* ``Dataset`` objects.

    Parameters
    ----------
    src : bytes
        A single encoded image frame to be decoded.
    ds : pydicom.dataset.Dataset, optional
        A :class:`~pydicom.dataset.Dataset` containing the group ``0x0028``
        elements corresponding to the image frame. If not used then `kwargs`
        must be supplied. Only used with version ``1``.
    version : int, optional

        * If ``1`` (default) then return the image data as an :class:`numpy.ndarray`
        * If ``2`` then return the image data as :class:`bytearray`
    **kwargs
        Required keys if `ds` is not supplied or if version is ``2``:

        * ``"rows"``: :class:`int` - the number of rows in the decoded image
        * ``"columns"``: :class:`int` - the number of columns in the decoded
          image
        * ``"bits_allocated"``: :class:`int` - the number of bits allocated
          to each pixel

        Current decoding options are:

        * ``{'byteorder': str}`` specify the byte ordering for the decoded data
        when more than 8 bits per pixel are used, should be '<' for little
        endian ordering (default) or '>' for big-endian ordering.

    Returns
    -------
    bytearray | numpy.ndarray
        The image data as either a bytearray or ndarray.

    Raises
    ------
    ValueError
        If the decoding failed.
    """
    if version == Version.v1:
        byteorder = kwargs.get("byteorder", "<")

        columns = kwargs.get("columns")
        rows = kwargs.get("rows")
        bits_allocated = kwargs.get("bits_allocated")
        no_kwargs = None in (columns, rows, bits_allocated)
        if ds is None and no_kwargs:
            raise ValueError("Either `ds` or `**kwargs` must be used")

        ds = cast("Dataset", ds)

        columns = ds.get("Columns", columns)
        rows = ds.get("Rows", rows)
        bits_allocated = ds.get("BitsAllocated", bits_allocated)

        frame = decode_frame(src, rows * columns, bits_allocated, byteorder)
        return np.frombuffer(frame, dtype="uint8")

    required_keys = ["bits_allocated", "columns", "rows"]
    missing = [k for k in required_keys if k not in kwargs]
    if missing:
        raise AttributeError(
            f"Missing expected keyword arguments: {', '.join(missing)}"
        )

    columns = cast(int, kwargs.get("columns"))
    rows = cast(int, kwargs.get("rows"))
    bits_allocated = cast(int, kwargs.get("bits_allocated"))
    byteorder = kwargs.get("byteorder", "<")

    return cast(bytearray, decode_frame(src, rows * columns, bits_allocated, byteorder))


def encode_array(
    arr: "np.ndarray", ds: Optional["Dataset"] = None, **kwargs: Any
) -> Iterator[bytes]:
    """Yield RLE encoded frames from `arr`.

    .. versionadded:: 1.1

    Parameters
    ----------
    arr : numpy.ndarray
        The array of data to be RLE encoded, should be ordered as (frames,
        rows, columns, planes), (rows, columns, planes), (frames, rows,
        columns) or (rows, columns).
    ds : pydicom.dataset.Dataset, optional
        The dataset corresponding to `arr` with matching values for *Rows*,
        *Columns*, *Samples per Pixel* and *Bits Allocated*. Required if
        the array properties aren't specified using `kwargs`.
    **kwargs
        Required keyword parameters if `ds` isn't used are:

        * ``'rows': int`` the number of rows contained in `src`
        * ``'columns': int`` the number of columns contained in `src`
        * ``samples_per_px': int`` the number of samples per pixel, either
          1 for monochrome or 3 for RGB or similar data.
        * ``'bits_per_px': int`` the number of bits needed to contain each
          pixel, either 8, 16, 32 or 64.
        * ``'nr_frames': int`` the number of frames in `arr`, required if
          more than one frame is present.

    Yields
    ------
    bytes
        An RLE encoded frame from `arr`.
    """
    byteorder = arr.dtype.byteorder
    if byteorder == "=":
        byteorder = "<" if sys.byteorder == "little" else ">"

    kwargs["byteorder"] = byteorder

    if ds:
        kwargs["rows"] = ds.Rows
        kwargs["columns"] = ds.Columns
        kwargs["samples_per_pixel"] = ds.SamplesPerPixel
        kwargs["bits_allocated"] = ds.BitsAllocated
        kwargs["number_of_frames"] = int(getattr(ds, "NumberOfFrames", 1) or 1)

    if kwargs["number_of_frames"] > 1:
        for frame in arr:
            yield encode_pixel_data(frame.tobytes(), **kwargs)
    else:
        yield encode_pixel_data(arr.tobytes(), **kwargs)


def encode_pixel_data(
    src: bytes,
    ds: Optional["Dataset"] = None,
    byteorder: Optional[str] = None,
    **kwargs: Any,
) -> bytes:
    """Return `src` encoded using the DICOM RLE (PackBits) algorithm.

    .. versionadded:: 1.1

    .. warning::

        *Samples per Pixel* x *Bits Allocated* must be less than or equal
        to 15 in order to meet the requirements of the *RLE Lossless*
        transfer syntax.

    Parameters
    ----------
    src : bytes
        The data for a single image frame data to be RLE encoded.
    ds : pydicom.dataset.Dataset, optional
        The dataset corresponding to `src` with matching values for *Rows*,
        *Columns*, *Samples per Pixel* and *Bits Allocated*. Required if
        the frame properties aren't specified using `kwargs`.
    byteorder : str, optional
        Required if the samples per pixel is greater than 1. If `src` is in
        little-endian byte order then ``'<'``, otherwise ``'>'`` for
        big-endian.
    **kwargs
        If `ds` is not used then the following are required:

        * ``'rows': int`` the number of rows contained in `src`
        * ``'columns': int`` the number of columns contained in `src`
        * ``samples_per_pixel': int`` the number of samples per pixel, either
          1 for monochrome or 3 for RGB or similar data.
        * ``'bits_allocated': int`` the number of bits needed to contain each
          pixel, either 8, 16, 32 or 64.

    Returns
    -------
    bytes
        The RLE encoded frame.
    """
    if ds:
        r = ds.Rows
        c = ds.Columns
        bpp = ds.BitsAllocated
        spp = ds.SamplesPerPixel
    else:
        r = kwargs["rows"]
        c = kwargs["columns"]
        bpp = kwargs["bits_allocated"]
        spp = kwargs["samples_per_pixel"]

    # Validate input
    if spp not in [1, 3]:
        msg = "(0028,0002) 'Samples per Pixel'" if ds else "'samples_per_pixel'"
        raise ValueError(f"{msg} must be 1 or 3")

    if bpp not in [8, 16, 32, 64]:
        msg = "(0028,0100) 'Bits Allocated'" if ds else "'bits_allocated'"
        raise ValueError(f"{msg} must be 8, 16, 32 or 64")

    if bpp / 8 * spp > 15:
        raise ValueError(
            "Unable to encode the data as the RLE format used by the DICOM "
            "Standard only allows a maximum of 15 segments"
        )

    byteorder = "<" if bpp == 8 else byteorder
    if byteorder not in ("<", ">"):
        raise ValueError(
            "A valid 'byteorder' is required when the number of bits per "
            "pixel is greater than 8"
        )

    if len(src) != (r * c * bpp / 8 * spp):
        raise ValueError("The length of the data doesn't match the image parameters")

    return cast(bytes, encode_frame(src, r, c, spp, bpp, byteorder))


def generate_frames(ds: "Dataset", reshape: bool = True) -> Iterator[np.ndarray]:
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

    from pydicom.encaps import generate_frames
    from pydicom.pixels.utils import pixel_dtype
    from pydicom.uid import RLELossless

    if ds.file_meta.TransferSyntaxUID != RLELossless:
        raise NotImplementedError(
            "Only RLE Lossless encoded pixel data encoded is supported"
        )

    # Check required elements
    required_elements = [
        "BitsAllocated",
        "Rows",
        "Columns",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PixelData",
    ]
    missing = [elem for elem in required_elements if elem not in ds]
    if missing:
        raise AttributeError(
            "Unable to convert the pixel data as the following required "
            "elements are missing from the dataset: " + ", ".join(missing)
        )

    nr_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    r = ds.Rows
    c = ds.Columns
    bpp = ds.BitsAllocated

    dtype = pixel_dtype(ds)
    for frame in generate_frames(ds.PixelData, number_of_frames=nr_frames):
        arr = np.frombuffer(decode_frame(frame, r * c, bpp, "<"), dtype=dtype)

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
        <part03/sect_C.7.6.3.html>` module and the *RLE Lossless* encoded
        *Pixel Data* to be decoded.

    Returns
    -------
    numpy.ndarray
        The contents of (7FE0,0010) *Pixel Data* as a little-endian ordered
        :class:`~numpy.ndarray` with shape (rows, columns), (rows, columns,
        components), (frames, rows, columns), or (frames, rows, columns,
        components) depending on the dataset.
    """
    from pydicom.pixels.utils import (
        get_expected_length,
        reshape_pixel_array,
        pixel_dtype,
    )

    expected_len = get_expected_length(ds, "pixels")
    frame_len = expected_len // getattr(ds, "NumberOfFrames", 1)
    # Empty destination array for our decoded pixel data
    arr = np.empty(expected_len, pixel_dtype(ds))

    generate_offsets = range(0, expected_len, frame_len)
    for frame, offset in zip(generate_frames(ds, False), generate_offsets):
        arr[offset : offset + frame_len] = frame

    return reshape_pixel_array(ds, arr)


def pixel_data(arr: "np.ndarray", ds: "Dataset") -> bytes:
    """Return `arr` as encapsulated and RLE encoded bytes.

    .. versionadded:: 1.1

    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to be encoded.
    ds : pydicom.dataset.Dataset
        The dataset corresponding to `arr` with matching values for *Rows*,
        *Columns*, *Samples per Pixel* and *Bits Allocated*.

    Returns
    -------
    bytes
        The encapsulated and RLE encoded `arr`, ready to be used to set
        the dataset's *Pixel Data* element.
    """
    from pydicom.encaps import encapsulate

    return encapsulate([ii for ii in encode_array(arr, ds)])
