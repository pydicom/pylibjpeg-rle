
use std::convert::TryFrom;
use std::error::Error;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyBytes, PyByteArray};
use pyo3::exceptions::{PyValueError};


// Python rle module members
#[pymodule]
fn rle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_header, m)?);
    m.add_function(wrap_pyfunction!(decode_segment, m)?);
    m.add_function(wrap_pyfunction!(decode_frame, m)?);

    m.add_function(wrap_pyfunction!(encode_row, m)?);
    m.add_function(wrap_pyfunction!(encode_segment, m)?);
    m.add_function(wrap_pyfunction!(encode_frame, m)?);

    m.add_function(wrap_pyfunction!(pack_bits, m)?);
    m.add_function(wrap_pyfunction!(unpack_bits, m)?);

    Ok(())
}


// Utilities
// ---------

#[pyfunction]
fn pack_bits<'py>(
    src: Vec<u8>, bitorder: char, py: Python<'py>
) -> PyResult<Bound<'py, PyByteArray>> {

    match bitorder {
        '>' | '<' => {},
        _ => { return Err(PyValueError::new_err("'bitorder' must be '>' or '<'")) }
    }

    // Check values are in (0, 1)
    if src.iter().max() > Some(&1u8) {
        return Err(
            PyValueError::new_err("Only binary input (containing zeros or ones) can be packed")
        )
    }

    let mut dst: Vec<u8> = Vec::new();

    if bitorder == '<' {
        // Bits use little endian ordering
        for chunk in src.chunks_exact(8) {
            let mut packed = 0u8;
            for idx in 0..8 {
                packed |= chunk[idx] << idx;
            }
            dst.push(packed);
        }
    } else {
        // Bits use big endian ordering
        for chunk in src.chunks_exact(8) {
            let mut packed = 0u8;
            for idx in 0..8 {
                packed |= chunk[idx] << (7 - idx);
            }
            dst.push(packed);
        }
    }

    let remainder = src.len() % 8;
    if remainder > 0 {
        let mut last_byte = 0u8;
        // ..., 0, 1, 0, 0 iterates as 0, 0, 1, 0
        if bitorder == '<' {
            for (idx, bit) in src.iter().rev().take(remainder).enumerate() {
                // 0 0 0 0 0 0 1 0
                last_byte |= bit << (remainder - idx - 1);
            }
        } else {
            for (idx, bit) in src.iter().rev().take(remainder).enumerate() {
                // 0 1 0 0 0 0 0 0
                last_byte |= bit << (8 - remainder - idx);
            }
        }
        dst.push(last_byte);
    }

    Ok(PyByteArray::new(py, &dst))
}


#[pyfunction]
fn unpack_bits<'py>(
    src: Vec<u8>, count: u128, bitorder: char, py: Python<'py>
) -> PyResult<Bound<'py, PyByteArray>> {
    match bitorder {
        '>' | '<' => {},
        _ => { return Err(PyValueError::new_err("'bitorder' must be '>' or '<'")) }
    }

    // The maximum value of `count` should be 2^65
    let nr_bits: u128;
    let nr_bytes = u128::try_from(src.len()).unwrap();
    if count == 0 || count > nr_bytes * 8 {
        nr_bits = nr_bytes * 8;
    } else {
        nr_bits = count;
    }

    let mut dst: Vec<u8> = Vec::new();
    // Shouldn't be more than 2^64
    let nr_whole_bytes = usize::try_from(nr_bits / 8).unwrap();
    // Shouldn't be more than 7
    let nr_remainder_bits = usize::try_from(nr_bits % 8).unwrap();

    if bitorder == '<' {
        // Unpack the whole bytes
        for offset in  0..nr_whole_bytes {
            for idx in 0..8 {
                dst.push((src[offset] >> idx) & 1u8);
            }
        }
        // Do the final (partial) byte, if required
        if nr_remainder_bits != 0 {
            for idx in 0..nr_remainder_bits {
                dst.push((src[nr_whole_bytes] >> idx) & 1u8);
            }
        }
    } else {
        for offset in  0..nr_whole_bytes {
            for idx in 0..8 {
                dst.push((src[offset] >> (7 - idx)) & 1u8);
            }
        }
        if nr_remainder_bits != 0 {
            for idx in 0..nr_remainder_bits {
                dst.push((src[nr_whole_bytes] >> (7 - idx)) & 1u8);
            }
        }

    }

    Ok(PyByteArray::new(py, &dst))
}


// RLE Decoding
// ------------

#[pyfunction]
fn parse_header(src: &[u8]) -> PyResult<Vec<u32>> {
    /* Return the segment offsets from the RLE header.

    Parameters
    ----------
    src : bytes
        The 64 byte RLE header.

    Returns
    -------
    List[int]
        All 15 segment offsets found in the header.
    */
    if src.len() != 64 {
        return Err(PyValueError::new_err("The RLE header must be 64 bytes long"))
    }

    let header = <&[u8; 64]>::try_from(&src[0..64]).unwrap();
    let mut offsets: Vec<u32> = Vec::new();
    offsets.extend(&_parse_header(header)[..]);

    Ok(offsets)
}


fn _parse_header(src: &[u8; 64]) -> [u32; 15] {
    /* Return the segment offsets from the RLE header.

    Parameters
    ----------
    src
        The 64 byte RLE header containing 15 little-endian ordered offset values.
    */
    return [
        u32::from_le_bytes([ src[4],  src[5],  src[6],  src[7]]),
        u32::from_le_bytes([ src[8],  src[9], src[10], src[11]]),
        u32::from_le_bytes([src[12], src[13], src[14], src[15]]),
        u32::from_le_bytes([src[16], src[17], src[18], src[19]]),
        u32::from_le_bytes([src[20], src[21], src[22], src[23]]),
        u32::from_le_bytes([src[24], src[25], src[26], src[27]]),
        u32::from_le_bytes([src[28], src[29], src[30], src[31]]),
        u32::from_le_bytes([src[32], src[33], src[34], src[35]]),
        u32::from_le_bytes([src[36], src[37], src[38], src[39]]),
        u32::from_le_bytes([src[40], src[41], src[42], src[43]]),
        u32::from_le_bytes([src[44], src[45], src[46], src[47]]),
        u32::from_le_bytes([src[48], src[49], src[50], src[51]]),
        u32::from_le_bytes([src[52], src[53], src[54], src[55]]),
        u32::from_le_bytes([src[56], src[57], src[58], src[59]]),
        u32::from_le_bytes([src[60], src[61], src[62], src[63]])
    ]
}


#[pyfunction]
fn decode_frame<'py>(
    src: &[u8], nr_pixels: u32, bpp: u8, byteorder: char, py: Python<'py>
) -> PyResult<Bound<'py, PyByteArray>> {
    /* Return the decoded frame.

    Parameters
    ----------
    src : bytes
        The RLE encoded frame.
    nr_pixels : int
        The total number of pixels in the frame (rows x columns),
        maximum (2^32 - 1).
    bpp : int
        The number of bits per pixel, supported values 1, 8, 16, 32, 64.
    byteorder : str
        The byte order of the returned data, '<' for little endian, '>' for
        big endian.

    Returns
    -------
    bytearray
        The decoded frame.
    */
    match _decode_frame(src, nr_pixels, bpp, byteorder) {
        Ok(frame) => return Ok(PyByteArray::new(py, &frame)),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _decode_frame(
    src: &[u8], nr_pixels: u32, bpp: u8, byteorder: char
) -> Result<Vec<u8>, Box<dyn Error>> {
    /* Return the decoded frame.

    Parameters
    ----------
    src
        The RLE encoded frame.
    nr_pixels
        The total number of pixels in the frame (rows x columns).
    bpp
        The number of bits per pixel, should be 1 or a multiple of 8 and no larger
        than 64.
    byteorder
        The byte order of the decoded data, '<' for little endian, '>' for
        big endian.
    */

    // Pre-define our errors for neatness
    let err_invalid_bits_allocated = Err(
        String::from(
            "The (0028,0100) 'Bits Allocated' value must be 1, 8, 16, 32 or 64"
        ).into()
    );
    let err_invalid_offset = Err(
        String::from("Invalid segment offset found in the RLE header").into()
    );
    let err_insufficient_data = Err(
        String::from("Frame is not long enough to contain RLE encoded data").into()
    );
    let err_invalid_nr_samples = Err(
        String::from("The (0028,0002) 'Samples per Pixel' must be 1 or 3").into()
    );
    let err_invalid_nr_samples_ba1 = Err(
        String::from(
            "The (0028,0002) 'Samples per Pixel' must be 1 if (0028,0100) 'Bits Allocated' is 1"
        ).into()
    );
    let err_segment_length = Err(
        String::from(
            "The decoded segment length does not match the expected length"
        ).into()
    );
    let err_invalid_byteorder = Err(
        String::from("'byteorder' must be '>' or '<'").into()
    );

    // Check 'Bits Allocated' is 1 or a multiple of 8
    let bytes_per_pixel: u8;  // Valid values are 1 | 2 | 4 | 8
    match bpp {
        0 => return err_invalid_bits_allocated,
        1 => {
                bytes_per_pixel = 1;
        },
        _ => match bpp % 8 {
            0 => {
                bytes_per_pixel = bpp / 8;
            },
            _ => return err_invalid_bits_allocated
        }
    }

    // Check `byteorder` is a valid character
    // Check *Bits Allocated* is in [1, 8, 16, 32, 64]
    match bytes_per_pixel {
        1 => {},
        2 | 4 | 8 => match byteorder {
            '>' | '<' => {},
            _ => return err_invalid_byteorder
        },
        _ => return err_invalid_bits_allocated
    }

    // Parse the RLE header and check results
    // --------------------------------------
    // Ensure we have at least enough data for the RLE header
    let encoded_length = src.len();
    if encoded_length < 64 { return err_insufficient_data }

    // Don't need to check the unwrap as we just checked
    //   there's enough data in `src`
    let header = <&[u8; 64]>::try_from(&src[0..64]).unwrap();
    // All 15 header offsets, however no guarantee they will be non-zero
    let all_offsets: [u32; 15] = _parse_header(header);

    // First offset must always be 64
    if all_offsets[0] != 64 { return err_invalid_offset }

    // Get non-zero offsets and determine the number of segments
    let mut nr_segments: u8 = 0;  // `nr_segments` is in [0, 15]
    let mut offsets: Vec<u32> = Vec::with_capacity(15);
    for val in all_offsets.iter().filter(|&n| *n != 0) {
        offsets.push(*val);
        nr_segments += 1u8;
    }

    // Ensure we have a final ending offset at the end of the data
    offsets.push(u32::try_from(encoded_length).unwrap());

    // Ensure offsets are in increasing order
    let mut last: u32 = 0;
    for val in offsets.iter() {
        if *val <= last {
            return err_invalid_offset
        }
        last = *val;
    }

    // Check the samples per pixel is conformant
    let spp: u8 = nr_segments / bytes_per_pixel;
    match spp {
        1 => {},
        3 => {
            // Bits allocated 1 must be 1 sample per pixel
            match bpp {
                1 => { return err_invalid_nr_samples_ba1 },
                _ => {}
            }
        },
        _ => return err_invalid_nr_samples
    }

    /* Example
    ----------
    RLE encoded data is ordered like this (for 16-bit, 3 sample):
      Segment: 1     | 2     | 3     | 4     | 5     | 6
               R MSB | R LSB | G MSB | G LSB | B MSB | B LSB

    A segment contains only the MSB or LSB parts of all the sample pixels

    To minimise the amount of array manipulation later, and to make things
    faster we interleave each segment in a manner consistent with a planar
    configuration of 1 (and maintain big endian byte ordering):
      All red samples             | All green samples           | All blue
      Pxl 1   Pxl 2   ... Pxl N   | Pxl 1   Pxl 2   ... Pxl N   | ...
      MSB LSB MSB LSB ... MSB LSB | MSB LSB MSB LSB ... MSB LSB | ...
    */

    // Decode each segment and place it into the vector
    // ------------------------------------------------
    // TODO: handle unwrap
    let pps = usize::try_from(nr_pixels).unwrap();
    // Concatenate sample planes into a frame
    // Watch for overflow here; u32 * u32 -> u64
    // Actual values are (u16 * u16) * u8
    let expected_length = usize::try_from(
        nr_pixels * u32::from(nr_segments)
    ).unwrap();

    // Pre-allocate a vector for the decoded frame
    let mut frame = vec![0u8; expected_length];

    for sample in 0..spp {  // 0 or (0, 1, 2)
        // Sample offset
        let so = usize::from(sample * bytes_per_pixel) * pps;

        // Interleave the segments into a sample plane
        for byte_offset in 0..bytes_per_pixel {  // 0, [1, 2, 3, ..., 7]
            // idx should be in range [0, 15]
            let idx: usize;
            if byteorder == '>' { // big-endian
                // e.g. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
                idx = usize::from(sample * bytes_per_pixel + byte_offset);
            } else { // little-endian
                // e.g. 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8
                idx = usize::from(
                    bytes_per_pixel - byte_offset + bytes_per_pixel * sample
                ) - 1;
            }

            // offsets[idx] is u32 -> usize not guaranteed
            let start = usize::try_from(offsets[idx]).unwrap();
            let end = usize::try_from(offsets[idx + 1]).unwrap();

            // Decode the segment into the frame
            let len = _decode_segment_into_frame(
                <&[u8]>::try_from(&src[start..end]).unwrap(),
                &mut frame,
                usize::from(bytes_per_pixel),
                usize::from(byte_offset) + so
            )?;

            if len != pps { return err_segment_length }
        }
    }

    return Ok(frame)
}


fn _decode_segment_into_frame(
    src: &[u8], dst: &mut Vec<u8>, bpp: usize, initial_offset: usize
) -> Result<usize, Box<dyn Error>> {
    /* Decode an RLE segment directly into a frame.

    Parameters
    ----------
    src
        The encoded segment.
    dst
        The Vec<u8> for the decoded frame.
    bpp
        The number of bytes per pixel.
    initial_offset
        The initial frame offset where the first sample value will be placed.

    Returns
    -------
    len
        The length of the decoded segment.
    */
    let mut idx = initial_offset;
    let mut pos = 0;
    let mut header_byte: usize;
    let max_offset = src.len() - 1;
    let max_frame = dst.len();
    let mut op_len: usize;
    let err_eod = Err(
        String::from(
            "The end of the data was reached before the segment was \
            completely decoded"
        ).into()
    );
    // let err_eof = Err(
    //     String::from(
    //         "The end of the frame was reached before the segment was \
    //         completely decoded"
    //     ).into()
    // );

    loop {
        // `header_byte` is equivalent to N in the DICOM Standard
        // usize is at least u8
        header_byte = usize::from(src[pos]);
        pos += 1;
        if header_byte > 128 {
            // Extend by copying the next byte (-N + 1) times
            // however since using uint8 instead of int8 this will be
            // (256 - N + 1) times
            op_len = 257 - header_byte;

            // Check we have enough encoded data
            if pos > max_offset {
                return err_eod
            }

            // Check segment for excess padding
            if (idx + op_len) > max_frame {
                // Only copy until we reach the end of frame
                for _ in 0..(max_frame - idx) {
                    dst[idx] = src[pos];
                    idx += bpp;
                }

                return Ok((idx - initial_offset) / bpp)
            }

            for _ in 0..op_len {
                dst[idx] = src[pos];
                idx += bpp;
            }
            pos += 1;
        } else if header_byte < 128 {
            // Extend by literally copying the next (N + 1) bytes
            op_len = header_byte + 1;

            // Check we have enough encoded data
            if (pos + header_byte) > max_offset {
                return err_eod
            }

            // Check segment for excess padding
            if (idx + op_len) > max_frame {
                // Only extend until the end of frame
                for ii in pos..pos + (max_frame - idx) {
                    dst[idx] = src[ii];
                    idx += bpp;
                }

                return Ok((idx - initial_offset) / bpp)
            }

            for ii in pos..pos + op_len {
                dst[idx] = src[ii];
                idx += bpp;
            }
            pos += header_byte + 1;
        } // header_byte == 128 is noop

        if pos >= max_offset {
            return Ok((idx - initial_offset) / bpp)
        }
    }
}


#[pyfunction]
fn decode_segment<'py>(src: &[u8], py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    /* Return a decoded RLE segment as bytes.

    Parameters
    ----------
    src : bytes
        The encoded segment.

    Returns
    -------
    bytes
        The decoded segment.
    */
    let mut dst: Vec<u8> = Vec::new();
    match _decode_segment(src, &mut dst) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _decode_segment(src: &[u8], dst: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
    /* Decode an RLE segment.

    Parameters
    ----------
    src
        The encoded segment.
    dst
        A Vec<u8> for the decoded segment.
    */
    let mut pos = 0;
    let mut header_byte: usize;
    let max_offset = src.len() - 1;
    let err = Err(
        String::from(
            "The end of the data was reached before the segment was \
            completely decoded"
        ).into()
    );

    loop {
        // `header_byte` is equivalent to N in the DICOM Standard
        // usize is at least u8
        header_byte = usize::from(src[pos]);
        pos += 1;
        if header_byte > 128 {
            if pos > max_offset {
                return err
            }
            // Extend by copying the next byte (-N + 1) times
            // however since using uint8 instead of int8 this will be
            // (256 - N + 1) times
            dst.extend(vec![src[pos]; 257 - header_byte]);
            pos += 1;
        } else if header_byte < 128 {
            if (pos + header_byte) > max_offset {
                return err
            }
            // Extend by literally copying the next (N + 1) bytes
            dst.extend(&src[pos..(pos + header_byte + 1)]);
            pos += header_byte + 1;
        } // header_byte == 128 is noop

        if pos >= max_offset {
            return Ok(())
        }
    }
}


// RLE Encoding
// ------------

#[pyfunction]
fn encode_frame<'py>(
    src: Vec<u8>, rows: u16, cols: u16, spp: u8, bpp: u8, byteorder: char, py: Python<'py>
) -> PyResult<Bound<'py, PyBytes>> {
    /* Return RLE encoded `src` as bytes.

    Parameters
    ----------
    src : bytes
        The data to be RLE encoded, ordered as R1, G1, B1, R2, G2, B2, ...,
        Rn, Gn, Bn (i.e. Planar Configuration 0).
    rows : int
        The number of rows in the data.
    cols : int
        The number of columns in the data.
    spp : int
        The number of samples per pixel, supported values are 1 or 3.
    bpp : int
        The number of bits per pixel, supported values are 1, 8, 16, 32 and 64.
    byteorder : str
        Required if `bpp` is greater than 1, '>' if `src` is in big endian byte
        order, '<' if little endian.

    Returns
    -------
    bytes
        The RLE encoded frame.
    */
    let mut dst: Vec<u8> = Vec::new();
    match _encode_frame(src, &mut dst, rows, cols, spp, bpp, byteorder) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_frame(
    src: Vec<u8>, dst: &mut Vec<u8>, rows: u16, cols: u16, spp: u8, bpp: u8, byteorder: char
) -> Result<(), Box<dyn Error>> {
    /*

    Parameters
    ----------
    src
        The data to be RLE encoded, with multi-sample data ordered as R1, G1, B1,
        R2, G2, B2, ..., Rn, Gn, Bn (i.e. Planar Configuration 0).
    dst
        The vector storing the encoded data.
    rows
        The number of rows in the data.
    cols
        The number of columns in the data.
    spp
        The number of samples per pixel, supported values are 1 or 3. May only be 1 if `bpp`
        is 1.
    bpp
        The number of bits per pixel, supported values are 1, 8, 16, 32 and 64.
    byteorder
        Required if bpp is greater than 1, '>' if `src` is in big endian byte
        order, '<' if little endian.
    */

    // Pre-define our errors for neatness
    let err_invalid_nr_samples = Err(
        String::from("The (0028,0002) 'Samples per Pixel' must be 1 or 3").into()
    );
    let err_invalid_nr_samples_ba1 = Err(
        String::from(
            "The (0028,0002) 'Samples per Pixel' must be 1 if (0028,0100) 'Bits Allocated' is 1"
        ).into()
    );
    let err_invalid_bits_allocated = Err(
        String::from(
            "The (0028,0100) 'Bits Allocated' value must be 1, 8, 16, 32 or 64"
        ).into()
    );
    let err_invalid_nr_segments = Err(
        String::from(
            "Unable to encode as the DICOM Standard only allows \
            a maximum of 15 segments in RLE encoded data"
        ).into()
    );
    let err_invalid_parameters = Err(
        String::from(
            "The length of the data to be encoded is not consistent with the \
            the values of the dataset's 'Rows', 'Columns', 'Samples per Pixel' \
            and 'Bits Allocated' elements"

        ).into()
    );
    let err_invalid_byteorder = Err(
        String::from("'byteorder' must be '>' or '<'").into()
    );

    // Check 'Samples per Pixel' is either 1 or 3
    // Check 'Bits Allocated' is 1 or a multiple of 8
    // Check 'Samples per Pixel' is 1 if 'Bits Allocated' is 1
    let bytes_per_pixel: u8;
    match spp {
        1 => {
            match bpp {
                1 => { bytes_per_pixel = 1; },
                8 | 16 | 32 | 64 => { bytes_per_pixel = bpp / 8; },
                _ => { return err_invalid_bits_allocated }
            }
        }
        3 => {
            match bpp {
                1 => { return err_invalid_nr_samples_ba1 },
                8 | 16 | 32 | 64 => { bytes_per_pixel = bpp / 8; },
                _ => { return err_invalid_bits_allocated }
            }
        },
        _ => return err_invalid_nr_samples
    }

    // Check `byteorder` is a valid character
    // Check *Bits Allocated* is in [1, 8, 16, 32, 64]
    match bytes_per_pixel {
        1 => {},
        2 | 4 | 8 => match byteorder {
            '>' | '<' => {},
            _ => return err_invalid_byteorder
        },
        _ => return err_invalid_bits_allocated
    }

    // Ensure parameters are consistent
    // TODO: handle unwrap
    let r = usize::try_from(rows).unwrap();
    let c = usize::try_from(cols).unwrap();

    let total_pixels = r * c * usize::from(spp);
    let total_length = total_pixels * usize::from(bytes_per_pixel);
    if src.len() != total_length {
        return err_invalid_parameters
    }

    let nr_segments: u8 = spp * bytes_per_pixel;
    if nr_segments > 15 { return err_invalid_nr_segments }

    // Reserve 64 bytes of `dst` for the RLE header
    // Values in the header are in little endian order
    dst.extend(u32::from(nr_segments).to_le_bytes().to_vec());
    dst.extend([0u8; 60].to_vec());

    // A vector of the start indexes used when segmenting
    // Start with big-endian ordered pixel sample values
    let mut start_indices: Vec<usize> = (0..usize::from(nr_segments)).collect();
    if byteorder != '>' {
        // Typically `src` uses little endian byte ordering
        for idx in 0..spp {
            let s = usize::from(idx * bytes_per_pixel);
            let e = usize::from((idx + 1) * bytes_per_pixel);
            start_indices[s..e].reverse();
        }
    }

    // Encode the data and update the RLE header segment offsets
    // Segments are ordered from most significant byte to least significant for
    //  multi-byte values
    for idx in 0..usize::from(nr_segments) {
        // Update RLE header: convert current offset to 4x le ordered u8s
        let current_offset = (u32::try_from(dst.len()).unwrap()).to_le_bytes();
        for ii in idx * 4 + 4..idx * 4 + 8 {
            dst[ii] = current_offset[ii - idx * 4 - 4];
        }

        // Encode! Note the offset start of the `src` iter
        let segment: Vec<u8> = src[start_indices[idx]..]
            .into_iter()
            .step_by(usize::from(spp * bytes_per_pixel))
            .cloned()
            .collect();

        _encode_segment_from_vector(segment, dst, cols)?;
    }

    Ok(())
}


#[pyfunction]
fn encode_segment<'py>(
    src: &[u8], cols: u16, py: Python<'py>
) -> PyResult<Bound<'py, PyBytes>> {
    /* Return an RLE encoded segment as bytes.

    Parameters
    ----------
    src : bytes
        The segment data to be encoded.
    cols : int
        The length of each row in the `src`.

    Returns
    -------
    bytes
        An RLE encoded segment
    */
    let mut dst = Vec::new();
    match _encode_segment_from_array(src, &mut dst, cols) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_segment_from_array(
    src: &[u8], dst: &mut Vec<u8>, cols: u16
) -> Result<(), Box<dyn Error>> {
    /*

    Parameters
    ----------
    src
        The data to be encoded.
    dst
        The destination for the encoded data.
    cols
        The length of each row in the `src`.
    */
    let err_invalid_length = Err(
        String::from("The (0028,0011) 'Columns' value is invalid").into()
    );

    let row_len: usize = usize::try_from(cols).unwrap();

    if src.len() % row_len != 0 { return err_invalid_length }

    let nr_rows = src.len() / row_len;
    let mut offset: usize;

    for row_idx in 0..nr_rows {
        offset = row_idx * row_len;
        _encode_row(&src[offset..offset + row_len], dst)?;
    }

    // Each segment must be even length or padded to even length with zero
    if dst.len() % 2 != 0 {
        dst.push(0);
    }

    Ok(())
}


fn _encode_segment_from_vector(
    src: Vec<u8>, dst: &mut Vec<u8>, cols: u16
) -> Result<(), Box<dyn Error>> {
    /* RLE encode a segment.

    Parameters
    ----------
    src
        The data to be encoded.
    dst
        The destination for the encoded data.
    cols
        The length of each row in the `src`.
    */
    let row_len: usize = usize::try_from(cols).unwrap();
    for row_idx in 0..(src.len() / row_len) {
        let offset = row_idx * row_len;
        _encode_row(&src[offset..offset + row_len], dst)?;
    }

    // Each segment must be even length or padded to even length with zero
    if dst.len() % 2 != 0 { dst.push(0); }

    Ok(())
}


#[pyfunction]
fn encode_row<'py>(src: &[u8], py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    /* Return `src` as RLE encoded bytes.

    Parameters
    ----------
    src : bytes
        The raw data to be encoded.

    Returns
    -------
    bytes
        The RLE encoded data.
    */
    let mut dst = Vec::new();
    match _encode_row(src, &mut dst) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_row(src: &[u8], dst: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
    /* RLE encode `src` into `dst`

    Parameters
    ----------
    src
        The data to be encoded.
    dst
        The destination for the encoded data.
    */

    // Reminders:
    // * Maximum length of literal/replicate runs is 128 bytes
    // * Replicate run: dst += [count, value]
    //     count: number of bytes in the run (i8 = -replicate + 1)
    //     value: the value of the repeating byte
    // * Literal run: dst += [count, a, b, c, ...]
    //     count: number of bytes in the literal stream (i8 = literal - 1)
    //     a, b, c, ...: the literal stream

    match src.len() {
        0 => { return Ok(()) },
        1 => {
            dst.push(0);  // literal run
            dst.push(src[0]);
            return Ok(())
        },
        _ => {}
    }

    let mut literal: Vec<u8> = Vec::new();

    // Chunk the source into groups of identical values
    for group in src.chunk_by(|a, b| a == b) {
        if group.len() == 1 {
            // Only a single value in the group -> add it to the saved literal values
            literal.extend(group);
        } else {
            // Multiple values in the group so one or more replicate runs are required

            //  If `literal` is not empty then add N literal runs to the output first
            if !literal.is_empty() {
                for chunk in literal.chunks(128) {
                    // 1 >= chunk.len() >= 128: usize -> u8
                    dst.push((chunk.len() - 1).try_into().unwrap());
                    dst.extend(chunk);
                }

                // Reset the saved literal run values
                literal.clear();
            }

            // Replicate run(s)
            for chunk in group.chunks(128) {
                if chunk.len() > 1 {
                    // Replicate runs if the chunks have more than 1 value
                    // 1 >= chunk.len() >= 128: usize -> u8
                    dst.push((257 - chunk.len()).try_into().unwrap());
                    dst.push(chunk[0]);
                } else {
                    // If the final chunk is only 1 value long do a literal run instead
                    dst.push(0);
                    dst.push(chunk[0]);
                }
            }
        }
    }

    // Final literal run(s) if literal isn't followed by a replicate run
    if !literal.is_empty() {
        for chunk in literal.chunks(128) {
            // 1 >= chunk.len() >= 128: usize -> u8
            dst.push((chunk.len() - 1).try_into().unwrap());
            dst.extend(chunk);
        }
    }

    Ok(())
}
