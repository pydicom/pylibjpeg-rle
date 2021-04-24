
use std::error::Error;
use std::convert::TryFrom;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyBytes, PyByteArray};
use pyo3::exceptions::{PyValueError};


// RLE Decoding

#[pyfunction]
fn parse_header(enc: &[u8]) -> PyResult<Vec<u32>> {
    /* Return the segment offsets from the RLE header.

    Parameters
    ----------
    b : bytes
        The 64 byte RLE header.

    Returns
    -------
    List[int]
        All 15 segment offsets found in the header.
    */
    if enc.len() != 64 {
        return Err(PyValueError::new_err("The RLE header must be 64 bytes long"))
    }

    let header = <&[u8; 64]>::try_from(&enc[0..64]).unwrap();
    let mut offsets: Vec<u32> = Vec::new();
    offsets.extend(&_parse_header(header)[..]);

    Ok(offsets)
}


fn _parse_header(b: &[u8; 64]) -> [u32; 15] {
    /* Return the segment offsets from the RLE header.

    Parameters
    ----------
    b
        The 64 byte RLE header.
    */
    return [
        u32::from_le_bytes([ b[4],  b[5],  b[6],  b[7]]),
        u32::from_le_bytes([ b[8],  b[9], b[10], b[11]]),
        u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        u32::from_le_bytes([b[16], b[17], b[18], b[19]]),
        u32::from_le_bytes([b[20], b[21], b[22], b[23]]),
        u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        u32::from_le_bytes([b[28], b[29], b[30], b[31]]),
        u32::from_le_bytes([b[32], b[33], b[34], b[35]]),
        u32::from_le_bytes([b[36], b[37], b[38], b[39]]),
        u32::from_le_bytes([b[40], b[41], b[42], b[43]]),
        u32::from_le_bytes([b[44], b[45], b[46], b[47]]),
        u32::from_le_bytes([b[48], b[49], b[50], b[51]]),
        u32::from_le_bytes([b[52], b[53], b[54], b[55]]),
        u32::from_le_bytes([b[56], b[57], b[58], b[59]]),
        u32::from_le_bytes([b[60], b[61], b[62], b[63]])
    ]
}


#[pyfunction]
fn decode_frame<'a>(
    enc: &[u8], px_per_sample: u32, bits_per_px: u8, py: Python<'a>
) -> PyResult<&'a PyByteArray> {
    /* Return the decoded frame.

    Parameters
    ----------
    enc : bytes
        The RLE encoded frame.
    px_per_sample : int
        The number of pixels per sample (rows x columns).
    bits_per_px : int
        The number of bits per pixel, should be a multiple of 8.

    Returns
    -------
    bytearray
        The decoded frame.
    */
    match _decode_frame(enc, px_per_sample, bits_per_px) {
        Ok(frame) => return Ok(PyByteArray::new(py, &frame)),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _decode_frame(
    enc: &[u8], px_per_sample: u32, bits_per_px: u8
) -> Result<Vec<u8>, Box<dyn Error>> {
    /* Return the decoded frame.

    Parameters
    ----------
    enc
        The RLE encoded frame.
    px_per_sample
        The number of pixels per sample (rows x columns), maximum (2^32 - 1).
    bits_per_px
        The number of bits per pixel, should be a multiple of 8 and no larger
        than 64.
    */

    // Pre-define our errors for neatness
    let err_bits_zero = Err(
        String::from(
            "The (0028,0010) 'Bits Allocated' value must be greater than 0"
        ).into(),
    );
    let err_bits_not_octal = Err(
        String::from(
            "The (0028,0010) 'Bits Allocated' value must be a multiple of 8"
        ).into(),
    );
    let err_invalid_bytes = Err(
        String::from(
            "A (0028,0010) 'Bits Allocated' value greater than 64 is not supported"
        ).into()
    );
    let err_invalid_offset = Err(
        String::from("Invalid segment offset found in the RLE header").into()
    );
    let err_insufficient_data = Err(
        String::from("Frame is not long enough to contain RLE encoded data").into()
    );
    let err_invalid_nr_samples = Err(
        String::from("The (0028,0002) 'Samples Per Pixel' must be 1 or 3").into()
    );
    let err_segment_length = Err(
        String::from(
            "The decoded segment length does not match the expected length"
        ).into()
    );

    // Ensure we have a valid bits/px value
    match bits_per_px {
        0 => return err_bits_zero,
        _ => match bits_per_px % 8 {
            0 => {},
            _ => return err_bits_not_octal
        }
    }

    // Ensure `bytes_per_pixel` is in [1, 8]
    let bytes_per_pixel: u8 = bits_per_px / 8;
    if bytes_per_pixel > 8 { return err_invalid_bytes }

    // Parse the RLE header and check results
    // --------------------------------------
    // Ensure we have at least enough data for the RLE header
    let encoded_length = enc.len();
    if encoded_length < 64 { return err_insufficient_data }

    let header = <&[u8; 64]>::try_from(&enc[0..64]).unwrap();
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
    let samples_per_px: u8 = nr_segments / bytes_per_pixel;
    match samples_per_px {
        1 => {},
        3 => {},
        _ => return err_invalid_nr_samples
    }

    // Watch for overflow here; u32 * u32 -> u64
    let expected_length = usize::try_from(
        px_per_sample * u32::from(bytes_per_pixel * samples_per_px)
    ).unwrap();

    // Pre-allocate a vector for the decoded frame
    let mut frame = vec![0u8; expected_length];

    /*
    Example
    -------
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
    let pps = usize::try_from(px_per_sample).unwrap();
    // Concatenate sample planes into a frame
    for sample in 0..samples_per_px {  // 0 or (0, 1, 2)
        // Sample offset
        let so = usize::from(sample * bytes_per_pixel) * pps;

        // Interleave the segments into a sample plane
        for byte_offset in 0..bytes_per_pixel {  // 0, [1, 2, 3, ..., 7]
            // idx should be in range [0, 23], but max is 15
            let idx = usize::from(sample * bytes_per_pixel + byte_offset);

            // offsets[idx] is u32 -> usize not guaranteed
            let start = usize::try_from(offsets[idx]).unwrap();
            let end = usize::try_from(offsets[idx + 1]).unwrap();

            // Decode the segment into the frame
            let len = _decode_segment_into_frame(
                <&[u8]>::try_from(&enc[start..end]).unwrap(),
                &mut frame,
                usize::from(bytes_per_pixel),
                usize::from(byte_offset) + so
            )?;
            if len != pps { return err_segment_length }
        }
    }

    Ok(frame)
}


fn _decode_segment_into_frame(
    enc: &[u8], frame: &mut Vec<u8>, bpp: usize, initial_offset: usize
) -> Result<usize, Box<dyn Error>> {
    /* Decode an RLE segment directly into a frame.

    Parameters
    ----------
    enc
        The encoded segment.
    frame
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
    let max_offset = enc.len() - 1;
    let max_frame = frame.len();
    let mut op_len: usize;
    let err_eod = Err(
        String::from(
            "The end of the data was reached before the segment was \
            completely decoded"
        ).into()
    );
    let err_eof = Err(
        String::from(
            "The end of the frame was reached before the segment was \
            completely decoded"
        ).into()
    );

    loop {
        // `header_byte` is equivalent to N in the DICOM Standard
        // usize is at least u8
        header_byte = usize::from(enc[pos]);
        pos += 1;
        if header_byte > 128 {
            // Extend by copying the next byte (-N + 1) times
            // however since using uint8 instead of int8 this will be
            // (256 - N + 1) times
            op_len = 257 - header_byte;
            // Check we have enough encoded data and remaining frame
            if (pos > max_offset) || (idx + op_len) > max_frame {
                match pos > max_offset {
                    true => return err_eod,
                    false => return err_eof
                }
            }

            for _ in 0..op_len {
                frame[idx] = enc[pos];
                idx += bpp;
            }
            pos += 1;
        } else if header_byte < 128 {
            // Extend by literally copying the next (N + 1) bytes
            op_len = header_byte + 1;
            // Check we have enough encoded data and remaining frame
            if ((pos + header_byte) > max_offset) || (idx + op_len > max_frame) {
                match (pos + header_byte) > max_offset {
                    true => return err_eod,
                    false => return err_eof
                }
            }

            for ii in pos..pos + op_len {
                frame[idx] = enc[ii];
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
fn decode_segment<'a>(enc: &[u8], py: Python<'a>) -> PyResult<&'a PyBytes> {
    /* Return a decoded RLE segment as bytes.

    Parameters
    ----------
    enc : bytes
        The encoded segment.

    Returns
    -------
    bytes
        The decoded segment.
    */
    let mut segment: Vec<u8> = Vec::new();
    match _decode_segment(enc, &mut segment) {
        Ok(()) => return Ok(PyBytes::new(py, &segment[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _decode_segment(enc: &[u8], out: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
    /* Decode an RLE segment.

    Parameters
    ----------
    enc
        The encoded segment.
    out
        A Vec<u8> for the decoded segment.
    */
    let mut pos = 0;
    let mut header_byte: usize;
    let max_offset = enc.len() - 1;
    let err = Err(
        String::from(
            "The end of the data was reached before the segment was \
            completely decoded"
        ).into()
    );

    loop {
        // `header_byte` is equivalent to N in the DICOM Standard
        // usize is at least u8
        header_byte = usize::from(enc[pos]);
        pos += 1;
        if header_byte > 128 {
            if pos > max_offset {
                return err
            }
            // Extend by copying the next byte (-N + 1) times
            // however since using uint8 instead of int8 this will be
            // (256 - N + 1) times
            out.extend(vec![enc[pos]; 257 - header_byte]);
            pos += 1;
        } else if header_byte < 128 {
            if (pos + header_byte) > max_offset {
                return err
            }
            // Extend by literally copying the next (N + 1) bytes
            out.extend(&enc[pos..(pos + header_byte + 1)]);
            pos += header_byte + 1;
        } // header_byte == 128 is noop

        if pos >= max_offset {
            return Ok(())
        }
    }
}


// RLE Encoding
#[pyfunction]
fn encode_frame() {}


fn _encode_segment() -> Result<(), Box<dyn Error>> {
    // Each segment must be even length or padded to even length with noop byte
    Ok(())
}


#[pyfunction]
fn encode_row<'a>(src: &[u8], py: Python<'a>) -> PyResult<&'a PyBytes> {
    /* Return RLE encoded data as Python bytes.

    Parameters
    ----------
    src : bytes
        The raw data to be encoded.

    Returns
    -------
    bytes
        The RLE encoded data.
    */
    // Be nice to make use of threading for row encoding

    // Assuming all literal runs, `dst` can never be greater than
    // ceil(src.len() / 128) + src.len()

    // Close enough...
    let mut dst: Vec<u8> = Vec::with_capacity(raw.len() + raw.len() / 128 + 1);
    match _decode_segment(src, &mut dst) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_row(src: &[u8], dst: Vec<u8>) -> Result<(), Box<dyn Error>> {
    /* RLE encode `src` into `dst`

    Parameters
    ----------
    src
        The data to be encoded, must contain at least 2 items.
    dst
        The destination for the encoded data.
    */
    let err_short_src = Err(
        String::from(
            "RLE encoding requires a segment row length of at least 2 bytes"
        ).into(),
    );

    // Check we have the minimum required amount of data
    if src.len() < 2 { return err_short_src }

    // Reminders:
    // * Each image row is encoded separately
    // * Literal runs are a non-repetitive stream
    // * Replicate runs are a repetitive stream
    // * 3-byte repeats shall be encoded as replicate runs
    // * Maximum length of literal/replicate runs is 128 bytes

    // Replicate run: dst += [count, value]
    //   count: number of bytes in the run (i8 = -replicate + 1)
    //   value: the value of the repeating byte

    // Literal run: dst += [count, a, b, c, ...]
    //   count: number of bytes in the literal stream (i8 = literal - 1)
    //   a, b, c, ...: the literal stream

    // Maximum length of a literal/replicate run
    let MAX_RUN: u8 = 128;
    // Track how long the current literal run is
    // TODO: Check if math is more efficient for i8
    let mut literal: usize = 0;
    // Track how long the current replicate run is
    let mut replicate: usize = 0;

    let mut ii: usize = 0;
    let mut previous: u8 = *src[ii];

    loop {
        current = *src[ii + 1];

        println!(
            "ii: {}, prev: {}, cur: {}, l: {}, r: {}",
            ii, previous, current, literal, replicate
        );

        // Run type switching/control
        // --------------------------
        if current == previous {
            // If in a literal run then write and reset to replicate
            if literal > 0 {  // Should be a run of at least 2?
                // Write out and reset literal runs
                dst.push(literal - 1);
                // Indexing here is probably wrong
                // Write from start of literal run to previous item
                dst.push(&src[(ii - literal)..(ii))]);
                literal = 0;
                // *switch to replicate run*
            }
            // If switching to replicate run this is `current` item
            replicate += 1;
        } else {
            // If in a replicate run then write and reset to literal
            if replicate > 0 {
                // Write out and reset
                dst.push(257 - replicate);
                dst.push(previous);
                replicate = 0;
                // *switch to literal run*
            }
            // If switching to literal run this is `current` item
            literal += 1;
        }

        // OK, to reach here run length must be > 0 (obviously)

        // If the run length is maxed, write out and reset
        if replicate == MAX_RUN {  // Should be more frequent
            // Write out replicate run and reset
            dst.push(129);
            dst.push(previous);  // previous or current?
            replicate = 0;
        } else if literal == MAX_RUN {
            // Write out literal run and reset
            dst.push(127);
            dst.extend(&src[a..b]);  // something like that
            literal = 0;
        } // 128 is noop!

        // At this point 0 < run length < MAX_RUN, so loop
        // --------------------------------------

        ii += 1; // fixme?
        if ii = MAX_SRC { break; }

        previous = current;
    }

    // No more source data, finish up the last write operation
    if replicate > 0 {
        // Write out and return
        dst.push(257 - replicate);
        dst.push(previous);
    } else if literal > 0 {
        // Write out and return
        dst.push(literal - 1);
        dst.push(&src[(MAX_SRC - literal)..MAX_SRC)]);
    }

    Ok(())
}


#[pymodule]
fn _rle(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_header, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_segment, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_frame, m)?).unwrap();

    m.add_function(wrap_pyfunction!(encode_row, m)?).unwrap();
    m.add_function(wrap_pyfunction!(encode_frame, m)?).unwrap();

    Ok(())
}
