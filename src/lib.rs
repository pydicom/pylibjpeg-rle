
use std::convert::TryFrom;
use std::error::Error;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyBytes, PyByteArray};
use pyo3::exceptions::{PyValueError};


//use rayon::prelude::*;

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
        String::from("The (0028,0002) 'Samples per Pixel' must be 1 or 3").into()
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
        1 | 3 => {},
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
fn encode_frame<'a>(
    enc: &[u8], rows: u16, cols: u16, spp: u8, bpp: u8, byteorder: char, py: Python<'a>
) -> PyResult<&'a PyBytes> {
    let mut dst: Vec<u8> = Vec::new();
    match _encode_frame(enc, &mut dst, rows, cols, spp, bpp, byteorder) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_frame(
    src: &[u8], dst: &mut Vec<u8>, rows: u16, cols: u16, spp: u8, bpp: u8, byteorder: char
) -> Result<(), Box<dyn Error>> {
    /*

    Parameters
    ----------
    src
        The data to be RLE encoded.
    dst

    rows
        The number of rows in the data.
    cols
        The number of columns in the data.
    spp
        The number of samples per pixel, supported values are 1 or 3.
    bpp
        The number of bits per pixel, supported values are 8, 16, 32 and 64.
    byteorder
        Required if bpp is greater than 1, '>' if `src` is in big endian byte
        order, '<' if little endian.

    */
    let err_invalid_nr_samples = Err(
        String::from("The (0028,0002) 'Samples per Pixel' must be 1 or 3").into()
    );
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
    let err_invalid_nr_segments = Err(
        String::from(
            "Unable to encode as the DICOM Standard only allows \
            a maximum of 15 segments in RLE encoded data"
        ).into()
    );
    let err_invalid_parameters = Err(
        String::from(
            "The length of the data to be encoded is not consistent with the \
            the values of the dataset's 'Rows', 'Columns', 'Samples per Pixel'  \
            and 'Bits Allocated' elements"

        ).into()
    );
    let err_invalid_byteorder = Err(
        String::from("Invalid byte order, must be '>' or '<'").into()
    );

    // Check Samples per Pixel
    match spp {
        1 | 3 => {},
        _ => return err_invalid_nr_samples
    }

    // Check Bits Allocated and byte ordering
    match bpp {
        0 => return err_bits_zero,
        _ => match bpp % 8 {
            0 => {},
            _ => return err_bits_not_octal
        }
    }
    // Ensure `bytes_per_pixel` is in [1, 8]
    let bytes_per_pixel: u8 = bpp / 8;
    if bytes_per_pixel > 8 { return err_invalid_bytes }

    if bytes_per_pixel > 1 {
        match byteorder {
            '>' | '<' => {},
            _ => return err_invalid_byteorder
        }
    }

    println!(
        "src len {}, r {}, c {}, spp {}, bpp {}, bo {}",
        src.len(), rows, cols, spp, bpp, byteorder
    );

    // Ensure parameters are consistent
    let r = usize::try_from(rows).unwrap();
    let c = usize::try_from(cols).unwrap();

    if src.len() != r * c * usize::from(spp * bytes_per_pixel) {
        return err_invalid_parameters
    }

    let nr_segments: u8 = spp * bytes_per_pixel;
    println!("Nr segs {}", nr_segments);
    if nr_segments > 15 { return err_invalid_nr_segments }

    let is_big_endian: bool = byteorder == '>';
    println!("is big {}", is_big_endian);


    // Big endian byte order '>'
    // spp 3, bpp 2, 100x100: a = MSB, b = LSB
    // R0a-0, R0b-0, G0a-0, G0b-0, B0a-0, B0a-0, ..., B99a-0, B99b-0, R0a-1, ...

    // Little endian byte order '<'
    // spp 3, bpp 2, 100x100: a = MSB, b = LSB
    // R0b-0, R0a-0, G0b-0, G0a-0, B0b-0, B0a-0, ..., B99b-0, B99a-0, R0b-1, ...

    // Need to be big endian u32s
    //let mut header: Vec<u32> = vec![0; 16];
    dst.extend(u32::from(nr_segments).to_le_bytes().to_vec());
    dst.extend([0u8; 60].to_vec());
    println!("header {}", dst.len());
    // Big endian
    let mut start = 0;
    let process_step = usize::from(spp * bytes_per_pixel);

    let mut start: Vec<usize> = (0..usize::from(nr_segments)).collect();
    if !is_big_endian {
        // Shuffle from BE to LE ordering
        // This is ridiculous!
        for ii in 0..usize::from(nr_segments / bytes_per_pixel) {
            println!("{} {}", ii, bpp);
            match bpp {
                8 => {},
                16 => start.swap(ii * 2, ii * 2 + 1),
                32 => {
                    start.swap(ii * 2, ii * 2 + 3);
                    start.swap(ii * 2 + 1, ii * 2 + 2);
                },
                64 => {
                    start.swap(ii * 2, ii * 2 + 7);
                    start.swap(ii * 2 + 1, ii * 2 + 6);
                    start.swap(ii * 2 + 2, ii * 2 + 5);
                    start.swap(ii * 2 + 3, ii * 2 + 4);
                },
                _ => { return err_invalid_bytes }
            }
        }
    }
    println!("starts {:?}", start);

    for idx in 0..usize::from(nr_segments) {

        // Convert to 4 big-endian ordered u8s
        let offset = (u32::try_from(dst.len()).unwrap()).to_le_bytes();
        //println!("{:?}", offset);
        for ii in idx * 4 + 4..idx * 4 + 8 {
            // idx = 0: 4, 5, 6, 7 -> 0, 1, 2, 3
            // idx = 1: 8, 9, 10, 11 -> 0, 1, 2, 3
            dst[ii] = offset[ii - idx * 4 - 4];
        }

        println!("idx {}, ps {}, o {:?}", idx, process_step, offset);
        // Correlate idx to which bytes to process as a segment
        // spp 1, bpp 1 -> nr segments 1 -> process all
        // spp 3, bpp 1 -> nr_segments 3 -> process every third then offset by 1
        // spp 1, bpp 2 -> nr_segments 2 -> process every second then offset by 1
        // spp 3, bpp 2 -> nr_segments 6 -> process every sixth then offset by 1
        // spp 3, bpp 4 -> nr_segments 12 -> process every twelfth then offset by 1

        // Need to reverse order of offset if little endian
        // i.e. 0 -> 1 -> 2 -> 3 becomes 3 -> 2 -> 1 -> 0

        // need to offset src iter start by idx
        let segment: Vec<u8> = src[start[idx]..].into_iter().step_by(process_step).cloned().collect();
        _encode_segment_b(segment, dst, cols);
        //offsets.push(dst.len());
    }
    println!("{:?}", &dst[..64]);
    println!("{:?}", &dst[64..164]);

    Ok(())
}


#[pyfunction]
fn encode_segment<'a>(src: &[u8], cols: u16, py: Python<'a>) -> PyResult<&'a PyBytes> {
    let mut dst = Vec::new();
    match _encode_segment(src, &mut dst, cols) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


fn _encode_segment(
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
    let mut offset: usize = 0;

    for row_idx in 0..nr_rows {
        offset = row_idx * row_len;
        _encode_row(&src[offset..offset + row_len], dst);
    }

    // Each segment must be even length or padded to even length with noop byte
    if dst.len() % 2 != 0 {
        dst.push(128);
    }

    Ok(())
}

fn _encode_segment_b(
    src: Vec<u8>, dst: &mut Vec<u8>, cols: u16
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
    let mut offset: usize = 0;

    for row_idx in 0..nr_rows {
        offset = row_idx * row_len;
        _encode_row(&src[offset..offset + row_len], dst);
    }

    // Each segment must be even length or padded to even length with noop byte
    if dst.len() % 2 != 0 {
        dst.push(128);
    }

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
    // Assuming all literal runs, `dst` can never be greater than
    // ceil(src.len() / 128) + src.len()

    // Close enough...
    let mut dst = Vec::with_capacity(src.len() + src.len() / 128 + 1);
    match _encode_row(src, &mut dst) {
        Ok(()) => return Ok(PyBytes::new(py, &dst[..])),
        Err(err) => return Err(PyValueError::new_err(err.to_string())),
    }
}


#[allow(overflowing_literals)]
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
    // * Each image row is encoded separately
    // * Literal runs are a non-repetitive stream
    // * Replicate runs are a repetitive stream
    // * 2 byte repeats are encoded as replicate runs
    // * Maximum length of literal/replicate runs is 128 bytes

    // Replicate run: dst += [count, value]
    //   count: number of bytes in the run (i8 = -replicate + 1)
    //   value: the value of the repeating byte

    // Literal run: dst += [count, a, b, c, ...]
    //   count: number of bytes in the literal stream (i8 = literal - 1)
    //   a, b, c, ...: the literal stream

    match src.len() {
        0 => { return Ok(()) },
        1 => {
            dst.push(0);  // literal run
            dst.push(src[0]);
            return Ok(())
        },
        _ => {}
    }

    // Maximum length of a literal/replicate run
    let MAX_RUN: u8 = 128;
    let MAX_SRC = src.len();

    // `replicate` and `literal` are the length of the current run
    let mut literal: u8 = 0;
    let mut replicate: u8 = 0;

    let mut previous: u8 = src[0];
    let mut current: u8 = src[1];
    let mut ii: usize = 1;

    // Account for the first item
    if current == previous { replicate = 1; }
    else { literal = 1; }

    loop {
        current = src[ii];

        //println!(
        //    "Start of loop - ii: {}/{}, prv: {}, cur: {}, l: {}, r: {}",
        //    ii, MAX_SRC, previous, current, literal, replicate
        //);

        // Run type switching/control
        if current == previous {
            if literal == 1 {
                // Switch over to a replicate run
                literal = 0;
                replicate = 1;
            } else if literal > 1 {
                // Write out literal run and reset
                dst.push(literal - 1u8);
                // Hmm, check the indexing here...
                dst.extend(&src[ii - usize::from(literal)..ii]);
                literal = 0;
             }
            replicate += 1;

        } else {
            if replicate == 1 {
                // Switch over to a literal run
                literal = 1;
                replicate = 0;
            } else if replicate > 1 {
                // Write out replicate run and reset
                // `replicate` must be at least 2 to avoid overflow
                dst.push(257u8.wrapping_sub(replicate));
                // Note use of previous item
                dst.push(previous);
                replicate = 0;
             }
            literal += 1;
        }

        // If the run length is maxed, write out and reset
        if replicate == MAX_RUN {  // Should be more frequent
            // Write out replicate run and reset
            dst.push(129);
            dst.push(previous);
            replicate = 0;
        } else if literal == MAX_RUN {
            // Write out literal run and reset
            dst.push(127);
            // The indexing is different here because ii is the current item, not previous
            dst.extend(&src[ii + 1 - usize::from(literal)..ii + 1]);
            literal = 0;
        } // 128 is noop!

        previous = current;
        ii += 1;

        if ii == MAX_SRC { break; }
    }

    // Handle cases where the last byte is continuation of a replicate run
    //  such as when 129 0x00 bytes -> replicate(128) + literal(1)
    if replicate == 1 {
        replicate = 0;
        literal = 1;
    }

    if replicate > 1 {
        // Write out and return
        // `replicate` must be at least 2 or we overflow
        dst.push(257u8.wrapping_sub(replicate));
        dst.push(current);
    } else if literal > 0 {
        // Write out and return
        // `literal` must be at least 1 or we undeflow
        dst.push(literal - 1u8);
        dst.extend(&src[MAX_SRC - usize::from(literal)..]);
    }

    Ok(())
}


#[pymodule]
fn _rle(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_header, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_segment, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_frame, m)?).unwrap();

    m.add_function(wrap_pyfunction!(encode_row, m)?).unwrap();
    m.add_function(wrap_pyfunction!(encode_segment, m)?).unwrap();
    m.add_function(wrap_pyfunction!(encode_frame, m)?).unwrap();

    Ok(())
}
