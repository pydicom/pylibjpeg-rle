// https://pyo3.rs/v0.13.2/conversions/tables.html
// bytes -> &[u8] or Vec<u8>
// bytearray -> Vec<u8>
// list[T] -> Vec<T>

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyBytes;


fn u32_le(b: &[u8; 4]) -> u32 {
    /* Return 4 u8s as a little-endian ordered unsigned long */
    ((b[0] as u32) <<  0) | ((b[1] as u32) <<  8) |
    ((b[2] as u32) << 16) | ((b[3] as u32) << 24)
}


fn parse_header(data: &[u8]) -> Vec<u32> {
    /* Return the frame offsets from the RLE header

    Parameters
    ----------
    data : bytes | &[u8]
        The 64 byte RLE header.

    Returns
    -------
    List[int] | Vec<u32>
        The non-zero frame offsets. Returns an empty list if the header
        doesn't contain 64 bytes or if the number of segments is 0.
    */
    let mut frame_offsets: Vec<u32> = Vec::new();

    // Check we have 64 bytes available
    if data.len() != 64 {
        return frame_offsets
    }

    let nr_segments: u32 = u32_le(&[data[0], data[1], data[2], data[3]]);

    if nr_segments <= 0 {
        return frame_offsets
    }

    for idx in (4..64).step_by(4) {
        let offset = u32_le(
            &[
                data[idx + 0],
                data[idx + 1],
                data[idx + 2],
                data[idx + 3],
            ]
        );
        if offset > 0 {
            frame_offsets.push(offset);
        }
    }

    frame_offsets
}


#[pyfunction]
fn decode_frame<'a>(
        enc: &[u8], px_per_sample: u32, bits_per_px: u8, py: Python<'a>
) -> PyResult<&'a PyBytes> {
    /* Return the decoded frame.

    Parameters
    ----------
    enc : bytes | &[u8]
        The RLE encoded frame.
    px_per_sample : int | u32
        The number of pixels per sample (rows x columns).
    bits_per_px : int | u8
        The number of bits per pixel, should be a multiple of 8.

    Returns
    -------
    bytearray
        The decoded frame.

    Raises
    ------
    */

    //let header: &[u8] = ;
    let offsets = parse_header(&enc[0..64]);
    let nr_segments = offsets.len();
    println!("Offsets {}", offsets[0]);
    println!("Segments {}", nr_segments);

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
    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0;
    let mut header_byte: u8;
    let mut nr_copies: u16;


    let result = PyBytes::new(py, &out[..]);
    Ok(result)
}


#[pyfunction]
fn decode_segment<'a>(enc: &[u8], py: Python<'a>) -> PyResult<&'a PyBytes> {
    /* Return a decoded RLE segment as bytes.

    Parameters
    ----------
    enc : bytes | &[u8]
        The encoded segment.

    Returns
    -------
    bytes
        The decoded segment.
    */
    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0;
    let mut header_byte: u8;
    let mut nr_copies: u16;

    loop {
        // header_byte is N
        header_byte = enc[pos];
        pos += 1;
        if header_byte > 128 {
            // Extend by copying the next byte (-N + 1) times
            // however since using uint8 instead of int8 this will be
            // (256 - N + 1) times
            nr_copies = 257 - u16::from(header_byte);
            for _ in 0..nr_copies {
                out.push(enc[pos]);
            }
            pos += 1;
        } else if header_byte < 128 {
            // Extend by literally copying the next (N + 1) bytes
            for _ in 0..(header_byte + 1) {
                out.push(enc[pos]);
                pos += 1;
            }
        }

        if pos >= (enc.len() - 1) {
            break;
        }
    }

    let result = PyBytes::new(py, &out[..]);
    Ok(result)
}


#[pymodule]
fn _rle(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_segment, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_frame, m)?).unwrap();

    Ok(())
}
