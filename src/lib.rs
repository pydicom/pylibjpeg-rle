// https://pyo3.rs/v0.13.2/conversions/tables.html
// bytes -> &[u8] or Vec<u8>
// bytearray -> Vec<u8>
// list[T] -> Vec<T>

use std::error::Error;
use std::convert::TryFrom;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyBytes, PyByteArray};
use pyo3::exceptions::{PyValueError, PyNotImplementedError};


fn u32_le(b: &[u8; 4]) -> u32 {
    /* Convert 4 u8 to a little endian ordered u32 (unsigned long) */
    (b[0] as u32) |
    ((b[1] as u32) <<  8) |
    ((b[2] as u32) << 16) |
    ((b[3] as u32) << 24)
}


fn parse_header(b: &[u8; 64]) -> [u32; 15] {
    /* Return the segment offsets from the RLE header.

    Parameters
    ----------
    b : &[u8; 64]
        The 64 byte RLE header.

    Returns
    -------
    [u32; 15]
        The segment offsets.
    */
    let offsets = [
        u32_le(&[ b[4],  b[5],  b[6],  b[7]]),
        u32_le(&[ b[8],  b[9], b[10], b[11]]),
        u32_le(&[b[12], b[13], b[14], b[15]]),
        u32_le(&[b[16], b[17], b[18], b[19]]),
        u32_le(&[b[20], b[21], b[22], b[23]]),
        u32_le(&[b[24], b[25], b[26], b[27]]),
        u32_le(&[b[28], b[29], b[30], b[31]]),
        u32_le(&[b[32], b[33], b[34], b[35]]),
        u32_le(&[b[36], b[37], b[38], b[39]]),
        u32_le(&[b[40], b[41], b[42], b[43]]),
        u32_le(&[b[44], b[45], b[46], b[47]]),
        u32_le(&[b[48], b[49], b[50], b[51]]),
        u32_le(&[b[52], b[53], b[54], b[55]]),
        u32_le(&[b[56], b[57], b[58], b[59]]),
        u32_le(&[b[60], b[61], b[62], b[63]])
    ];

    return offsets
}


#[pyfunction]
fn decode_frame<'a>(
    enc: &[u8], px_per_sample: u32, bits_per_px: u8, py: Python<'a>
) -> PyResult<&'a PyByteArray> {
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
    */
    match _decode_frame(enc, px_per_sample, bits_per_px) {
        Ok(frame) => return Ok(PyByteArray::new(py, &frame)),
        Err(e) => {
            return Err(
                PyValueError::new_err(e.to_string())
            )
        }
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
        The number of pixels per sample (rows x columns).
    bits_per_px
        The number of bits per pixel, should be a multiple of 8.
    */
    // Check the number of pixels per sample is valid
    // TODO

    // Check the number of bits per pixel is valid
    match bits_per_px {
        0 => {
            return Err(
                String::from(
                    "The 'Bits Allocated' value must be greater than 0"
                ).into(),
            )
        },
        _ => match bits_per_px % 8 {
            0 => {},
            _ => {
                return Err(
                    String::from(
                        "The 'Bits Allocated' value must be a multiple of 8"
                    ).into()
                )
            }
        }
    }

    let bytes_per_pixel = bits_per_px / 8;

    println!("Parsing header...");
    if enc.len() < 64 {
        return Err(
            String::from(
                "Frame is not long enough to contain RLE encoded data"
            ).into()
        )
    }
    let arr = <&[u8; 64]>::try_from(&enc[0..64]).unwrap();
    let all_offsets: [u32; 15] = parse_header(arr);

    // Get non-zero offsets and find the number of segments
    // `nr_segments` is in [0, 15]
    let mut nr_segments = 0;
    let mut offsets: Vec<u32> = Vec::new();
    for val in all_offsets.iter().filter(|&n| *n != 0) {
        offsets.push(*val);
        nr_segments += 1;
    }

    // Ensure we have a final ending offset
    offsets.push(<u32>::try_from(enc.len()).unwrap());

    println!("Offsets {:?}", offsets);
    println!("Segments {}", nr_segments);
    println!("Bits per pixel {}", bits_per_px);

    // Check the samples per pixel is conformant
    let samples_per_px = nr_segments / bytes_per_pixel;
    match samples_per_px {
        1 => {},
        3 => {},
        _ => return Err(
            String::from("The 'Samples Per Pixel' must be 1 or 3").into()
        ),
    }

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
    let length = (
        px_per_sample * u32::from(bytes_per_pixel) * u32::from(samples_per_px)
    ) as usize;

    let mut out: Vec<u8> = vec![0; length];

    for sample in 0..samples_per_px {
        for byte_offset in 0..bytes_per_pixel {
            let idx = (sample * bytes_per_pixel + byte_offset) as usize;
            let start = offsets[idx] as usize;
            let end = offsets[idx + 1] as usize;
            println!("Start..end {} {}", start, end);
            let data = <&[u8]>::try_from(&enc[start..end]).unwrap();
            let segment = _decode_segment(data)?;
            let segment_len = segment.len();
            println!("Segment length {}", segment_len);
            // Check decoded segment length is good
            if segment_len != px_per_sample as usize {
                //
                println!("Bad segment length");
            }
            //out[]
        }
    }

    Ok(out)
}


fn _decode_segment(enc: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    /* Return a decoded RLE segment as bytes.

    Parameters
    ----------
    enc : &[u8]
        The encoded segment.

    Returns
    -------
    Vec<u8>
        The decoded segment.
    */
    // TODO: make sure we don't grab from outside the array...
    // Maybe return Result instead
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

    Ok(out)
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
    match _decode_segment(enc) {
        Ok(frame) => return Ok(PyBytes::new(py, &frame[..])),
        Err(_e) => {
            return Err(
                PyValueError::new_err("An error occurred decoding the segment")
            )
        }
    }
}


#[pymodule]
fn _rle(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_segment, m)?).unwrap();
    m.add_function(wrap_pyfunction!(decode_frame, m)?).unwrap();

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_le_tests() {
        assert_eq!(0u32, u32_le(&[0u8, 0u8, 0u8, 0u8]));
        assert_eq!(255u32, u32_le(&[255u8, 0u8, 0u8, 0u8]));
        assert_eq!(4278190080u32, u32_le(&[0u8, 0u8, 0u8, 255u8]));
        assert_eq!(u32::MAX, u32_le(&[255u8, 255u8, 255u8, 255u8]));
    }
}
