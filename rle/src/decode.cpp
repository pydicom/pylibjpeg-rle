
// Includes
#include "std/stdio.hpp"
#include "std/string.hpp"
#include <sstream>

#include "decode.hpp"


std::string Decode(char *inArray, char *outArray, int inLength, int outLength)
{
    /*

    Parameters
    ----------
    char *inArray
        Pointer to the first element of a byte array containing the RLE
        data to be decompressed.
    char *outArray
        Pointer to the first element of a numpy.ndarray where the decompressed
        RLE data should be written.
    int inLength
        Length of the input array
    int outLength
        Expected length of the output array
    */

    //
};

int decode_segment(const char *segment, unsigned int expected_length) {
    /*

    Parameters
    ----------
    char *segment
        The RLE encoded segment to be decoded
    int length
        Length of the segment
    */

    // Output
    std::stringstream out;
    // Current position
    unsigned int pos = 0;
    // Length of the output
    unsigned int output_length = 0;
    unsigned char header_byte;
    // Can only copy or extend by at most 128 bytes per loop
    // TODO:
    //  Need proper output location
    //  Need to fix up pointers, etc
    //  Need to make sure we don't go past the end of the input
    while (output_length < expected_length) {
        header_byte = segment[pos] + 1;
        pos += 1;
        if (header_byte > 129) {
            // Extend by copying the next byte (-N + 1) times, however since
            // we are using uint8 this will be (256 - N + 1) times
            // memset(ptr out, int value, int nr of bytes to set to `value`)
            extend_len = 258 - header_byte;
            memset(out, segment[pos], extend_len);
            output_length += extend_len;
            pos += 1;
        } else if (header_byte < 129) {
            // Extend by literally copying the next (N + 1) bytes
            // memcopy(ptr out, ptr src, int nr of bytes to copy)
            memcopy(out, segment[pos], header_byte)
            //result.extend(segment[pos:pos + header_byte])
            output_length += header_byte;
            pos += header_byte;
        }
    }

    if (output_length != expected_length) {
        return -1;
    }

    return out;
};
