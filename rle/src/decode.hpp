
#include <iostream>
#include <string>

#ifndef DECODE_HPP
#define DECODE_HPP

    // Prototypes
    // Decode an RLE encoded frame `inArray` to `outArray`
    //extern std::string decode_frame(
    //    char *inArray,
    //    char *outArray,
    //    int inLength,
    //    int outLength,
    //);
    // Decode an RLE encoded segment
    //extern decode_segment();
    // Parse the RLE header
    extern int decode_segment(const char *segment, int length);

#endif
