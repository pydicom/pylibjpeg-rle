

from struct import unpack


def parse_header(data):
    # Little endian
    nr_segments = unpack('<L', data[:4])[0]

    offsets = unpack(
        '<{}L'.format(nr_segments), data[4:4 * (nr_segments + 1)]
    )
    print(nr_segments, offsets)
    return data[:64]
