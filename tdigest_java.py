import base64
import struct

import tdigest

def read_tdigest(binary_tdigest):
    if isinstance(binary_tdigest, str):
        binary_tdigest = base64.b64decode(binary_tdigest)
    tdigest_encoding, = struct.unpack_from('>i', binary_tdigest)
    assert tdigest_encoding == 2
    t_min, t_max, t_compression, size, buffer_size, centroids = struct.unpack_from('>ddfhhh', binary_tdigest, 4)
    td = tdigest.TDigest()
    for i in range(centroids):
        weight, mean = struct.unpack_from('>ff', binary_tdigest, 30 + i * 8)
        td.update(mean, weight)
    td.compress()
    return td