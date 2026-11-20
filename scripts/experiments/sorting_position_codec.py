"""CPU-only position codec for the frozen matched-geometry diagnostic."""
from pathlib import Path
import struct

import numpy as np


MAGIC = b"SGSPOS01"
HEADER = struct.Struct("<8sQIII")
DESCRIPTOR = struct.Struct("<I6d3B")
BLOCK_SIZE = 4096
UNIFORM_BITS = (8, 12, 16, 20, 24, 32)


def _coordinates(xyz):
    xyz = np.asarray(xyz)
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be an N x 3 float32 array")
    if not 0 < len(xyz) <= np.iinfo(np.uint32).max or not np.isfinite(xyz).all():
        raise ValueError("xyz must be nonempty, finite, and indexable by uint32")
    return xyz


def morton_order(xyz):
    """Morton21 of floor(normalized * (2**21 - 1)); ties use original ID."""
    xyz = _coordinates(xyz).astype(np.float64)
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    span = high - low
    normalized = np.divide(xyz - low, span, out=np.zeros_like(xyz), where=span != 0)
    q = np.floor(normalized * ((1 << 21) - 1)).astype(np.uint64)
    key = np.zeros(len(xyz), dtype=np.uint64)
    for k in range(21):
        for axis in range(3):
            key |= ((q[:, axis] >> k) & 1) << (3 * k + axis)
    return np.argsort(key, kind="stable").astype(np.uint32)


def _adaptive_bits(low, high, epsilon):
    if high == low:
        return 0
    magnitude = max(abs(low), abs(high))
    with np.errstate(over="ignore"):
        rounding = abs(float(np.spacing(np.float32(magnitude)))) / 2
    guard = 2 * abs(float(np.spacing(np.float64(magnitude))))
    for b in range(1, 25):
        bound = (high - low) / (2 * ((1 << b) - 1)) + rounding + guard
        if bound <= epsilon:
            return b
    return 32


def decode(path):
    """Read the complete encoded file and restore its original point order."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated header")
    magic, n, local, block_size, block_count = HEADER.unpack_from(raw)
    if magic != MAGIC or local not in (0, 1) or n == 0:
        raise ValueError("invalid position header")
    expected_size = BLOCK_SIZE if local else n
    if block_size != expected_size or block_count != (n + block_size - 1) // block_size:
        raise ValueError("inconsistent block layout")
    offset = HEADER.size
    permutation = None
    if local:
        if offset + 4 * n > len(raw):
            raise ValueError("truncated permutation")
        permutation = np.frombuffer(raw, dtype="<u4", count=n, offset=offset).copy()
        if not np.array_equal(np.sort(permutation), np.arange(n, dtype=np.uint32)):
            raise ValueError("invalid permutation")
        offset += 4 * n
    ordered = np.empty((n, 3), dtype=np.float32)
    start = 0
    for _ in range(block_count):
        if offset + DESCRIPTOR.size > len(raw):
            raise ValueError("truncated descriptor")
        record = DESCRIPTOR.unpack_from(raw, offset)
        offset += DESCRIPTOR.size
        count, low, high, widths = record[0], record[1:4], record[4:7], record[7:10]
        if count != min(block_size, n - start):
            raise ValueError("invalid block count")
        for axis, b in enumerate(widths):
            if b not in tuple(range(25)) + (32,) or low[axis] > high[axis]:
                raise ValueError("invalid axis descriptor")
            axis_bytes = 4 * count if b == 32 else b * ((count + 7) // 8)
            if offset + axis_bytes > len(raw):
                raise ValueError("truncated axis payload")
            if b == 32:
                value = np.frombuffer(raw, dtype="<f4", count=count, offset=offset)
            elif b == 0:
                if low[axis] != high[axis]:
                    raise ValueError("nonconstant zero-bit axis")
                value = np.full(count, low[axis], dtype=np.float64)
            else:
                plane_size = (count + 7) // 8
                q = np.zeros(count, dtype=np.uint32)
                for k in range(b):
                    plane = np.frombuffer(raw, dtype=np.uint8, count=plane_size,
                                          offset=offset + k * plane_size)
                    q |= np.unpackbits(plane, bitorder="little")[:count].astype(np.uint32) << k
                value = low[axis] + q.astype(np.float64) * ((high[axis] - low[axis]) / ((1 << b) - 1))
            ordered[start:start + count, axis] = value
            offset += axis_bytes
        start += count
    if offset != len(raw) or start != n or not np.isfinite(ordered).all():
        raise ValueError("trailing bytes or invalid decoded positions")
    if permutation is None:
        return ordered
    result = np.empty_like(ordered)
    result[permutation] = ordered
    return result


def encode_decode(xyz_float32, path: Path, *, bits=None, epsilon=None, local=False):
    """Encode, independently decode all bytes, and report measured storage/error.

    Exactly one of uniform ``bits`` or absolute per-coordinate ``epsilon`` is
    required. Adaptive axes above 24 bits escape to their raw float32 values.
    Uniform32 always stores all axes raw, including constant axes.
    """
    xyz = _coordinates(xyz_float32)
    if (bits is None) == (epsilon is None):
        raise ValueError("provide exactly one of bits and epsilon")
    if bits is not None and bits not in UNIFORM_BITS:
        raise ValueError("unsupported uniform bit width")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive")
    path = Path(path)
    n = len(xyz)
    permutation = morton_order(xyz) if local else None
    ordered = xyz[permutation] if local else xyz
    block_size = BLOCK_SIZE if local else n
    block_count = (n + block_size - 1) // block_size
    bits_counts = {str(b): 0 for b in list(range(25)) + [32]}
    payload_bytes = escape_bytes = bitplane_padding_bits = 0
    encoded_values = np.empty_like(ordered)
    with path.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, n, int(local), block_size, block_count))
        if local:
            stream.write(permutation.astype("<u4", copy=False).tobytes())
        for start in range(0, n, block_size):
            block = ordered[start:start + block_size]
            count = len(block)
            block64 = block.astype(np.float64)
            low, high = block64.min(axis=0), block64.max(axis=0)
            widths = [(_adaptive_bits(low[a], high[a], epsilon) if bits is None
                       else (0 if low[a] == high[a] and bits != 32 else bits)) for a in range(3)]
            stream.write(DESCRIPTOR.pack(count, *low, *high, *widths))
            for axis, b in enumerate(widths):
                bits_counts[str(b)] += count
                if b == 32:
                    payload = block[:, axis].astype("<f4", copy=False).tobytes()
                    value = block[:, axis]
                    escape_bytes += len(payload)
                elif b == 0:
                    payload = b""
                    value = np.full(count, low[axis], dtype=np.float64)
                else:
                    levels = (1 << b) - 1
                    q = np.rint((block64[:, axis] - low[axis]) / (high[axis] - low[axis]) * levels).astype(np.uint32)
                    payload = b"".join(np.packbits(((q >> k) & 1).astype(np.uint8),
                                                  bitorder="little").tobytes() for k in range(b))
                    value = low[axis] + q.astype(np.float64) * ((high[axis] - low[axis]) / levels)
                    bitplane_padding_bits += b * (8 * ((count + 7) // 8) - count)
                stream.write(payload)
                payload_bytes += len(payload)
                encoded_values[start:start + count, axis] = value
    expected = np.empty_like(xyz)
    if local:
        expected[permutation] = encoded_values
    else:
        expected[:] = encoded_values
    decoded = decode(path)
    if not np.array_equal(decoded.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("file decode differs from the evaluated encoded positions")
    if bits == 32 and not np.array_equal(decoded.view(np.uint32), xyz.view(np.uint32)):
        raise AssertionError("uniform32 did not preserve the original float32 bit patterns")
    error = np.abs(decoded.astype(np.float64) - xyz.astype(np.float64))
    max_axis_error = error.max(axis=0)
    epsilon_ok = epsilon is None or bool(np.all(max_axis_error <= epsilon))
    if not epsilon_ok:
        raise AssertionError("adaptive file violates the predeclared coordinate error bound")
    header_bytes = HEADER.size + block_count * DESCRIPTOR.size
    permutation_bytes = 4 * n if local else 0
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + permutation_bytes + payload_bytes:
        raise AssertionError("file byte accounting does not balance")
    return decoded, {
        "points": n, "local": bool(local), "block_size": block_size,
        "block_count": block_count, "uniform_bits": bits, "epsilon": epsilon,
        "bytes": file_bytes, "header_bytes": header_bytes,
        "fixed_header_bytes": HEADER.size, "descriptor_bytes": block_count * DESCRIPTOR.size,
        "permutation_bytes": permutation_bytes, "payload_bytes": payload_bytes,
        "escape_bytes": escape_bytes, "escape_coordinates": bits_counts["32"],
        "bits_counts_coordinates": bits_counts, "bitplane_padding_bits": bitplane_padding_bits,
        "max_coordinate_error": float(max_axis_error.max()),
        "max_axis_error": max_axis_error.tolist(), "epsilon_compliant": epsilon_ok,
        "complete_file_decode_verified": True,
        "morton_rule": "floor(normalized * (2**21 - 1)), stable original-ID ties" if local else None,
    }
