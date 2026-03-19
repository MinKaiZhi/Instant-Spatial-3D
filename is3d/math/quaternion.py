from __future__ import annotations

import numpy as np


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / norm


def _quantize_signed_10(value: float) -> int:
    clipped = float(np.clip(value, -1.0, 1.0))
    return int(round((clipped + 1.0) * 0.5 * 1023.0))


def _dequantize_signed_10(value: int) -> float:
    return (float(value) / 1023.0) * 2.0 - 1.0


def pack_quaternion_32(quat_xyzw: np.ndarray) -> int:
    q = normalize_quaternion(quat_xyzw)
    sign_w = 1 if q[3] < 0.0 else 0
    qx = _quantize_signed_10(q[0])
    qy = _quantize_signed_10(q[1])
    qz = _quantize_signed_10(q[2])

    packed = (sign_w << 31) | (qx << 21) | (qy << 11) | (qz << 1)
    return int(packed)


def unpack_quaternion_32(packed: int) -> np.ndarray:
    sign_w = (packed >> 31) & 0x1
    qx = (packed >> 21) & 0x3FF
    qy = (packed >> 11) & 0x3FF
    qz = (packed >> 1) & 0x3FF

    x = _dequantize_signed_10(qx)
    y = _dequantize_signed_10(qy)
    z = _dequantize_signed_10(qz)

    w_sq = max(0.0, 1.0 - x * x - y * y - z * z)
    w = np.sqrt(w_sq)
    if sign_w:
        w = -w

    return normalize_quaternion(np.array([x, y, z, w], dtype=np.float32))


def nlerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    q0 = normalize_quaternion(q0_xyzw)
    q1 = normalize_quaternion(q1_xyzw)

    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1

    blended = (1.0 - t) * q0 + t * q1
    return normalize_quaternion(blended)
