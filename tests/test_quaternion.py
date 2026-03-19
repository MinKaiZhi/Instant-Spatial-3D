import numpy as np

from is3d.math.quaternion import nlerp, pack_quaternion_32, unpack_quaternion_32


def _random_unit_quaternion(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4).astype(np.float32)
    q /= np.linalg.norm(q)
    return q


def test_quaternion_pack_unpack_roundtrip() -> None:
    q = _random_unit_quaternion(7)
    packed = pack_quaternion_32(q)
    unpacked = unpack_quaternion_32(packed)

    similarity = abs(float(np.dot(q, unpacked)))
    assert similarity > 0.95


def test_nlerp_outputs_unit_quaternion() -> None:
    q0 = _random_unit_quaternion(1)
    q1 = _random_unit_quaternion(2)

    q = nlerp(q0, q1, 0.35)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
