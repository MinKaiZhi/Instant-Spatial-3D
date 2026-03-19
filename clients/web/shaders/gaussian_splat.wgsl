// IS3D shader placeholder: rotation unpack must normalize immediately.

fn unpack_rotation(q: vec4<f32>) -> vec4<f32> {
  // Mandatory rule from project spec: normalize right after unpack.
  return normalize(q);
}
