from __future__ import annotations

import numpy as np


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape [H, W, 3]")

    image_f32 = image.astype(np.float32)
    if image_f32.max() > 1.0:
        image_f32 /= 255.0

    luminance = image_f32[..., 0] * 0.2126 + image_f32[..., 1] * 0.7152 + image_f32[..., 2] * 0.0722
    semantic_mask = luminance > 0.02
    return image_f32, semantic_mask


def extract_features(image: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722

    grad_x = np.zeros_like(luminance)
    grad_y = np.zeros_like(luminance)
    grad_x[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1])
    grad_y[1:, :] = np.abs(luminance[1:, :] - luminance[:-1, :])
    edge_strength = np.clip(grad_x + grad_y, 0.0, 1.0)

    chroma = np.std(image, axis=-1)

    features = np.stack([luminance, edge_strength, chroma], axis=-1)
    features *= semantic_mask[..., None]
    return features.astype(np.float32)
