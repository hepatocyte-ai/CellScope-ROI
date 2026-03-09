import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_mask_bytes(data: bytes, filename: str) -> np.ndarray:
    """Загружает маску из сырых байт."""
    if filename.endswith(".npy"):
        return np.load(io.BytesIO(data))
    return np.array(Image.open(io.BytesIO(data)).convert("L"))


def load_mask(uploaded_file) -> np.ndarray:
    """Загружает маску из UploadedFile Streamlit."""
    return load_mask_bytes(uploaded_file.read(), uploaded_file.name)


def to_rgb_image(arr: np.ndarray) -> np.ndarray:
    """Нормализует любой массив к 3-канальному uint8 RGB."""
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.copy()


def auto_resize_image(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, bool]:
    """
    Если размеры изображения и маски отличаются не более чем на `threshold`
    (по умолчанию 5 %) по каждой оси — масштабирует изображение под маску.
    Возвращает (изображение, флаг_масштабирования).
    """
    H_img, W_img   = image_np.shape[:2]
    H_mask, W_mask = mask_np.shape[:2]

    if (H_img, W_img) == (H_mask, W_mask):
        return image_np, False

    h_diff = abs(H_img - H_mask) / max(H_mask, 1)
    w_diff = abs(W_img - W_mask) / max(W_mask, 1)

    if h_diff < threshold and w_diff < threshold:
        resized = cv2.resize(
            image_np,
            (W_mask, H_mask),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, True

    return image_np, False