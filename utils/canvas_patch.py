"""
Патч совместимости для streamlit-drawable-canvas на Streamlit Cloud.
Добавляет image_to_url если Streamlit >= 1.26 убрал её из публичного API.
"""
import base64
from io import BytesIO
from typing import Any

import streamlit.elements.image as _st_img
from PIL import Image


def _synthetic_image_to_url(
    image: Any,
    width: int,
    clamp: bool,
    channels: str,
    output_format: str,
    image_id: str,
    *args,
    **kwargs,
) -> str:
    """Конвертирует PIL Image → data URL (base64 PNG/JPEG)."""
    if isinstance(image, Image.Image):
        fmt = (output_format or "PNG").upper()
        # JPEG не поддерживает прозрачность
        if fmt == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if fmt == "JPEG" else "png"
        return f"data:image/{mime};base64,{b64}"

    # Если уже строка — вернуть как есть
    if isinstance(image, str):
        return image

    return ""


def apply() -> None:
    """
    Вызывать один раз в начале app.py.
    Безопасен — не трогает ничего если функция уже есть.
    """
    if not hasattr(_st_img, "image_to_url"):
        _st_img.image_to_url = _synthetic_image_to_url