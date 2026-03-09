"""
🔬 Cell Counter — точка входа Streamlit
"""

import hashlib
import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import streamlit as st
from PIL import Image

from ui import render_sidebar, render_tab_full, render_tab_roi
from utils.cache import ensure_pair_saved, list_saved_pairs, load_pair
from utils.io import auto_resize_image, load_mask_bytes

logging.basicConfig(level=logging.WARNING)

# Метки режимов — используются в двух местах, вынесены в константы
_MODE_UPLOAD  = "⬆️ Загрузить новую пару"
_MODE_HISTORY = "🗂️ Выбрать из истории"


def _load_from_upload() -> Optional[tuple]:
    """
    Показывает виджеты загрузки файлов.
    Возвращает (img_bytes, mask_bytes, img_name, mask_name, pair_hash) или None.
    """
    col1, col2 = st.columns(2)
    with col1:
        up_img = st.file_uploader(
            "Исходное изображение",
            type=["png", "jpg", "jpeg", "tiff"],
            help="Гистологический срез",
        )
    with col2:
        up_mask = st.file_uploader(
            "Маска сегментации",
            type=["png", "jpeg", "npy"],
            help="Grayscale PNG или NumPy .npy; значения пикселей = id класса",
        )

    if up_img is None or up_mask is None:
        st.info("⬆️ Загрузите изображение и маску, чтобы начать анализ.")
        return None

    img_bytes  = up_img.getvalue()
    mask_bytes = up_mask.getvalue()

    try:
        _, pair_hash = ensure_pair_saved(
            img_bytes, mask_bytes, up_img.name, up_mask.name
        )
    except Exception as exc:
        st.warning(f"Не удалось сохранить данные на диск: {exc}")
        pair_hash = hashlib.sha1(img_bytes + mask_bytes).hexdigest()[:12]

    return img_bytes, mask_bytes, up_img.name, up_mask.name, pair_hash


def _load_from_history(saved_pairs: list) -> Optional[tuple]:
    """
    Показывает выпадающий список сохранённых пар.
    Возвращает (img_bytes, mask_bytes, img_name, mask_name, pair_hash) или None.
    """
    # Строим читаемые метки: «image.png + mask.png  [2025-06-10T14:32:07]»
    labels = [
        f"{p['img_name']} + {p['mask_name']}  [{p.get('saved_at', '—')}]"
        for p in saved_pairs
    ]
    hashes = [p["pair_hash"] for p in saved_pairs]

    chosen_idx = st.selectbox(
        "Доступные пары",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key="history_selectbox",
    )
    pair_hash = hashes[chosen_idx]

    result = load_pair(pair_hash)
    if result is None:
        st.error(
            "❌ Не удалось прочитать файлы из истории. "
            "Возможно, они были удалены вручную."
        )
        return None

    img_bytes, mask_bytes, img_name, mask_name = result

    st.success(
        f"✅ Загружена пара: **{img_name}** + **{mask_name}**  "
        f"·  хэш `{pair_hash}`"
    )
    return img_bytes, mask_bytes, img_name, mask_name, pair_hash


def main() -> None:
    st.set_page_config(
        page_title="🔬 Cell Counter",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Cell Counter")
    st.caption(
        "Автоматический подсчёт клеток на масках сегментации · "
        "Distance Transform + Watershed"
    )

    class_names, palette, config, vis_params = render_sidebar()

    # ── Выбор источника данных ────────────────────────────────────────
    st.subheader("📂 Загрузка данных")

    saved_pairs = list_saved_pairs()

    if saved_pairs:
        source = st.radio(
            "Источник данных",
            [_MODE_UPLOAD, _MODE_HISTORY],
            horizontal=True,
            key="source_mode",
        )
    else:
        source = _MODE_UPLOAD

    st.divider()

    # ── Получение байтов и имён файлов ───────────────────────────────
    if source == _MODE_UPLOAD:
        loaded = _load_from_upload()
    else:
        loaded = _load_from_history(saved_pairs)

    if loaded is None:
        return

    img_bytes, mask_bytes, img_name, mask_name, pair_hash = loaded

    # ── Декодирование массивов ────────────────────────────────────────
    try:
        image_np = np.array(Image.open(io.BytesIO(img_bytes)))
        mask_np  = load_mask_bytes(mask_bytes, mask_name)
    except Exception as exc:
        st.error(f"Ошибка при загрузке файлов: {exc}")
        return

    # ── Автоподгонка при расхождении размеров < 5 % ──────────────────
    h_orig, w_orig = image_np.shape[:2]
    image_np, was_resized = auto_resize_image(image_np, mask_np)

    if was_resized:
        H_new, W_new = image_np.shape[:2]
        st.warning(
            f"⚠️ Размеры отличались менее чем на 5 %: "
            f"изображение {w_orig} × {h_orig} px → "
            f"масштабировано до {W_new} × {H_new} px (по маске)."
        )
    elif image_np.shape[:2] != mask_np.shape[:2]:
        st.error(
            f"Размеры не совпадают (разница ≥ 5 %): "
            f"изображение {image_np.shape[1]} × {image_np.shape[0]}, "
            f"маска {mask_np.shape[1]} × {mask_np.shape[0]}."
        )
        return

    H, W = image_np.shape[:2]

    with st.expander("ℹ️ Информация о файлах"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Размер", f"{W} × {H} px")
        c2.metric("Dtype маски", str(mask_np.dtype))
        c3.metric(
            "Классы в маске",
            str(sorted(np.unique(mask_np).tolist())),
        )

    st.divider()

    tab_full, tab_roi = st.tabs(
        ["🖼️ Полное изображение", "✂️ Область интереса (ROI)"]
    )

    with tab_full:
        render_tab_full(
            image_np, mask_np, class_names, config, palette, vis_params, pair_hash
        )

    with tab_roi:
        render_tab_roi(
            image_np, mask_np, class_names, config, palette, vis_params, pair_hash
        )


if __name__ == "__main__":
    main()