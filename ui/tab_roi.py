"""ROI-вкладка: прямоугольник, полигон и свободный контур."""

import hashlib
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.config import CellCounterConfig
from core.counter import CellCounter
from core.visualizer import SegmentationVisualizer
from ui.results import show_comparison, show_results
from utils.cache import load_cached_result, run_cache_dir, save_cached_result
from utils.io import to_rgb_image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ── Режимы рисования ──────────────────────────────────────────────────────────

_MODE_RECT    = "🔲 Прямоугольник"
_MODE_POLYGON = "🔷 Полигон"
_MODE_FREE    = "✏️ Свободный контур"

_CANVAS_MODE = {
    _MODE_RECT:    "rect",
    _MODE_POLYGON: "polygon",
    _MODE_FREE:    "freedraw",
}

_HINTS = {
    _MODE_RECT:    "🖱️ Нарисуйте прямоугольник.",
    _MODE_POLYGON: "🖱️ Кликайте для добавления вершин; **двойной клик** замыкает контур.",
    _MODE_FREE:    "✏️ Нарисуйте замкнутый контур вручную, стараясь вернуться к точке старта.",
}


# ── Пайплайн ──────────────────────────────────────────────────────────────────

def _run(image_np, mask_np, class_names, config, palette, vis_params):
    counts, labeled = CellCounter(class_names, config).count(mask_np)
    vis = SegmentationVisualizer(class_names, palette, **vis_params).draw(
        image_np, labeled, counts
    )
    return counts, vis


# ── Геометрические утилиты ────────────────────────────────────────────────────

def _extract_rect(
    obj: dict, scale: float, H: int, W: int
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box прямоугольника из JSON-объекта canvas (canvas-px → оригинал)."""
    x0 = max(0, int(obj["left"] / scale))
    y0 = max(0, int(obj["top"]  / scale))
    x1 = min(W, int((obj["left"] + obj["width"])  / scale))
    y1 = min(H, int((obj["top"]  + obj["height"]) / scale))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _shape_mask_from_image_data(
    img_data: np.ndarray,
    H: int,
    W: int,
    canvas_mode: str,
) -> Optional[np.ndarray]:
    """
    Извлекает бинарную uint8-маску (H×W) из RGBA image_data canvas.

    polygon  — canvas рисует fill с alpha>0, порог достаточен.
    freedraw — только stroke; морфологическое замыкание + fillPoly
               превращают незамкнутую линию в область.
    """
    alpha = img_data[:, :, 3]
    if alpha.max() == 0:
        return None

    # Масштабируем к исходному размеру
    alpha_full = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
    binary = (alpha_full > 10).astype(np.uint8)

    if canvas_mode == "polygon":
        # fill уже нанесён canvas-движком — берём как есть
        return binary

    # freedraw: замыкаем пробелы адаптивным ядром и заливаем
    k = max(4, min(W, H) // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return binary  # fallback — возвращаем сам штрих

    filled = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(filled, [max(contours, key=cv2.contourArea)], 1)
    return filled if filled.max() > 0 else binary


def _bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) ненулевых пикселей маски."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0 = int(np.where(rows)[0][0]);  y1 = int(np.where(rows)[0][-1]) + 1
    x0 = int(np.where(cols)[0][0]);  x1 = int(np.where(cols)[0][-1]) + 1
    return x0, y0, x1, y1


def _prefix_for_shape(shape_mask: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(shape_mask).tobytes()).hexdigest()[:12]
    return f"roi_shape_{h}"


# ── Кнопка запуска + кэш ─────────────────────────────────────────────────────

def _roi_button(
    roi_image_np: np.ndarray,
    roi_mask_np: np.ndarray,
    class_names, config, palette, vis_params,
    cache_dir: str,
    roi_prefix: str,
    button_key: str,
) -> None:
    """Инвалидирует кэш при смене ROI/параметров, запускает анализ по кнопке."""
    roi_ck = f"{cache_dir}/{roi_prefix}"

    if st.session_state.get("_roi_cache_key") != roi_ck:
        st.session_state["_roi_cache_key"] = roi_ck
        cached = load_cached_result(cache_dir, roi_prefix)
        if cached:
            st.session_state["counts_roi"], st.session_state["vis_roi"] = cached
            st.session_state["_roi_from_cache"] = True
        else:
            st.session_state.pop("counts_roi", None)
            st.session_state.pop("vis_roi", None)
            st.session_state["_roi_from_cache"] = False

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        clicked = st.button("▶️ Анализ ROI", type="primary", key=button_key)
    with col_hint:
        if st.session_state.get("_roi_from_cache"):
            st.info("⚡ Результат загружен из кэша. Нажмите «Запустить» для пересчёта.")

    if clicked:
        with st.spinner("Анализ выбранной области..."):
            c, v = _run(
                roi_image_np, roi_mask_np,
                class_names, config, palette, vis_params,
            )
            save_cached_result(cache_dir, roi_prefix, c, v)
        st.session_state.update(counts_roi=c, vis_roi=v, _roi_from_cache=False)


# ── Canvas-вариант ────────────────────────────────────────────────────────────

def _canvas_roi(
    image_np, mask_np, class_names, config, palette, vis_params,
    H, W, pair_hash,
):
    # ── Выбор режима ──────────────────────────────────────────────────
    draw_label  = st.radio(
        "Режим выделения",
        [_MODE_RECT, _MODE_POLYGON, _MODE_FREE],
        horizontal=True,
        key="draw_mode",
    )
    canvas_mode = _CANVAS_MODE[draw_label]
    st.info(_HINTS[draw_label] + "  Затем нажмите **«Анализ ROI»**.")

    # ── Canvas ────────────────────────────────────────────────────────
    MAX_DISP = 720
    scale    = min(MAX_DISP / W, MAX_DISP / H, 1.0)
    dw, dh   = int(W * scale), int(H * scale)

    # Для прямоугольника и полигона показываем полупрозрачную заливку;
    # для фриханда — только штрих (fill alpha = 0).
    fill_alpha   = {"rect": "0.15", "polygon": "0.30", "freedraw": "0.00"}[canvas_mode]
    stroke_width = 4 if canvas_mode == "freedraw" else 2

    canvas_result = st_canvas(
        fill_color       = f"rgba(255, 100, 0, {fill_alpha})",
        stroke_color     = "#FF4500",
        stroke_width     = stroke_width,
        background_image = Image.fromarray(image_np),
        update_streamlit = True,
        width            = dw,
        height           = dh,
        drawing_mode     = canvas_mode,
        # Уникальный ключ по режиму — сброс canvas при переключении
        key              = f"roi_canvas_{canvas_mode}",
    )

    cache_dir  = run_cache_dir(pair_hash, class_names, config, palette, vis_params)
    roi_ready  = False
    roi_image  = None
    roi_mask   = None
    roi_prefix = None

    # ── Извлечение ROI ────────────────────────────────────────────────
    if draw_label == _MODE_RECT:
        if canvas_result.json_data:
            objs = canvas_result.json_data.get("objects", [])
            if objs:
                bbox = _extract_rect(objs[-1], scale, H, W)
                if bbox:
                    x0, y0, x1, y1 = bbox
                    roi_ready  = True
                    roi_image  = image_np[y0:y1, x0:x1]
                    roi_mask   = mask_np[y0:y1, x0:x1]
                    roi_prefix = f"roi_{x0}_{y0}_{x1}_{y1}"
                    st.success(
                        f"📐 x=[{x0}:{x1}], y=[{y0}:{y1}]  ·  {x1-x0} × {y1-y0} px"
                    )

    else:  # polygon / freedraw
        if canvas_result.image_data is not None:
            shape_mask = _shape_mask_from_image_data(
                canvas_result.image_data, H, W, canvas_mode
            )
            if shape_mask is not None and shape_mask.max() > 0:
                x0, y0, x1, y1 = _bounding_box(shape_mask)
                crop_shape = shape_mask[y0:y1, x0:x1]
                roi_ready  = True
                roi_image  = image_np[y0:y1, x0:x1].copy()
                # Пиксели вне формы обнуляем — в подсчёт не войдут
                roi_mask   = (
                    mask_np[y0:y1, x0:x1] * crop_shape
                ).astype(mask_np.dtype)
                roi_prefix = _prefix_for_shape(shape_mask)
                icon       = "🔷" if draw_label == _MODE_POLYGON else "✏️"
                st.success(
                    f"{icon} Bbox: x=[{x0}:{x1}], y=[{y0}:{y1}]  "
                    f"·  площадь формы {int(shape_mask.sum()):,} px"
                )

    # ── Кнопка анализа ────────────────────────────────────────────────
    if roi_ready and roi_image is not None:
        _roi_button(
            roi_image, roi_mask,
            class_names, config, palette, vis_params,
            cache_dir, roi_prefix, "btn_roi_canvas",
        )
    else:
        st.button("▶️ Анализ ROI", type="primary", key="btn_roi_canvas", disabled=True)


# ── Слайдеры-фолбэк ───────────────────────────────────────────────────────────
def _extract_rotated_roi(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    cx: int, cy: int,
    rw: int, rh: int,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вырезает ROI с учётом угла наклона.
    При angle=0 — быстрый путь без трансформации.
    При angle≠0 — разворачивает снимок вокруг (cx,cy) на -angle,
    после чего прямоугольник становится оси-выровненным, и делает кроп.
    """
    H, W = image_np.shape[:2]
    x0 = max(0, cx - rw // 2);  x1 = min(W, cx + rw // 2)
    y0 = max(0, cy - rh // 2);  y1 = min(H, cy + rh // 2)

    if angle == 0:
        return image_np[y0:y1, x0:x1].copy(), mask_np[y0:y1, x0:x1].copy()

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)

    rot_img = cv2.warpAffine(
        image_np, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    rot_mask = cv2.warpAffine(
        mask_np.astype(np.float32), M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(mask_np.dtype)

    return rot_img[y0:y1, x0:x1].copy(), rot_mask[y0:y1, x0:x1].copy()


def _draw_rotated_rect(
    img: np.ndarray,
    cx: int, cy: int,
    rw: int, rh: int,
    angle: float,
    color: tuple = (255, 69, 0),
    fill_alpha: float = 0.15,
) -> None:
    """Рисует полупрозрачный наклонный прямоугольник + крест в центре (in-place)."""
    rect    = ((float(cx), float(cy)), (float(rw), float(rh)), float(-angle))
    box     = cv2.boxPoints(rect).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [box], color)
    cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0, img)
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, markerSize=18, thickness=2)


def _sliders_roi(
    image_np, mask_np, class_names, config, palette, vis_params,
    H, W, pair_hash,
):
    # ── Инициализация состояния ──────────────────────────────────────
    for key, val in {
        "roi_cx": W // 2, "roi_cy": H // 2,
        "roi_rw": W,      "roi_rh": H,
        "roi_angle": 0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Пресеты ──────────────────────────────────────────────────────
    st.caption("⚡ Быстрый выбор")
    pc = st.columns(5)
    _presets = [
        (0, "🖼️ Весь снимок", W,          H         ),
        (1, "▪️ 75%",         int(W*.75),  int(H*.75)),
        (2, "▪️ 50%",         W // 2,      H // 2    ),
        (3, "▪️ 25%",         W // 4,      H // 4    ),
        (4, "↺ Сброс угла",  None,        None      ),
    ]
    for i, label, pw, ph in _presets:
        if pc[i].button(label, use_container_width=True, key=f"roi_preset_{i}"):
            if pw is not None:
                st.session_state.update(
                    roi_cx=W // 2, roi_cy=H // 2,
                    roi_rw=pw,     roi_rh=ph, roi_angle=0,
                )
            else:
                st.session_state["roi_angle"] = 0
            st.rerun()

    st.divider()

    # ── Положение и размер ───────────────────────────────────────────
    pos_col, size_col = st.columns(2)

    with pos_col:
        st.caption("📍 Центр области")
        cx = st.slider("X центра", 0, W, key="roi_cx")
        cy = st.slider("Y центра", 0, H, key="roi_cy")

    with size_col:
        st.caption("📐 Размер")
        lock_ar = st.checkbox("🔒 Зафиксировать пропорции", key="roi_lock_ar")
        rw = st.slider("Ширина", 2, W, step=1, key="roi_rw")

        if lock_ar:
            # Сохраняем AR в момент включения замка
            if "roi_locked_ar" not in st.session_state:
                st.session_state["roi_locked_ar"] = (
                    st.session_state["roi_rh"] / max(2, st.session_state["roi_rw"])
                )
            ar = st.session_state["roi_locked_ar"]
            rh = max(2, min(H, int(rw * ar)))
            st.session_state["roi_rh"] = rh          # синхронизируем для разблокировки
            st.info(f"Высота: **{rh} px** · пропорция 1 : {ar:.2f}")
        else:
            st.session_state.pop("roi_locked_ar", None)
            rh = st.slider("Высота", 2, H, step=1, key="roi_rh")

    # ── Угол наклона ─────────────────────────────────────────────────
    st.divider()
    angle_col, angle_meta = st.columns([3, 2])

    with angle_col:
        angle = st.slider(
            "🔄 Угол наклона (°)", -180, 180, step=1, key="roi_angle",
            help=(
                "Поворот ROI вокруг его центра. "
                "Перед вырезкой снимок разворачивается на обратный угол — "
                "наклонная область анализируется как прямоугольная."
            ),
        )
    with angle_meta:
        direction = ("↻ по часовой" if angle > 0
                     else ("↺ против часовой" if angle < 0 else "горизонтально"))
        st.metric("Угол", f"{angle}°", delta=direction)

    # ── Финальные значения ───────────────────────────────────────────
    cx = int(np.clip(cx, 0, W))
    cy = int(np.clip(cy, 0, H))
    rw = int(np.clip(rw, 2, W))
    rh = int(np.clip(rh, 2, H))

    # ── Сводные метрики ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ширина",     f"{rw} px")
    m2.metric("Высота",     f"{rh} px")
    m3.metric("Площадь",    f"{rw * rh:,} px²")
    m4.metric("% снимка",   f"{rw * rh / max(1, W * H) * 100:.1f}%")

    # ── Двойной превью ───────────────────────────────────────────────
    overview_col, zoom_col = st.columns([3, 2])

    preview = to_rgb_image(image_np.copy())
    _draw_rotated_rect(preview, cx, cy, rw, rh, angle)

    with overview_col:
        st.image(
            preview,
            caption="Обзор: выделенная область на снимке",
            use_container_width=True,
        )

    roi_img, roi_msk = _extract_rotated_roi(
        image_np, mask_np, cx, cy, rw, rh, angle
    )

    with zoom_col:
        if roi_img.size > 0:
            st.image(
                to_rgb_image(roi_img),
                caption=f"Предпросмотр ROI · {rw}×{rh} px · {angle}°",
                use_container_width=True,
            )
        else:
            st.warning("⚠️ ROI выходит за границы снимка")

    # ── Кнопка анализа ───────────────────────────────────────────────
    cache_dir  = run_cache_dir(pair_hash, class_names, config, palette, vis_params)
    roi_prefix = f"roi_{cx}_{cy}_{rw}_{rh}_{angle}"

    if roi_img.size > 0:
        _roi_button(
            roi_img, roi_msk,
            class_names, config, palette, vis_params,
            cache_dir, roi_prefix, "btn_roi_sliders",
        )
    else:
        st.error("ROI пустой — переместите центр или уменьшите размер")
        st.button("▶️ Анализ ROI", disabled=True, key="btn_roi_sliders")


# ── Публичная точка входа ─────────────────────────────────────────────────────

def render_tab_roi(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    class_names: Dict[int, str],
    config: CellCounterConfig,
    palette: dict,
    vis_params: dict,
    pair_hash: str,
) -> None:
    H, W = image_np.shape[:2]

    _sliders_roi(
        image_np, mask_np, class_names, config, palette, vis_params,
        H, W, pair_hash,
    )

    if "vis_roi" in st.session_state:
        st.divider()
        show_results(
            st.session_state["counts_roi"],
            st.session_state["vis_roi"],
            prefix="roi_",
        )
        if "counts_full" in st.session_state:
            st.divider()
            show_comparison(
                st.session_state["counts_full"],
                st.session_state["counts_roi"],
            )