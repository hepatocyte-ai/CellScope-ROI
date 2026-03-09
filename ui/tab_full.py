from typing import Dict

import numpy as np
import streamlit as st

from core.config import CellCounterConfig
from core.counter import CellCounter
from core.visualizer import SegmentationVisualizer
from ui.results import show_results
from utils.cache import load_cached_result, run_cache_dir, save_cached_result


def _run(image_np, mask_np, class_names, config, palette, vis_params):
    counts, labeled = CellCounter(class_names, config).count(mask_np)
    vis = SegmentationVisualizer(class_names, palette, **vis_params).draw(
        image_np, labeled, counts
    )
    return counts, vis


def render_tab_full(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    class_names: Dict[int, str],
    config: CellCounterConfig,
    palette: dict,
    vis_params: dict,
    pair_hash: str,
) -> None:
    cache_dir = run_cache_dir(pair_hash, class_names, config, palette, vis_params)

    # ── Автозагрузка / инвалидация при смене параметров ──────────────
    if st.session_state.get("_full_cache_dir") != cache_dir:
        st.session_state["_full_cache_dir"] = cache_dir
        cached = load_cached_result(cache_dir, "full")
        if cached:
            st.session_state["counts_full"], st.session_state["vis_full"] = cached
            st.session_state["_full_from_cache"] = True
        else:
            st.session_state.pop("counts_full", None)
            st.session_state.pop("vis_full", None)
            st.session_state["_full_from_cache"] = False

    # ── Кнопка запуска ───────────────────────────────────────────────
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶️ Запустить анализ", type="primary", key="btn_full")
    with col_hint:
        if st.session_state.get("_full_from_cache"):
            st.info("⚡ Результат загружен из кэша. Нажмите «Запустить» для пересчёта.")

    if run_clicked:
        with st.spinner("Выполняется подсчёт клеток…"):
            counts, vis = _run(image_np, mask_np, class_names, config, palette, vis_params)
            save_cached_result(cache_dir, "full", counts, vis)
        st.session_state.update(
            counts_full=counts, vis_full=vis, _full_from_cache=False
        )

    if "vis_full" in st.session_state:
        show_results(
            st.session_state["counts_full"],
            st.session_state["vis_full"],
            prefix="full_",
        )