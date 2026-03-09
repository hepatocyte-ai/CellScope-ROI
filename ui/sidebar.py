from typing import Dict, Set, Tuple

import streamlit as st
from matplotlib.colors import to_rgb

from core.config import ClassConfig, CellCounterConfig
from utils.constants import DEFAULT_COLORS


def render_sidebar() -> Tuple[
    Dict[int, str],   # class_names
    dict,             # palette
    CellCounterConfig,
    dict,             # vis_params
]:
    with st.sidebar:
        st.header("⚙️ Настройки")

        # ── Классы ──────────────────────────────────────────
        st.subheader("🏷️ Классы")
        num_classes = int(st.number_input("Количество классов", 2, 10, 5))

        class_names: Dict[int, str] = {}
        class_colors: Dict[int, str] = {}
        skip_ids: Set[int] = set()

        for i in range(num_classes):
            col_c, col_n, col_bg = st.columns([1, 4, 1])
            with col_c:
                class_colors[i] = st.color_picker(
                    f"c{i}",
                    value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                    key=f"color_{i}",
                    label_visibility="collapsed",
                )
            with col_n:
                default_name = "background" if i == 0 else f"class_{i}"
                class_names[i] = st.text_input(
                    f"name{i}",
                    value=default_name,
                    key=f"name_{i}",
                    label_visibility="collapsed",
                )
            with col_bg:
                if st.checkbox("bg", value=(i == 0), key=f"bg_{i}", help="Не считать"):
                    skip_ids.add(i)

        st.caption("🔲 bg = фон, не считается")
        st.divider()

        # ── Глобальные параметры алгоритма ──────────────────
        st.subheader("🔧 Алгоритм")
        g_area = st.slider("Min area (px²)", 10, 2000, 100)
        g_dist = st.slider("Min distance (px)", 3, 60, 15)
        g_morph = st.select_slider(
            "Morph kernel", options=[1, 3, 5, 7, 9], value=3
        )

        # ── Параметры по классам ─────────────────────────────
        with st.expander("🔬 Параметры по классам"):
            per_class_config: Dict[int, ClassConfig] = {}
            for i in range(num_classes):
                if i in skip_ids:
                    continue
                st.markdown(f"**{class_names.get(i, f'class_{i}')}**")
                col1, col2 = st.columns(2)
                with col1:
                    area_i = st.number_input(
                        "Min area", value=g_area, min_value=1,
                        step=10, key=f"pa_{i}",
                    )
                    dist_i = st.number_input(
                        "Min dist", value=g_dist, min_value=1,
                        step=1, key=f"pd_{i}",
                    )
                with col2:
                    eros = st.checkbox("Эрозия", value=False, key=f"pe_{i}")
                    ei = st.number_input(
                        "Итераций", value=3, min_value=1, max_value=10,
                        key=f"pei_{i}", disabled=not eros,
                    )
                    hmax = st.checkbox("h-maxima", value=False, key=f"ph_{i}")
                    ht = st.number_input(
                        "h-порог", value=2.0, min_value=0.1, step=0.1,
                        key=f"pht_{i}", disabled=not hmax,
                    )
                per_class_config[i] = ClassConfig(
                    min_cell_area=int(area_i),
                    min_distance=int(dist_i),
                    morph_kernel_size=g_morph,
                    use_separation_erosion=eros,
                    erosion_iterations=int(ei),
                    use_h_maxima=hmax,
                    h_threshold=float(ht),
                )
                st.divider()

        st.divider()

        # ── Визуализация ─────────────────────────────────────
        st.subheader("🎨 Визуализация")
        alpha  = st.slider("Прозрачность заливки", 0.0, 1.0, 0.35, 0.05)
        thick  = st.slider("Толщина контура", 1, 5, 1)
        fscale = st.slider("Размер текста", 0.1, 1.5, 0.35, 0.05)
        crad   = st.slider("Радиус центроида", 1, 10, 3)

    config = CellCounterConfig(
        skip_class_ids=skip_ids,
        default_min_cell_area=g_area,
        default_min_distance=g_dist,
        default_morph_kernel_size=g_morph,
        per_class_config=per_class_config,
    )
    palette = {i: to_rgb(class_colors[i]) for i in range(num_classes)}
    vis_params = dict(
        overlay_alpha=alpha,
        contour_thickness=thick,
        font_scale=fscale,
        centroid_radius=crad,
    )
    return class_names, palette, config, vis_params