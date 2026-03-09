import io
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def show_results(
    counts: Dict[str, int],
    vis_img: np.ndarray,
    prefix: str = "",
) -> None:
    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.image(
            vis_img,
            caption="Визуализация сегментации",
            use_container_width=True,
        )
        buf = io.BytesIO()
        Image.fromarray(vis_img).save(buf, format="PNG")
        st.download_button(
            "📥 Скачать визуализацию (PNG)",
            buf.getvalue(),
            f"{prefix}visualization.png",
            "image/png",
            key=f"{prefix}dl_img",
        )

    with col_stats:
        st.subheader("📊 Результаты")
        total = sum(counts.values())
        for name, cnt in counts.items():
            pct = cnt / total * 100 if total > 0 else 0.0
            st.metric(name, f"{cnt} кл.", f"{pct:.1f}%")
        st.divider()
        st.metric("Итого", total)

        df = pd.DataFrame(
            [
                {
                    "Класс": k,
                    "Кол-во": v,
                    "Доля %": round(v / total * 100, 1) if total else 0,
                }
                for k, v in counts.items()
            ]
        )
        st.download_button(
            "📥 Скачать CSV",
            df.to_csv(index=False).encode("utf-8"),
            f"{prefix}results.csv",
            "text/csv",
            key=f"{prefix}dl_csv",
        )


def show_comparison(
    counts_full: Dict[str, int],
    counts_roi: Dict[str, int],
) -> None:
    st.subheader("📊 Сравнение: ROI vs Полное изображение")
    all_names = sorted(set(list(counts_full) + list(counts_roi)))
    rows = [
        {
            "Класс": n,
            "Полное изображение": counts_full.get(n, 0),
            "ROI": counts_roi.get(n, 0),
            "ROI / Полное %": (
                round(counts_roi.get(n, 0) / counts_full[n] * 100, 1)
                if counts_full.get(n, 0) > 0
                else "—"
            ),
        }
        for n in all_names
    ]
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True
    )