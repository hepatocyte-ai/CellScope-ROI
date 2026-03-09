from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.measure import regionprops


class SegmentationVisualizer:
    def __init__(
        self,
        class_mapping: Dict[int, str],
        palette: Dict[int, Tuple[float, float, float]],
        overlay_alpha: float = 0.35,
        contour_thickness: int = 1,
        font_scale: float = 0.35,
        centroid_radius: int = 3,
    ):
        self.class_mapping = class_mapping
        self.palette = palette
        self.overlay_alpha = overlay_alpha
        self.contour_thickness = contour_thickness
        self.font_scale = font_scale
        self.centroid_radius = centroid_radius

    def _bgr(self, cid: int) -> Tuple[int, int, int]:
        r, g, b = (int(c * 255) for c in self.palette.get(cid, (0.5, 0.5, 0.5)))
        return b, g, r

    def draw(
        self,
        original_image: np.ndarray,
        labeled_masks: Dict[int, np.ndarray],
        counts: Dict[str, int],
    ) -> np.ndarray:
        if original_image.ndim == 2:
            canvas = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 4:
            canvas = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
        else:
            canvas = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # Полупрозрачная заливка
        for cid, labeled in labeled_masks.items():
            color = self._bgr(cid)
            overlay = canvas.copy()
            overlay[labeled > 0] = color
            cv2.addWeighted(
                overlay, self.overlay_alpha,
                canvas, 1 - self.overlay_alpha,
                0, canvas,
            )

        # Контуры, центроиды, порядковые номера
        for cid, labeled in labeled_masks.items():
            color = self._bgr(cid)
            for region in regionprops(labeled):
                bin_r = (labeled == region.label).astype(np.uint8)
                ctrs, _ = cv2.findContours(
                    bin_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, ctrs, -1, color, self.contour_thickness)
                cy, cx = int(region.centroid[0]), int(region.centroid[1])
                cv2.circle(canvas, (cx, cy), self.centroid_radius, color, -1)
                cv2.putText(
                    canvas, str(region.label), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)