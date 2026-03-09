import cv2
import numpy as np
from skimage.morphology import remove_small_objects


class MaskPreprocessor:
    @staticmethod
    def clean(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
        return remove_small_objects(
            mask.astype(bool), min_size=min_area
        ).astype(np.uint8)

    @staticmethod
    def separate_touching(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.erode(mask, kernel, iterations=iterations)