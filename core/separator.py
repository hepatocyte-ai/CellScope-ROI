import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import extrema
from skimage.segmentation import watershed


class WatershedSeparator:
    def __init__(
        self,
        min_distance: int = 10,
        h_threshold: float = 2.0,
        use_h_maxima: bool = False,
    ):
        self.min_distance = min_distance
        self.h_threshold = h_threshold
        self.use_h_maxima = use_h_maxima

    def get_labeled_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        dist = ndimage.distance_transform_edt(binary_mask)

        if self.use_h_maxima:
            h_max = extrema.h_maxima(dist, h=self.h_threshold) * binary_mask
            markers, _ = ndimage.label(h_max)
            if markers.max() == 0:
                markers, _ = ndimage.label(binary_mask)
        else:
            coords = peak_local_max(
                dist,
                min_distance=self.min_distance,
                labels=binary_mask,
                threshold_rel=0.3,
            )
            if len(coords) == 0:
                labeled, _ = ndimage.label(binary_mask)
                return labeled.astype(np.int32)
            markers = np.zeros(binary_mask.shape, dtype=np.int32)
            for idx, (r, c) in enumerate(coords, 1):
                markers[r, c] = idx

        return watershed(-dist, markers, mask=binary_mask).astype(np.int32)