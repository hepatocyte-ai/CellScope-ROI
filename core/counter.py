from typing import Dict, Optional, Tuple

import numpy as np

from core.config import CellCounterConfig
from core.preprocessor import MaskPreprocessor
from core.separator import WatershedSeparator


class CellCounter:
    def __init__(
        self,
        class_mapping: Dict[int, str],
        config: Optional[CellCounterConfig] = None,
    ):
        self.class_mapping = class_mapping
        self.config = config or CellCounterConfig()
        self._pre = MaskPreprocessor()

    def count(
        self, mask: np.ndarray
    ) -> Tuple[Dict[str, int], Dict[int, np.ndarray]]:
        counts: Dict[str, int] = {}
        labeled_masks: Dict[int, np.ndarray] = {}

        for class_id, name in self.class_mapping.items():
            if class_id in self.config.skip_class_ids:
                continue

            cfg = self.config.get_class_config(class_id)
            binary = (mask == class_id).astype(np.uint8)
            cleaned = self._pre.clean(binary, cfg.morph_kernel_size)
            cleaned = self._pre.remove_small(cleaned, cfg.min_cell_area)

            if cleaned.sum() == 0:
                counts[name] = 0
                labeled_masks[class_id] = np.zeros_like(binary, dtype=np.int32)
                continue

            if cfg.use_separation_erosion:
                cleaned = self._pre.separate_touching(
                    cleaned, cfg.erosion_iterations
                )

            sep = WatershedSeparator(
                cfg.min_distance, cfg.h_threshold, cfg.use_h_maxima
            )
            labeled = sep.get_labeled_mask(cleaned)
            counts[name] = int(np.sum(np.unique(labeled) > 0))
            labeled_masks[class_id] = labeled

        return counts, labeled_masks