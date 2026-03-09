from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class ClassConfig:
    """Гиперпараметры алгоритма для одного класса."""
    min_cell_area: int = 50
    min_distance: int = 10
    morph_kernel_size: int = 3
    use_separation_erosion: bool = False
    erosion_iterations: int = 3
    h_threshold: float = 2.0
    use_h_maxima: bool = False


@dataclass
class CellCounterConfig:
    skip_class_ids: Set[int] = field(default_factory=lambda: {0})
    default_min_cell_area: int = 100
    default_min_distance: int = 15
    default_morph_kernel_size: int = 3
    per_class_config: Dict[int, ClassConfig] = field(default_factory=dict)

    def get_class_config(self, class_id: int) -> ClassConfig:
        return self.per_class_config.get(
            class_id,
            ClassConfig(
                min_cell_area=self.default_min_cell_area,
                min_distance=self.default_min_distance,
                morph_kernel_size=self.default_morph_kernel_size,
            ),
        )