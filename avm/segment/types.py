from typing import Literal, TypeAlias

import numpy as np

SegmentModelKey = Literal["sam3", "sam3-finetuned"]
SegmentConfig: TypeAlias = dict[str, float | int]
InstanceMasks: TypeAlias = list[np.ndarray]
InstanceScores: TypeAlias = list[float]
SegmentInstances: TypeAlias = tuple[InstanceMasks, InstanceScores]
MergedMask: TypeAlias = np.ndarray | None
