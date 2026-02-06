"""Model zoo for dispersion training."""

from .advanced import FrequencySequenceModel, GraphCurveModel, soft_argmax_velocities

__all__ = [
    "FrequencySequenceModel",
    "GraphCurveModel",
    "soft_argmax_velocities",
]
