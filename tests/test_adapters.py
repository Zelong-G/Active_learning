import numpy as np

from active_learning.adapters import from_torchvision_output


def test_torchvision_style_output_filters_low_confidence_instances() -> None:
    output = {
        "boxes": np.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=float),
        "labels": np.array([1, 2]),
        "scores": np.array([0.9, 0.1]),
        "masks": np.ones((2, 1, 4, 4), dtype=float),
    }
    result = from_torchvision_output(output, score_threshold=0.5)
    assert len(result.instances) == 1
    assert result.instances[0].label == 1
    assert result.instances[0].mask.shape == (4, 4)
