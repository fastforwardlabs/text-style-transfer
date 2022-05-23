import numpy as np
from pyemd import emd


def calculate_emd(input_dist, output_dist, target_class_idx):
    """
    Calculate the direction-corrected Earth Mover's Distance (aka Wasserstein distance)
    between two distributions of equal length. Here we penalize the EMD score if
    the output text style moved further away from the target style.

    Reference: https://github.com/passeul/style-transfer-model-evaluation/blob/master/code/style_transfer_intensity.py

    Args:
        input_dist (list) - probabilities assigned to the style classes
            from the input text to style transfer model
        output_dist (list) - probabilities assigned to the style classes
            from the outut text of the style transfer model

    Returns:
        emd (float) - Earth Movers Distance between the two distributions

    """

    N = len(input_dist)
    distance_matrix = np.ones((N, N))
    dist = emd(np.array(input_dist), np.array(output_dist), distance_matrix)

    transfer_direction_correction = (
        1 if output_dist[target_class_idx] >= input_dist[target_class_idx] else -1
    )

    return round(dist * transfer_direction_correction, 4)
