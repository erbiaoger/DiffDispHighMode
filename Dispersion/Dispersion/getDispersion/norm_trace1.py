import numpy as np

def norm_trace(data_in):
    """
    Normalize each column of the input matrix by dividing each column 
    by its maximum absolute value.

    Args:
      data_in: Input data matrix (2D NumPy array).

    Returns:
      Normalized data matrix.
    """
    data_out = np.zeros_like(data_in)
    ng = data_in.shape[1]

    for k in range(ng):
        col_max = np.max(np.abs(data_in[:, k]))
        if col_max != 0:  # To avoid division by zero
            data_out[:, k] = data_in[:, k] / col_max
        else:
            data_out[:, k] = data_in[:, k]

    return data_out

# Example usage
# data_in = ... # Input data matrix (2D NumPy array)
# data_out = norm_trace(data_in)
