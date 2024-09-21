import numpy as np

def normalize_data(data):
    # find the maximum value of each column
    max_values = np.max(np.abs(data), axis=0)
    data_norm  = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        data_norm[i, :] = data[i, :] / max_values
    # return the normalized data
    return data_norm
