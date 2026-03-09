import numpy as np

def nn_stats(distances):
    arr = np.array(distances, dtype=np.float32)

    return (
        arr.mean(),
        arr.var(),
        np.median(arr),
        np.percentile(arr, 25),
        np.percentile(arr, 75)
    )
