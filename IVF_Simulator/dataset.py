import numpy as np

def load_vectors(path):
    """
    Load vectors from CSV -> contiguous float32 numpy array.
    """
    print("Loading:", path)
    data = np.loadtxt(path, delimiter=',', dtype=np.float32)
    return np.ascontiguousarray(data)


def sample_training_queries(train_vectors, frac=0.01, cap=1000, seed=42):
    """
    Sample 1% training queries (max 1000) deterministically.
    """
    np.random.seed(seed)
    n = len(train_vectors)
    k = min(int(n * frac), cap)
    idx = np.random.choice(n, k, replace=False)
    return train_vectors[idx], idx
