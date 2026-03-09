import numpy as np
import os

def sample_queries(base_vectors, frac=0.01, cap=1000, seed=42, save_path="sampled_ids.npy"):

    # if os.path.exists(save_path):
    #     print("Loading saved sampled queries...")
    #     idx = np.load(save_path)
    #     return base_vectors[idx], idx

    print("Sampling queries from training dataset...")
    np.random.seed(seed)

    n = len(base_vectors)
    k = min(int(n * frac), cap)

    idx = np.random.choice(n, k, replace=False)
    np.save(save_path, idx)

    return base_vectors[idx], idx
