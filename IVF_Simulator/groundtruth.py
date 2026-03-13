import numpy as np
import os
import faiss

GT_K = 200

def compute_groundtruth(base, queries, qids, save_file):

    if os.path.exists(save_file):
        print("Loading cached groundtruth:", save_file)
        return np.load(save_file)

    print("Computing groundtruth... (leave-one-out exact search)")

    dim = base.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base)

    # search one extra neighbor
    D, I = index.search(queries, GT_K + 1)

    gt = np.empty((len(queries), GT_K), dtype=np.int64)

    for i in range(len(queries)):
        # remove self-match (database id of this query)
        mask = I[i] != qids[i]
        filtered = I[i][mask]

        # keep only first GT_K neighbors
        gt[i] = filtered[:GT_K]

    # name issue
    #np.save(save_file, gt)

    # print("Groundtruth saved.")

    return gt
