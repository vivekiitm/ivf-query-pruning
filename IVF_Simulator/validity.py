import pandas as pd
df = pd.read_csv("darth_train.csv")

print(df["recall"].min(), df["recall"].max())

import pandas as pd

df = pd.read_csv("darth_train.csv")
def read_fvecs(path):
    import numpy as np

    with open(path, "rb") as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        raw = np.fromfile(f, dtype=np.float32)

    raw = raw.reshape(-1, dim + 1)
    vectors = raw[:, 1:]

    return np.ascontiguousarray(vectors)

# IMPORTANT: reload xb
xb = read_fvecs("siftsmall_base.fvecs")

print("xb shape:", xb.shape)


import os
import numpy as np
# IMPORTANT: reload xb
xb = read_fvecs("siftsmall_base.fvecs")

print("xb shape:", xb.shape)

import numpy as np

# count exact duplicates
_, counts = np.unique(xb, axis=0, return_counts=True)

print("total vectors:", len(xb))
print("unique vectors:", len(counts))
print("duplicates:", np.sum(counts > 1))
print("max duplicate frequency:", counts.max())


import pandas as pd
import matplotlib.pyplot as plt


plt.scatter(df["nstep"], df["recall"])
plt.xlabel("Clusters scanned")
plt.ylabel("Recall")
plt.show()

print((df["closestNN"] <= df["furthestNN"]).all())

import pandas as pd

df = pd.read_csv("darth_train.csv")

violations = []
query_id = 0
start = 0

# find where a new query starts (ndis resets to 1)
query_starts = df.index[df["ndis"] == 1].tolist()
query_starts.append(len(df))   # sentinel for last query

for i in range(len(query_starts) - 1):

    s = query_starts[i]
    e = query_starts[i + 1]

    g = df.iloc[s:e].sort_values("ndis")

    # compute change in recall
    recall_drop = g["recall"].diff() < -1e-9

    if recall_drop.any():
        violations.append(i)

print("Total queries:", len(query_starts) - 1)
print("Queries where recall decreased:", len(violations))
print("First few violating queries:", violations[:10])
