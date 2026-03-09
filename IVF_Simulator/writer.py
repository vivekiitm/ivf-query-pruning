import numpy as np
import os

DTYPE = np.float32
FEATURE_DIM = 12   # 11 features + recall label

class TrainingDataWriter:

    def __init__(self, filepath):
        self.filepath = filepath
        self.f = open(filepath, "ab")
        self.rows = 0

    def write(self, arr):
        if arr.dtype != DTYPE:
            arr = arr.astype(DTYPE)
        arr.tofile(self.f)
        self.rows += len(arr)

    def close(self):
        self.f.close()
        size = os.path.getsize(self.filepath) / (1024*1024)
        print(f"\nWritten rows: {self.rows}")
        print(f"Dataset size: {size:.2f} MB")
