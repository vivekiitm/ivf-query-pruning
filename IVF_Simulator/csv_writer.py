import csv
import os
from scan_logger import ScanLogger

HEADER = ["index_acc"] + ScanLogger.column_names()


class CSVTrainingWriter:

    def __init__(self, filepath):
        self.filepath = filepath

        file_exists = os.path.exists(filepath)
        self.f = open(filepath, "a", newline="")
        self.writer = csv.writer(self.f)

        # write header once
        if not file_exists:
            self.writer.writerow(HEADER)

        self.rows = 0

    def write(self, arr, index_acc):
        for row in arr:
            self.writer.writerow([index_acc] + list(row))
        self.rows += len(arr)
        self.f.flush()


    def close(self):
        self.f.close()
        print("Total rows written:", self.rows)
