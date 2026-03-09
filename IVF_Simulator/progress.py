import os

def load_progress(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return int(f.read().strip())

def save_progress(path, qid):
    with open(path, "w") as f:
        f.write(str(qid))
