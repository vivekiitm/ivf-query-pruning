import gc
from tqdm import tqdm

from fvecs import read_fvecs
from dataset import load_vectors
from sampler import sample_queries
from groundtruth import compute_groundtruth
from ivf_index import build_ivf_index
from scan_logger import ScanLogger
from csv_writer import CSVTrainingWriter


BASE_FILE = "sift_base.fvecs"
QUERY_FILE = "sift_query.fvecs"
OUTPUT_FILE = "sift_new_test_data.csv"
GT_FILE = "gt_sift_new_query.npy"

INDEX_ACCURACIES = [70, 90, 95]
INDEX_ACCURACIES = [70]
K_VALUES = list(range(10, 201, 10))


def main():

    # ---------- load dataset ----------
    # base = read_fvecs(BASE_FILE)
    base = read_fvecs(BASE_FILE)
    queries = read_fvecs(QUERY_FILE)
    print("Base:", base.shape)

    # ---------- sample queries ----------
    queries, qids = sample_queries(base, frac=1, cap=1000, save_path="sample_sift_train_query_ids.npy")
    print("Sampled queries:", len(queries))
    
    
    # ---------- ground truth ----------
    gt = compute_groundtruth(base, queries, qids, GT_FILE)


    # ---------- CSV ----------
    writer = CSVTrainingWriter(OUTPUT_FILE)

    
    # ---------- run experiments ----------
    for idx_acc in INDEX_ACCURACIES:

        print("\n====================================")
        print("Building index for accuracy:", idx_acc)
        print("====================================")

        # build index with different nlist
        index = build_ivf_index(base, accuracy=idx_acc)

        logger = ScanLogger(
            index,
            base,
            gt,
            K_VALUES
        )

        print("Running queries...")
        query_id = 0
        for qi in tqdm(range(len(queries))):
            query_id += 1
            feats = logger.scan_query(queries[qi], qi, qids[qi], query_id)

            writer.write(feats, idx_acc)

            del feats
            gc.collect()

    writer.close()


if __name__ == "__main__":
    main()
