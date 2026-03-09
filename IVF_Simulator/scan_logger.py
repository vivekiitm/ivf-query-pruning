import numpy as np
import faiss
from features import nn_stats


class ScanLogger:

    def __init__(self, index, base_vectors, groundtruth, k_values):
        self.index = index
        self.base = base_vectors
        self.gt = groundtruth

        self.k_values = k_values
        self.max_k = max(k_values)

        self.centroids = self._get_all_centroids()
        self.total_vectors = len(self.base)
        self.nlist = self.index.nlist
        self.top_m = int(np.clip(0.05 * self.nlist, 8, 100))  # dynamic neighbourhood size

    # ---------- recall@k ----------
    def recall_at_k(self, found_ids, true_ids, k):
        if len(found_ids) == 0:
            return 0.0
        return 100.0 * len(found_ids.intersection(true_ids[:k])) / k

    # ---------- get IVF centroids ----------
    def _get_all_centroids(self):
        quantizer = self.index.quantizer
        d = self.base.shape[1]
        nlist = self.index.nlist

        centroids = np.zeros((nlist, d), dtype=np.float32)
        for i in range(nlist):
            centroids[i] = quantizer.reconstruct(i)

        return centroids

    # ---------- raw feature row ----------
    def _build_raw_row(
        self,
        query_id,
        k,
        step,
        ndis,
        ninsert,
        current_centroid_dist,
        current_list_size,
        # centroid geometry — all computed from full centroid dist distribution
        first_centroid_dist,
        d1, d2, d3,
        margin1, margin2,
        # top-16 stats (local neighbourhood)
        cdist_top16_mean, cdist_top16_std, peaked,
        # full centroid dist stats (global query hardness)
        cdist_all_mean, cdist_all_std,
        cdist_all_p25, cdist_all_p50, cdist_all_p75, cdist_all_p90,
        cdist_all_min, cdist_all_max,
        results_sorted,
        recall
    ):
        """
        One row of raw independent features per (query, k, step).
        All derived features (slopes, diffs, rates, etc.) are computed
        later in the notebook via add_more_features().

        When heap_size < k, NN stats are computed over whatever is in
        the heap so the model can learn from the heap-filling phase too.
        """

        if len(results_sorted) == 0:
            closest  = 0.0
            furthest = 0.0
            avg = var = med = p25 = p75 = 0.0
        else:
            topk      = results_sorted[:k]
            dist_only = [x[1] for x in topk]
            closest   = dist_only[0]
            furthest  = dist_only[-1]
            avg, var, med, p25, p75 = nn_stats(dist_only)

        return [
            # --- identity ---
            query_id,               # global query counter (for groupby in notebook)
            k,
            step + 1,               # 1-indexed probe count (nstep in paper)

            # --- scan progress ---
            ndis,                   # cumulative distance computations
            ninsert,                # cumulative heap updates
            current_list_size,      # size of the cluster just scanned

            # --- centroid geometry fixed per query ---
            first_centroid_dist,
            d1, d2, d3,
            margin1, margin2,

            # --- top-16 centroid stats (local neighbourhood shape) ---
            cdist_top16_mean, cdist_top16_std, peaked,

            # --- full centroid dist stats (global query hardness) ---
            cdist_all_mean, cdist_all_std,
            cdist_all_p25, cdist_all_p50, cdist_all_p75, cdist_all_p90,
            cdist_all_min, cdist_all_max,

            # --- current centroid ---
            current_centroid_dist,

            # --- heap state ---
            closest,                # closestNN
            furthest,               # furthestNN / kth distance

            # --- heap distribution ---
            avg, var, med, p25, p75,

            # --- label ---
            recall
        ]

    # ---------- main scan ----------
    def scan_query(self, q, qi, dbid, query_id):
        """
        Scans one query through the IVF index and returns a 2D array
        of raw per-(k, step) feature rows.

        Parameters
        ----------
        q          : query vector
        qi         : index of query in the sampled batch
        dbid       : database id of this query (for leave-one-out)
        query_id   : global monotonic query counter for notebook groupby
        """

        # ---------- centroid distances (fixed for this query) ----------
        centroid_dists   = np.linalg.norm(self.centroids - q, axis=1)
        ordered_clusters = np.argsort(centroid_dists)

        first_centroid_dist = centroid_dists[ordered_clusters[0]]

        # top-M: local neighbourhood shape
        # M = ~5% of clusters, clamped to [8, 100] — computed once in __init__
        TOP_M = self.top_m
        topM           = centroid_dists[ordered_clusters[:TOP_M]]
        d1, d2, d3     = topM[0], topM[1], topM[2]
        margin1        = d2 - d1
        margin2        = d3 - d2
        cdist_top16_mean = float(np.mean(topM))
        cdist_top16_std  = float(np.std(topM))
        peaked           = (cdist_top16_mean - d1) / (cdist_top16_mean + 1e-8)

        # full distribution: global query hardness signal
        cdist_all_mean = float(np.mean(centroid_dists))
        cdist_all_std  = float(np.std(centroid_dists))
        cdist_all_min  = float(centroid_dists.min())
        cdist_all_max  = float(centroid_dists.max())
        cdist_all_p25  = float(np.percentile(centroid_dists, 25))
        cdist_all_p50  = float(np.percentile(centroid_dists, 50))
        cdist_all_p75  = float(np.percentile(centroid_dists, 75))
        cdist_all_p90  = float(np.percentile(centroid_dists, 90))

        # ---------- scan state ----------
        results  = []   # heap: list of (id, dist)
        features = []

        ndis    = 0
        ninsert = 0

        converged        = False
        converge_step    = None
        tail_limit_step  = None

        for step, cid in enumerate(ordered_clusters):

            cid = int(cid)

            list_size = self.index.invlists.list_size(cid)
            if list_size == 0:
                continue

            current_centroid_dist = float(centroid_dists[cid])

            ids_ptr = self.index.invlists.get_ids(cid)
            ids     = faiss.rev_swig_ptr(ids_ptr, list_size)

            vecs  = self.base[ids]
            dists = np.linalg.norm(vecs - q, axis=1)
            ndis += len(dists)

            # ---------- maintain heap ----------
            for vid, dist in zip(ids, dists):
                if int(vid) == dbid:
                    continue

                if len(results) < self.max_k:
                    results.append((int(vid), float(dist)))
                    ninsert += 1
                else:
                    worst_idx = np.argmax([x[1] for x in results])
                    if dist < results[worst_idx][1]:
                        results[worst_idx] = (int(vid), float(dist))
                        ninsert += 1

            # always log — heap-filling phase is informative too
            results_sorted = sorted(results, key=lambda x: x[1])
            heap_size      = len(results_sorted)

            # ---------- log one row per k value ----------
            for k in self.k_values:

                # recall is 0 if heap doesn't have k entries yet
                if heap_size >= k:
                    ids_k  = {x[0] for x in results_sorted[:k]}
                    recall = self.recall_at_k(ids_k, self.gt[qi], k)
                else:
                    recall = 0.0

                row = self._build_raw_row(
                    query_id,
                    k,
                    step,
                    ndis,
                    ninsert,
                    current_centroid_dist,
                    list_size,
                    first_centroid_dist,
                    d1, d2, d3,
                    margin1, margin2,
                    cdist_top16_mean, cdist_top16_std, peaked,
                    cdist_all_mean, cdist_all_std,
                    cdist_all_p25, cdist_all_p50, cdist_all_p75, cdist_all_p90,
                    cdist_all_min, cdist_all_max,
                    results_sorted,
                    recall
                )
                features.append(row)

            # ---------- convergence detection ----------
            if not converged and heap_size >= self.max_k:
                ids_max = {x[0] for x in results_sorted[:self.max_k]}
                r_max   = self.recall_at_k(ids_max, self.gt[qi], self.max_k)

                if r_max >= 100.0:
                    converged       = True
                    converge_step   = step
                    tail            = max(25, int(0.10 * converge_step))
                    tail            = min(tail, 150)
                    tail_limit_step = converge_step + tail

            if converged and step > tail_limit_step:
                break

        return np.array(features, dtype=np.float32)

    # ---------- column names (keep in sync with _build_raw_row) ----------
    @staticmethod
    def column_names():
        return [
            # identity
            "query_id", "k", "nstep",
            # scan progress
            "ndis", "ninsert", "list_size",
            # centroid geometry (fixed per query)
            "first_centroid_dist",
            "d1", "d2", "d3",
            "margin1", "margin2",
            # top-16 centroid stats
            "cdist_top16_mean", "cdist_top16_std", "peaked",
            # full centroid dist stats (query hardness)
            "cdist_all_mean", "cdist_all_std",
            "cdist_all_p25", "cdist_all_p50", "cdist_all_p75", "cdist_all_p90",
            "cdist_all_min", "cdist_all_max",
            # current centroid
            "current_centroid_dist",
            # heap state
            "closestNN", "furthestNN",
            # heap distribution
            "avg", "var", "med", "p25", "p75",
            # label
            "recall"
        ]
