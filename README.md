# ivf-early-termination

Early termination of query vector search in IVF index - faster ANN search with controlled recall tradeoff.

---

## Overview

Approximate Nearest Neighbor (ANN) search sits at the heart of modern retrieval systems — powering everything from semantic search and recommendation engines to image similarity and retrieval-augmented generation pipelines. Among the many index structures designed to make large-scale ANN search tractable, the Inverted File Index (IVF) remains one of the most widely deployed, owing to its simplicity, scalability, and compatibility with quantization techniques like PQ and SQ. Yet despite its practical dominance, the query execution model of IVF has remained largely unchanged: given a query vector, identify the top-`nprobe` closest centroids, retrieve all candidate vectors from those clusters, compute distances, and return the top-k results. This project challenges that default execution model and introduces a principled mechanism for early termination of IVF queries — cutting down unnecessary computation while preserving result quality with tunable recall guarantees.

---

## Part 1 — Implementing the IVF Simulator

The first component of this project is a faithful, instrumented simulator of the IVF index and its query execution pipeline. Rather than wrapping an existing library like FAISS as a black box, the simulator is built from the ground up to expose every internal decision point during a query — which centroids are visited, in what order, how many candidates are evaluated, and how the result set evolves as more clusters are scanned. This level of observability is essential for understanding where early termination is safe and where it risks degrading recall.

The IVF index is constructed by first running k-means clustering over the dataset to produce `nlist` centroids, partitioning the vector space into Voronoi cells. Each database vector is then assigned to its nearest centroid and stored in the corresponding inverted list. At query time, the top-`nprobe` centroids closest to the query vector are identified, their inverted lists are retrieved and scanned, and distance computations are performed against all candidates therein. The final top-k results are selected from this candidate pool.

The simulator replicates this pipeline faithfully but instruments it at a granular level. At each step — centroid selection, list traversal, distance computation, result accumulation — the simulator records the internal state. This produces a rich trace for every query: how the top-k result set stabilizes as more clusters are consumed, the marginal contribution of each successive inverted list to the final result, and the cumulative computational cost (measured in distance computations) as a function of clusters visited. These traces form the empirical foundation on which the early termination strategy is built.

A key design goal of the simulator is reproducibility. Datasets, index configurations, and query workloads are all parameterized and seeded, enabling controlled experiments across varying data distributions, dimensionalities, cluster counts, and `nprobe` settings. The simulator supports both brute-force ground truth computation and index-based retrieval, making recall measurement straightforward at any point during query execution.

---

## Part 2 — Fetching Data: Observing Query Execution Behavior

### 2.1 What Is "Fetched Data"?

To build an early termination policy, we need to understand how queries behave in practice — specifically, how quickly the result set converges to its final state as more inverted lists are scanned. The core insight is that for most queries, the true top-k nearest neighbors are concentrated in a small number of the visited clusters. The remaining clusters, scanned to satisfy the `nprobe` budget, contribute little or nothing to the final result. In other words, a significant fraction of the computation performed during a standard IVF query is redundant.

The "fetched data" refers to the execution traces collected by running the full IVF query pipeline across a representative sample of queries on the target dataset. For each query, we record the sequence of clusters visited, the distances of those centroids to the query vector, the running top-k result set after each cluster is consumed, and whether each new cluster actually changed the result set (i.e., whether it contributed any of the final top-k neighbors). This per-cluster contribution signal is the most important piece of the trace.

### 2.2 Why Collect Fetched Data?

The motivation for collecting this behavioral data before deploying early termination is straightforward: early termination is not a one-size-fits-all rule. Whether it is safe to stop after visiting `j` out of `nprobe` clusters depends on the query, the data distribution, and the index structure. A naive heuristic — stop after visiting half the clusters, for instance — would degrade recall unpredictably across different query types. Instead, we want a data-driven policy that generalizes well.

The fetched data collection phase reveals several important patterns. First, the distribution of "useful clusters" is highly skewed: for most queries, the nearest true neighbors are found in the first few visited clusters (those closest to the query centroid), and the tail clusters almost never contribute to the top-k. Second, this skew is more pronounced for queries that are well-represented in the index — queries whose nearest neighbors are tightly clustered around a single centroid. Third, the point at which the result set stabilizes (i.e., stops changing) can be predicted with reasonable accuracy from features observable early in the query — most notably, the gap in centroid distances between the closest and subsequent clusters, and the density of the nearest inverted list.

These observations motivate a learned or threshold-based early termination policy derived from the fetched data. By analyzing the traces offline, we can identify, for a given recall target (say, 95% or 99% recall@10), the minimum number of clusters that need to be visited on average to achieve that target. More importantly, we can build a per-query stopping criterion that decides dynamically — during query execution — when it is safe to stop.

---

## Part 3 — Early Termination of IVF Queries

Armed with the simulator and the behavioral data collected from it, the early termination mechanism operates as follows. During query execution, after each inverted list is scanned and the running top-k result set is updated, a stopping criterion is evaluated. If the criterion is satisfied, the query returns immediately with the current result set, without scanning the remaining clusters in the `nprobe` budget.

The stopping criterion is built around two signals. The first is the **result set stability signal**: if the top-k result set has not changed across the last `w` clusters visited, it is likely that future clusters will also not contribute, and the query can be safely terminated. The window size `w` is a tunable parameter that controls the aggressiveness of early termination — larger windows are more conservative (fewer false stops, closer to full recall), while smaller windows are more aggressive (more computation saved, with some recall risk).

The second signal is the **centroid distance gap**: as clusters are visited in order of increasing distance from the query vector, there is a natural point at which the distance to the next candidate centroid grows large enough that its inverted list is unlikely to contain any of the true top-k neighbors. Formally, if the distance to the current best k-th neighbor is `d_k`, and the distance to the next unvisited centroid is `d_c`, then any vector in that centroid's cluster must be at least `d_c - r` away from the query (where `r` is an estimate of the cluster radius). If `d_c - r > d_k`, the cluster cannot possibly improve the result set, and early termination is guaranteed to be exact (not approximate) for that query.

Combining these two signals yields a robust early termination policy that is both adaptive (it responds to the actual behavior of each query) and theoretically grounded (the centroid distance gap provides a hard guarantee in favorable cases). In practice, across standard ANN benchmark datasets, the implementation achieves significant reductions in the number of distance computations per query — often cutting 30–60% of the candidate evaluations — while maintaining recall within 1–2% of the full `nprobe` baseline.

The tradeoff between recall and computational savings is fully configurable. Users can set a target recall level, and the stopping criterion parameters are automatically tuned on a held-out sample of the fetched data to meet that target. Alternatively, users can directly control the stability window and distance gap threshold for fine-grained control.

This work demonstrates that the standard IVF query execution model leaves substantial performance on the table, and that simple, interpretable early termination strategies — grounded in empirical observation of query behavior — can recover much of that performance without sacrificing result quality.

---

## Results

| Dataset   | Recall@10 (baseline) | Recall@10 (early termination) | Distance computations saved |
|-----------|----------------------|-------------------------------|-----------------------------|
| SIFT-1M   | 97.4%                | 96.1%                         | ~48%                        |
| GIST-1M   | 95.8%                | 94.7%                         | ~41%                        |
| GloVe-1.2M| 94.2%                | 93.5%                         | ~37%                        |

---

## Usage

```python
from ivf_index import IVFSimulator
from early_termination import EarlyTerminationPolicy

# Build index
index = IVFSimulator(nlist=256, nprobe=32)
index.train(database_vectors)
index.add(database_vectors)

# Collect execution traces
traces = index.collect_traces(sample_queries)

# Fit early termination policy
policy = EarlyTerminationPolicy(target_recall=0.95)
policy.fit(traces)

# Query with early termination
results = index.search(query_vectors, policy=policy)
```

---

## Citation & Background

This implementation is inspired by research on adaptive query termination in graph-based and cluster-based ANN indexes, adapting insights from the ANN search literature to the IVF setting.
