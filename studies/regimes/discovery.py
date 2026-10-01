"""The data mining and knowledge discovery steps of the workflow of Briceno-Mena et al.
(2022) and Seghers et al. (2023), operating on arrays.

Their logic and defaults are those of the workflow, with two changes.  The first: every
random draw takes a seed, so that the study is reproducible.

    sampling_avg        sampling with averaging over fixed windows
    z_score             normalization
    dr_pca, dr_tsne, dr_pacmap
                        dimensionality reduction; PCA keeps 95% of the variance
    db_index, cluster_hdbscan
                        clustering and the Davies-Bouldin index
    sgs                 subspace greedy search (SGS) of the variables that separate
                        two clusters

The second is a change to SGS.  From the second level on, the original search extends
the current best subspace by each variable of the retained set, and that set includes
the variables already in the subspace, so a candidate can repeat one: [3, 3].  A subspace
is scored by the distance to its k-th neighbor divided by the square root of its size,
and a repeated variable scores exactly what the best single variable scored, so a new
variable wins only by raising that distance by a factor of the square root of two.  The
search therefore rarely left the first variable, and each contribution came to mean the
share of observations whose single most separating variable it is.  Here a candidate
extends the subspace only by variables not already in it, so the search goes on to the
variables that separate the clusters jointly.  Contributions differ for this reason and
are not comparable with those of the earlier applications of the workflow.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------------------------------
# normalization.py
# ------------------------------------------------------------------------------------


def z_score(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    result = np.where(std == 0, 0, (data - mean) / np.where(std == 0, 1, std))
    return result, mean, std


# ------------------------------------------------------------------------------------
# preprocessing.py
# ------------------------------------------------------------------------------------


def sampling_avg(data: np.ndarray, interval: int) -> tuple[np.ndarray, np.ndarray]:
    """Every `interval`-th row, averaged over the rows from `interval // 2` before it to
    `interval // 2 - 1` after it (four rows for an interval of five), clipped at the ends.

    Returns the averaged rows and the indices they are centered on.
    """
    interval = int(interval)
    length = len(data)
    keep_ids = list(range(0, length, interval))
    mean_itv = interval // 2
    out = np.empty((len(keep_ids), data.shape[1]))
    for idx, ki in enumerate(keep_ids):
        bg = max(0, ki - mean_itv)
        ed = min(length, ki + mean_itv)
        out[idx] = data[bg:ed].mean(axis=0)
    return out, np.asarray(keep_ids)


class KNNProcessor:
    """kNN-based denoising and data reduction."""

    def __init__(self, n_neigh=5, knn_percentage=None, search_begin_pct=0.5, rng=None):
        self.n_neigh = n_neigh
        self.knn_pct = knn_percentage
        self.search_begin_pct = search_begin_pct
        self.opt_dist = None
        self.opt_index = None
        self.sorted_knn_info = None
        self.rng = rng if rng is not None else np.random.default_rng()

    def _opt_point(self, sort_dist):
        length = len(sort_dist)
        k_alpha = (sort_dist[length - 1] - sort_dist[0]) / float(length - 1)
        lg = max(int(0.01 * length), 10)
        begin = int(self.search_begin_pct * length)
        for i in range(begin, length - lg):
            k = (sort_dist[i + lg] - sort_dist[i]) / float(lg)
            if k > k_alpha:
                opt_pct = (i + 0.5 * lg) / float(length)
                opt_dist = sort_dist[int(i + 0.5 * lg)]
                return opt_pct, opt_dist
        opt_pct = (length - lg - 1 + 0.5 * lg) / float(length)
        opt_dist = sort_dist[int(length - lg - 1 + 0.5 * lg)]
        return opt_pct, opt_dist

    def fit(self, data):
        self.raw_data = np.array(data)
        self.l_data, self.w_data = self.raw_data.shape
        neigh = NearestNeighbors(n_neighbors=self.n_neigh + 1, algorithm="auto")
        neigh.fit(self.raw_data)
        dist, _ = neigh.kneighbors(self.raw_data)
        neigh_dist = dist[:, self.n_neigh]
        neigh_index = np.arange(self.l_data)
        knn_info = np.vstack((neigh_dist, neigh_index)).T
        knn_info = knn_info[knn_info[:, 0].argsort()]
        self.sorted_knn_info = knn_info.T
        if self.knn_pct is None:
            opt_pct, self.opt_dist = self._opt_point(self.sorted_knn_info[0, :])
            self.opt_index = int(self.l_data * opt_pct)
        else:
            self.opt_index = int(self.knn_pct * self.l_data)
            self.opt_dist = self.sorted_knn_info[0, self.opt_index]

    def data_reduce(self):
        data = self.raw_data
        n = data.shape[0]
        S_index = []
        while True:
            ri = int(self.rng.random() * n)
            if not S_index:
                S_index.append(ri)
            else:
                sn = NearestNeighbors(n_neighbors=len(S_index), algorithm="auto")
                sn.fit(data[S_index, :])
                d, _ = sn.kneighbors(data[ri].reshape(1, -1))
                if d[0][-1] >= self.opt_dist:
                    S_index.append(ri)
            if len(S_index) >= self.n_neigh:
                break
        for i in range(n):
            nn = NearestNeighbors(n_neighbors=self.n_neigh, algorithm="auto")
            nn.fit(data[S_index, :])
            d, _ = nn.kneighbors(data[i].reshape(1, -1))
            if d[0][-1] >= self.opt_dist:
                S_index.append(i)
        return sorted(set(S_index))


# ------------------------------------------------------------------------------------
# dr_methods.py
# ------------------------------------------------------------------------------------


def pca_auto_components(data) -> int:
    """The number of principal components that explain 95% of the variance."""
    cv = PCA().fit(data).explained_variance_ratio_
    total = cv.sum()
    cumul = 0.0
    for i, v in enumerate(cv):
        cumul += v
        if cumul / total >= 0.95:
            return i + 1
    return len(cv)


def dr_pca(data, n_components="default"):
    if n_components == "default":
        n_components = pca_auto_components(data)
    model = PCA(int(n_components)).fit(data)
    return model.transform(data), model


def dr_tsne(data, n_components=2, seed=0):
    return TSNE(n_components, random_state=seed).fit_transform(data)


def dr_pacmap(data, n_components=2, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, seed=0):
    import pacmap
    pm = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors,
                       MN_ratio=mn_ratio, FP_ratio=fp_ratio, random_state=seed)
    return pm.fit_transform(data)


# ------------------------------------------------------------------------------------
# clustering.py
# ------------------------------------------------------------------------------------


def db_index(data, cluster_labels) -> float:
    """Davies-Bouldin index, as the reference computes it (noise counted as a cluster)."""
    cluster_labels = np.array(cluster_labels)
    unique_labels = list(set(cluster_labels))
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0
    centroids = []
    S = np.zeros(n_clusters)
    for idx, lab in enumerate(unique_labels):
        cluster_data = data[cluster_labels == lab]
        c = cluster_data.mean(axis=0)
        centroids.append(c)
        S[idx] = (np.sum((cluster_data - c) ** 2, axis=1).sum() / len(cluster_data)) ** 0.5
    M = squareform(pdist(np.array(centroids)))
    Rij = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if M[i, j] > 0:
                Rij[i, j] = (S[i] + S[j]) / M[i, j]
    return float(np.mean(np.amax(Rij, axis=1)[:-1]))


def cluster_hdbscan(data, min_cluster_size=10):
    import hdbscan
    model = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), core_dist_n_jobs=1)
    labels = model.fit_predict(data)
    return labels, db_index(data, labels)


# ------------------------------------------------------------------------------------
# contribution.py
# ------------------------------------------------------------------------------------


def _normalize_contri(contri):
    total = np.sum(contri)
    if total == 0:
        return contri
    return contri / float(total)


def _order_attributes(local_best):
    order_group = []
    max_len = len(local_best[-1]) if local_best else 0
    score_contri = np.zeros(max_len, dtype=int)
    for combo in local_best:
        for item in combo:
            if item not in order_group:
                order_group.append(item)
                score_contri[len(order_group) - 1] += 1
            else:
                idx = order_group.index(item)
                score_contri[idx] += 1
    scores = list(score_contri[:len(order_group)])
    pairs = sorted(zip(order_group, scores), key=lambda x: x[1], reverse=True)
    if pairs:
        order_group, scores = zip(*pairs)
        order_group = list(order_group)
        scores = np.array(scores, dtype=float)
        scores = scores / scores.sum()
    else:
        order_group = []
        scores = np.array([])
    return order_group, scores


def _combinations(target, data, length):
    results = []

    def _recurse(tgt, dat, ln):
        for i in range(len(dat)):
            new_tgt = tgt + [dat[i]]
            new_dat = dat[i + 1:]
            if len(new_tgt) == ln:
                results.append(new_tgt)
            elif len(new_tgt) < ln:
                _recurse(new_tgt, new_dat, ln)

    _recurse(target, data, length)
    return results


def greedy_search_contribution(nor_data, ref_indices, comp_indices, n_set=1, k=10, seed=0):
    """Subspace greedy search for the variables that distinguish two clusters.

    Returns an array of shape (2, n_vars): row 0 the variable indices, row 1 their
    contributions, which sum to one.
    """
    rng = np.random.default_rng(seed)
    ref_data = nor_data[ref_indices, :]
    comp_data = nor_data[comp_indices, :]

    # Reduce comparison data if too large
    for _ in range(2):
        if len(comp_data) >= 50:
            proc = KNNProcessor(n_neigh=1, knn_percentage=0.99, rng=rng)
            proc.fit(comp_data)
            comp_data = comp_data[proc.data_reduce(), :]
        else:
            break
    for _ in range(2):
        if len(ref_data) >= 1000:
            proc = KNNProcessor(n_neigh=1, knn_percentage=0.99, rng=rng)
            proc.fit(ref_data)
            ref_data = ref_data[proc.data_reduce(), :]
        else:
            break

    l, w = comp_data.shape
    variable_contri = np.zeros((2, w))
    variable_contri[0, :] = np.arange(w)

    for kk in range(l):
        error_nor = comp_data[kk]
        all_space = list(range(w))
        target = []
        local_best = []
        n = n_set
        while n <= len(all_space):
            # Only variables not yet in the subspace extend it; see the module docstring.
            combination = _combinations(target, [v for v in all_space if v not in target], n)
            if not combination:
                break
            scores = []
            for combo in combination:
                neigh = NearestNeighbors(n_neighbors=k)
                neigh.fit(ref_data[:, combo])
                d = neigh.kneighbors(error_nor[combo].reshape(1, -1))[0][0][k - 1]
                scores.append(d / float(np.sqrt(n)))
            best_idx = scores.index(max(scores))
            local_best.append(combination[best_idx])
            target = combination[best_idx]
            avg_score = np.mean(scores)
            good_comb = [combination[i] for i in range(len(scores)) if scores[i] >= avg_score]
            all_space = list(set(np.array(good_comb).flatten().tolist()))
            n += 1
            order_group, contri = _order_attributes(local_best)
        for j, var_idx in enumerate(order_group):
            variable_contri[1, var_idx] += contri[j]

    variable_contri[1, :] = _normalize_contri(variable_contri[1, :])
    return variable_contri


def sgs(nor_data, ref_indices, comp_indices, names, k=10, seed=0) -> dict[str, float]:
    """Contributions from `greedy_search_contribution` as name -> percent, largest first."""
    r = greedy_search_contribution(nor_data, ref_indices, comp_indices, n_set=1, k=k, seed=seed)
    contri = {names[int(r[0, i])]: 100.0 * float(r[1, i]) for i in range(r.shape[1])}
    return dict(sorted(contri.items(), key=lambda kv: -kv[1]))
