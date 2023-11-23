import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class KMeans(object):
    
    def __init__(self, norm, pca, idx, index, marginal):
        self.norm = norm
        self.pca = pca
        self.idx = idx
        self.index = index
        self.marginal = marginal
    
    def clustering(self, features):
        data = features
        if self.norm in ['l2', 'l1']:
            data = normalize(data, norm=self.norm, axis=1)
        data = self.pca.transform(data)[:, :self.idx+1]
        data = data.astype(np.float32)
        _, labels = self.index.search(data, 1)
        labels = labels.reshape(-1)
        return labels


def cluster_feat(features, num_clusters,
                 norm='none', whiten=True,
                 pca_max_data=-1,
                 explained_variance=0.9,
                 num_redo=5, max_iter=500, seed=0):
    assert 0 < explained_variance < 1
    assert norm in ['none', 'l2', 'l1', None]
    data1 = features
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed+5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints')
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    data1 = pca.transform(data1)[:, :idx+1]
    # Cluster
    data1 = data1.astype(np.float32)
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          nredo=num_redo, update_index=True, seed=seed+2, min_points_per_centroid=50)
    kmeans.train(data1)
    index = kmeans.index
    _, labels = index.search(data1, 1)
    
    # Drop clusters with low frequency
    ids, counts = np.unique(labels, return_counts=True)
    to_remove = ids[counts < 50]
    if len(to_remove) > 0:
        index.remove_ids(to_remove)
        _, labels = index.search(data1, 1)
    
    _, counts = np.unique(labels, return_counts=True)
    cluster = KMeans(norm, pca, idx, index, counts/np.sum(counts))
    return labels.reshape(-1), cluster