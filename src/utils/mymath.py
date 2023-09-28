import numpy as np
from scipy.stats import entropy


def a_clip(v, g, r=1.0):
    return min(r, 1 - abs(v - g) / g)

def jsdiv(p, q):
    return (entropy(p, p + q, base=2) + entropy(q, p + q, base=2)) / 2

def grid_cnt(data, ranges, n_grids=10, normalize=True):
    eps = 1e-10
    d = data.shape[1]
    res = np.zeros([n_grids] * d)
    itvs = (ranges[:, 1] - ranges[:, 0]) * ((1 + eps) / n_grids)

    for item in data:
        indexes = tuple((item // itvs))
        res[indexes] = res[indexes] + 1
    if normalize:
        res /= res.size
    return res

def crowdivs(distmat, inner=False):
    mat = np.array(distmat)
    m, n = mat
    if inner and m == n:
        vmax = np.max(mat)
        return np.min(mat + np.identity(n) * vmax, axis=1).sum() / n ** 0.5
    return np.min(mat, axis=1).sum() / m

def lpdist_mat(X, Y, p=2):
    diff = np.abs(np.expand_dims(X, axis=1) - np.expand_dims(Y, axis=0))
    distance_matrix = np.sum(diff ** p, axis=-1) ** (1 / p)
    return distance_matrix

def linfdist_mat(X, Y):
    diff = np.abs(np.expand_dims(X, axis=1) - np.expand_dims(Y, axis=0))
    distance_matrix = np.max(diff, axis=-1)
    return distance_matrix

if __name__ == '__main__':
    x = [[1, 0], [0, 1], [3, -1]]
    print(lpdist_mat(x, x, 1))
    print(lpdist_mat(x, x, 2))
    print(linfdist_mat(x, x))
