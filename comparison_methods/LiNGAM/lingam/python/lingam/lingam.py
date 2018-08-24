# -*- coding: utf-8 -*-

"""Estimate LiNGAM model. 
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

from copy import deepcopy
from munkres import Munkres
import numpy as np
from sklearn.decomposition import FastICA

def _nzdiaghungarian(w):
    """Permurate rows of w to minimize sum(diag(1 / w)). 
    """
    assert(0 < np.min(np.abs(w)))

    w_ = 1 / np.abs(w)
    m = Munkres()
    ixs = np.vstack(m.compute(deepcopy(w_)))

    # Sort by row indices
    ixs = ixs[np.argsort(ixs[:, 0]), :]

    # Return permutation indices
    # r-th row moves to `ixs[r]`. 
    return ixs[:, 1]

def _slttestperm(b_i):
    """Permute rows and cols of the given matrix. 
    """
    n = b_i.shape[0]
    remnodes = np.arange(n)
    b_rem = deepcopy(b_i)
    p = list()

    for i in range(n):
        # Find the row with all zeros
        ixs = np.where(np.sum(np.abs(b_rem), axis=1) < 1e-12)[0]

        if len(ixs) == 0:
            # If empty, return None
            return None
        else:
            # If more than one, rbitrarily select the first
            ix = ixs[0]
            p.append(remnodes[ix])

            # Remove the node (and the row and column from b_rem)
            remnodes = np.hstack((remnodes[:ix], remnodes[(ix + 1):]))
            ixs = np.hstack((np.arange(ix), np.arange(ix + 1, len(b_rem))))
            b_rem = b_rem[ixs, :]
            b_rem = b_rem[:, ixs]

    return np.array(p)

def _sltprune(b):
    """Finds an permutation for approximate lower triangularization. 
    """
    n = b.shape[0]
    assert(b.shape == (n, n))

    # Sort the elements of b
    ixs = np.argsort(np.abs(b).ravel())

    for i in range(int(n * (n + 1) / 2) - 1, (n * n) - 1):
        b_i = deepcopy(b)

        # NOTE: `ravel()` returns a view of the given array
        b_i.ravel()[ixs[:i]] = 0

        ixs_perm = _slttestperm(b_i)

        if ixs_perm is not None:
            b_opt = deepcopy(b)
            b_opt = b_opt[ixs_perm, :]
            b_opt = b_opt[:, ixs_perm]
            return b_opt, ixs_perm

    raise ValueError("Failed to do lower triangularization.")

def estimate(xs, random_state=1234):
    """Estimate LiNGAM model. 

    Parameters
    ----------
    xs : numpy.ndarray, shape=(n_samples, n_features)
        Data matrix.
    seed : int
        The seed of random number generator used in the function. 

    Returns
    -------
    b_est : numpy.ndarray, shape=(n_features, n_features)
        Estimated coefficient matrix with LiNGAM. This can be transformed to 
        a strictly lower triangular matrix by permuting rows and columns, 
        implying that the directed graph represented by b_est is acyclic. 
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX. 
    """
    n_samples, n_features = xs.shape

    ica = FastICA(random_state=random_state, max_iter=1000).fit(xs)
    w = np.linalg.pinv(ica.mixing_)
    assert(w.shape == (n_features, n_features))

    # TODO: check statistical independence of icasig
    # icasig = ica.components_

    # Permute rows of w so that np.sum(1/np.diag(w_perm)) is minimized
    # Permutation order does not make sense in the following processing, 
    # because the permutation is canceled with independent components, whose 
    # order is arbitrary. 
    ixs_perm = _nzdiaghungarian(w)
    w_perm = np.zeros_like(w)
    w_perm[ixs_perm] = w

    # Divide each row of wp by the diagonal element
    w_perm = w_perm / np.diag(w_perm)[:, np.newaxis]

    # Estimate b
    b_est = np.eye(n_features) - w_perm

    # Permute the rows and columns of b_est
    b_csl, p_csl = _sltprune(b_est)

    # Set the upper triangular to zero
    b_csl = np.tril(b_csl, -1)

    # Permute b_csl back to the original variable
    b_est = b_csl # just rename here
    b_est[p_csl, :] = deepcopy(b_est)
    b_est[:, p_csl] = deepcopy(b_est)

    return b_est
