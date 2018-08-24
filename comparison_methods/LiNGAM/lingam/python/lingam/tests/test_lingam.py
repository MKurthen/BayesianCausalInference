# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

from copy import deepcopy
import numpy as np

from ..lingam import _slttestperm, _sltprune, estimate

def gen_data_given_model(b, s, c, n_samples=10000, random_state=0):
    """Generate artificial data based on the given model.

    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix. 
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX. 
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables. 

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`. 
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors. 

    """
    rng = np.random.RandomState(random_state)
    n_vars = b.shape[0]

    # Check args
    assert(b.shape == (n_vars, n_vars))
    assert(s.shape == (n_vars,))
    assert(np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
    # (<1 gives subgaussian, >1 gives supergaussian)
    q = rng.rand(n_vars) * 1.1 + 0.5    
    ixs = np.where(q > 0.8)
    q[ixs] = q[ixs] + 0.4

    # Generates disturbance variables
    ss = rng.randn(n_samples, n_vars)
    ss = np.sign(ss) * (np.abs(ss)**q)

    # Normalizes the disturbance variables to have the appropriate scales
    ss = ss / np.std(ss, axis=0) * s

    # Generate the data one component at a time
    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

    # Permute variables
    p = rng.permutation(n_vars)
    xs[:, :] = xs[:, p]
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    b_[:, :] = b_[p, :]
    b_[:, :] = b_[:, p]
    c_[:] = c[p]

    return xs, b_, c_

def test_slttestperm(repeat=1):
    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.RandomState(1234)

    for i in range(repeat):
        n = rng.choice([3, 4, 5, 6, 7])

        b_i = np.vstack(
            [np.hstack((rng.randn(i), np.zeros(n - i))) for i in range(n)]
        )
        b_i_org = deepcopy(b_i)
        print('Original matrix, lower triangular')
        print(b_i)
        print()

        ixs = rng.permutation(n)
        b_i[:, :] = b_i[ixs, :]
        b_i[:, :] = b_i[:, ixs]
        print('Permutated matrix')
        print(b_i)
        print()

        p = _slttestperm(b_i)
        b_i[:, :] = b_i[p, :]
        b_i[:, :] = b_i[:, p]
        print('Should be the original matrix')
        print(b_i)
        print()

        assert(np.sum(b_i - b_i_org) == 0)

def test_sltprune(repeat=1):
    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.RandomState(1234)

    for i in range(repeat):
        n = rng.choice([3, 4, 5, 6, 7])
        b_i = np.vstack(
            [np.hstack((rng.randn(i), np.zeros(n - i))) for i in range(n)]
        ) + 0.001 * rng.randn(n, n)
        b_i_org = deepcopy(b_i)
        print('Original matrix, approximately lower triangular')
        print(b_i)
        print()

        ixs = rng.permutation(n)
        b_i[:, :] = b_i[ixs, :]
        b_i[:, :] = b_i[:, ixs]
        print('Permutated matrix')
        print(b_i)
        print()

        b_opt, p = _sltprune(b_i)
        print('Should be the original matrix')
        print(b_opt)
        print()

        assert(np.sum(b_opt - b_i_org) == 0)

def test_gen_data_given_model(plot=False):
    b = np.array([[0.0, 0.0, 0.0], 
                  [1.0, 0.0, 0.0], 
                  [2.0, 3.0, 0.0]])
    s = np.array([3.0, 2.0, 1.0])
    c = np.array([5.0, 6.0, 7.0])

    xs, b_, c_ = gen_data_given_model(b, s, c)

    if plot:
        import seaborn as sns
        import pandas as pd
        df = pd.DataFrame({'x1': xs[:, 0], 'x2': xs[:, 1], 'x3': xs[:, 2]})
        sns.pairplot(df)

def test_ica(plot=False):
    b = np.array([[0.0, 0.0, 0.0], 
                  [1.0, 0.0, 0.0], 
                  [2.0, 3.0, 0.0]])
    s = np.array([3.0, 2.0, 1.0])
    c = np.array([5.0, 6.0, 7.0])

    xs, b_, c_ = gen_data_given_model(b, s, c, n_samples=10000)

    from sklearn.decomposition import FastICA
    ica = FastICA(random_state=0).fit(xs)
    xs_ = ica.transform(xs).dot(ica.mixing_.T)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xs)
        plt.ylim(-60, 60)
        plt.figure()
        plt.plot(xs_)
        plt.title('ica.transform(xs).dot(ica.mixing_.T)')
        plt.ylim(-60, 60)

def test_estimate():
    b = np.array([[0.0, 0.0, 0.0], 
                  [1.0, 0.0, 0.0], 
                  [-2.0, 3.0, 0.0]])
    s = np.array([5.0, 6.0, 8.0])
    c = np.array([3.0, 2.0, 1.0])

    xs, b_, c_ = gen_data_given_model(b, s, c, n_samples=200)

    b_est = estimate(xs)

    print(b_)
    print(b_est)
