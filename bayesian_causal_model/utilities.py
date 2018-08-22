import numpy as np

import nifty5


def probe_operator(operator):
    """
    probe a symmetric operator in dim terations by just applying times to unit
        vectors. volume factors not included
    """
    domain = operator.domain[0]
    dim = domain.shape[0]
    operator_matrix = np.zeros((dim, dim))
    for i in range(dim):
        a = nifty5.Field(domain=domain, val=np.zeros(dim))
        a.val[i] = 1
        right = operator.times(a)
        operator_matrix[:, i] = np.array(right.val)

    return operator_matrix


def get_count_vector(x, grid_coordinates, return_indices=False):
        """
        get the count vector k

        Parameters:
        ----------
        return_indices: boolean, whether the indices of the given samples wrt
            the grid should be returned

        Returns:
        ----------
        k: numpy array
        x_indices: numpy array, optional

        """
        # get the indices which represent the closest grid points
        x_indices = np.array([
            np.abs(grid_coordinates - x[i]).argmin()
            for i in range(len(x))])
        k = np.bincount(x_indices, minlength=len(grid_coordinates))
        if return_indices:
            return (k, x_indices)
        else:
            return k

def remove_duplicates(x, y):
    u, unique_x_indices = np.unique(x, return_index=True)
    x, y = x[unique_x_indices], y[unique_x_indices]
    u, unique_y_indices = np.unique(y, return_index=True)
    x, y = x[unique_y_indices], y[unique_y_indices]
    return x, y
