import numpy as np


def get_count_vector(x, grid_coordinates, return_indices=False):
        """
        get the count vector k

        Parameters:
        ----------
        x : numpy array of observations
        grid_coordinates : numpy array of the bin means
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


def get_diff_matrix(N):
    """
    quick hack to get the matrix corresponding to numerical differentiation
        (finite differences)

    Parameters:
    ----------
    N : int, number of points
    """
    matrix = np.zeros((N, N))
    for i in range(N):
        left = np.zeros(N)
        left[i] = 1
        for j in range(N):
            right = np.zeros(N)
            right[j] = 1
            matrix[i, j] = np.gradient(left)@np.gradient(right)
    return matrix
                                                    


def remove_duplicates(x, y):
    u, unique_x_indices = np.unique(x, return_index=True)
    x, y = x[unique_x_indices], y[unique_x_indices]
    u, unique_y_indices = np.unique(y, return_index=True)
    x, y = x[unique_y_indices], y[unique_y_indices]
    return x, y
