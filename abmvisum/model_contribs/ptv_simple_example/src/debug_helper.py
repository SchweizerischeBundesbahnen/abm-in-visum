import numpy as np

def warn_NaN_in_dask_matrix(dask_matrix):
    assert len(dask_matrix.shape)==2
    matrix = dask_matrix.compute()
    for i in range(dask_matrix.shape[0]):
        for j in range(dask_matrix.shape[1]):
            val = matrix[i][j]
            assert not np.isnan(val)
            assert not np.isposinf(val)
            assert not np.isneginf(val)
