
import numpy as np
import itertools
import pytest
from functools import partial
from pylops.utils import dottest

from nifty_solve.operators import FinufftRealOperator, Finufft1DRealOperator, Finufft2DRealOperator, Finufft3DRealOperator, expand_to_dim, permute_mask


def design_matrix_as_is(xs, P):
    """
    Take in a set of x positions and return the Fourier design matrix.

    ## Bugs:
    - Needs comment header.
    
    ## Comments:
    - The code looks different from the paper because Python zero-indexes.
    - This could be replaced with something that makes use of finufft.
    """
    X = np.ones_like(xs).reshape(len(xs), 1)
    for j in range(1, P):
        if j % 2 == 0:
            X = np.concatenate((X, np.cos(j * xs)[:, None]), axis=1)
        else:
            X = np.concatenate((X, np.sin((j + 1) * xs)[:, None]), axis=1)
    return X



def dottest_1d_real_operator(N, P, eps=1e-10):
    x = np.linspace(-np.pi, np.pi, N)
    A = Finufft1DRealOperator(x, P, eps=eps)
    dottest(A)


def check_design_matrix_uniqueness(Op, points, P, eps=1e-10, **kwargs):
    A = Op(*points, P, eps=eps, **kwargs)
    A_dense = A.todense()
    A_unique = np.unique(A_dense, axis=1)
    if A_dense.shape != A_unique.shape:
        for i in range(A_dense.shape[1]):
            foo = np.where(np.all(A_dense == A_dense[:, [i]], axis=0))[0]
            if len(foo) > 1:
                print(i, foo, np.unravel_index(foo, A.permute.shape))
        
        assert False


def check_design_matrix_uniqueness_1d_real_operator(N, P, eps=1e-10):
    x = np.linspace(-np.pi, np.pi, N)
    check_design_matrix_uniqueness(Finufft1DRealOperator, (x, ), P, eps=eps)

def check_design_matrix_uniqueness_2d_real_operator(N, P, eps=1e-10):
    Nx, Ny = expand_to_dim(N, 2)

    x = np.random.uniform(-np.pi, np.pi, Nx)
    y = np.random.uniform(-np.pi, np.pi, Ny)
    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))

    check_design_matrix_uniqueness(Finufft2DRealOperator, (X, Y), P, eps=eps)


def check_design_matrix_uniqueness_3d_real_operator(N, P, eps=1e-10):
    Nx, Ny, Nz = expand_to_dim(N, 3)

    x = np.random.uniform(-np.pi, np.pi, Nx)
    y = np.random.uniform(-np.pi, np.pi, Ny)
    z = np.random.uniform(-np.pi, np.pi, Nz)
    X, Y, Z = map(lambda x: x.flatten(), np.meshgrid(x, y, z))

    check_design_matrix_uniqueness(Finufft3DRealOperator, (X, Y, Z), P, eps=eps)
    
    

def check_1d_real_operator_matches_design_matrix(N, P, eps=1e-10):
    x = np.linspace(-np.pi, np.pi, N)

    A = Finufft1DRealOperator(x, P, eps=1e-10)

    mode_indices = np.zeros(P, dtype=int)
    mode_indices[2::2] = np.arange(1, P//2 + (P % 2))
    mode_indices[1::2] = np.arange(P//2 + (P % 2), P)[::-1]

    A1 = design_matrix_as_is(x/2, P)
    assert np.allclose(A.todense()[:, mode_indices], A1)


def dottest_2d_real_operator(N, P, eps=1e-10):
    if isinstance(N, int):
        Nx = Ny = N
    else:
        Nx, Ny = N
    x = np.linspace(-np.pi, np.pi, Nx)
    y = np.linspace(-np.pi, np.pi, Ny)

    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))
    A = Finufft2DRealOperator(X, Y, P, eps=eps)
    dottest(A)


def dottest_3d_real_operator(N, P, eps=1e-10):
    X = np.random.uniform(-np.pi, +np.pi, N)
    Y = np.random.uniform(-np.pi, +np.pi, N)
    Z = np.random.uniform(-np.pi, +np.pi, N)
    A = Finufft3DRealOperator(X, Y, Z, P, eps=eps)
    dottest(A)

def test_incorrect_data_lengths_2d_real_operator():
    x = np.linspace(-np.pi, np.pi, 10)
    y = np.linspace(-np.pi, np.pi, 11)
    with pytest.raises(ValueError):
        Finufft2DRealOperator(x, y, 10)
    with pytest.raises(ValueError):
        Finufft2DRealOperator(y, x, 10)

def test_expand_to_dims():
    assert expand_to_dim(6, 1) == (6, )
    assert expand_to_dim(10, 3) == (10, 10, 10)
    assert expand_to_dim(1, 2) == (1, 1)
    with pytest.raises(ValueError):
        expand_to_dim((10, 3), 3)
    with pytest.raises(ValueError):
        expand_to_dim((1,2,3), 2)
    with pytest.raises(TypeError):
        expand_to_dim("10", 3)


def test_permute_mask():
    assert permute_mask(10).shape == (10, )
    assert permute_mask(5, ).shape == (5, )
    assert permute_mask(1, 6, 3).shape == (1, 6, 3)
    assert permute_mask(8, 1, 2).shape == (8, 1, 2)


def test_operator_base_api():
    Nx, Ny, Nz = (4, 3, 7)
    x = np.random.uniform(-np.pi, +np.pi, Nx)
    y = np.random.uniform(-np.pi, +np.pi, Ny)
    z = np.random.uniform(-np.pi, +np.pi, Nz)
    points = list(map(lambda _: _.flatten(), np.meshgrid(x, y, z)))
    assert FinufftRealOperator(*points, n_modes=10).shape == (Nx * Ny * Nz, 10 * 10 * 10)
    assert FinufftRealOperator(x, n_modes=11).shape == (Nx, 11)
    X, Y = list(map(lambda _: _.flatten(), np.meshgrid(x, y)))
    with pytest.raises(ValueError):
        assert FinufftRealOperator(X, Y, n_modes=(6, 3, 1))
    
    assert FinufftRealOperator(X, Y, n_modes=(6, 3)).shape == (Nx * Ny, 6 * 3)



# 1D Operator

# N > P
test_1d_real_operator_dottest_N_even_gt_P_even = partial(dottest_1d_real_operator, 1200, 10)
test_1d_real_operator_dottest_N_even_gt_P_odd = partial(dottest_1d_real_operator, 1200, 11)
test_1d_real_operator_dottest_N_odd_gt_P_odd = partial(dottest_1d_real_operator, 1201, 11)
test_1d_real_operator_dottest_N_odd_gt_P_even = partial(dottest_1d_real_operator, 1201, 10)

"""
# N < P
test_1d_real_operator_dottest_N_even_lt_P_even = partial(dottest_1d_real_operator, 170, 338)
test_1d_real_operator_dottest_N_even_lt_P_odd = partial(dottest_1d_real_operator, 170, 341)
test_1d_real_operator_dottest_N_odd_lt_P_odd = partial(dottest_1d_real_operator, 171, 341)
test_1d_real_operator_dottest_N_odd_lt_P_even = partial(dottest_1d_real_operator, 171, 338)

# N > P, check design matrix
test_1d_real_operator_matches_design_matrix_N_even_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 1200, 10)
test_1d_real_operator_matches_design_matrix_N_even_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 1200, 11)
test_1d_real_operator_matches_design_matrix_N_odd_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 1201, 11)
test_1d_real_operator_matches_design_matrix_N_odd_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 1201, 10)

# N < P, check design matrix
test_1d_real_operator_matches_design_matrix_N_even_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 170, 338)
test_1d_real_operator_matches_design_matrix_N_even_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 170, 341)
test_1d_real_operator_matches_design_matrix_N_odd_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 173, 341)
test_1d_real_operator_matches_design_matrix_N_odd_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 173, 338)

# Test uniqueness of the dense matrix
test_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 10)
test_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 11)
test_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 11)
test_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 10)

test_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 10, 100)
test_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 11, 100)
test_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 11, 101)
test_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 10, 101)

test_1d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 100)
test_1d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 101)

# 2D operator

# N > P
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, 1200, 10)
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_odd = partial(dottest_2d_real_operator, 1200, 11)
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, 1201, 11) 
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_even = partial(dottest_2d_real_operator, 1201, 10) 

# N < P
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, 170, 338)
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_odd = partial(dottest_2d_real_operator, 170, 341)
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, 173, 341)
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_even = partial(dottest_2d_real_operator, 173, 338)

# N > P, Px != Py
test_2d_real_operator_dottest_N_equal_even_gt_P_odd_even = partial(dottest_2d_real_operator, 1200, (11, 10))
test_2d_real_operator_dottest_N_equal_even_gt_P_even_odd = partial(dottest_2d_real_operator, 1200, (10, 11))
test_2d_real_operator_dottest_N_equal_odd_gt_P_odd_even = partial(dottest_2d_real_operator, 1201, (11, 10))
test_2d_real_operator_dottest_N_equal_odd_gt_P_even_odd = partial(dottest_2d_real_operator, 1201, (10, 11))

# N < P, Px != Py
test_2d_real_operator_dottest_N_equal_even_lt_P_odd_even = partial(dottest_2d_real_operator, 170, (341, 338))
test_2d_real_operator_dottest_N_equal_even_lt_P_even_odd = partial(dottest_2d_real_operator, 170, (338, 341))
test_2d_real_operator_dottest_N_equal_odd_lt_P_odd_even = partial(dottest_2d_real_operator, 173, (341, 338))
test_2d_real_operator_dottest_N_equal_odd_lt_P_even_odd = partial(dottest_2d_real_operator, 173, (338, 341))

# N > P, Nx != Ny
test_2d_real_operator_dottest_N_even_odd_lt_P_equal_even = partial(dottest_2d_real_operator, (170, 173), 338)
test_2d_real_operator_dottest_N_even_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, (170, 173), 341)
test_2d_real_operator_dottest_N_odd_even_lt_P_equal_even = partial(dottest_2d_real_operator, (173, 170), 338)
test_2d_real_operator_dottest_N_odd_even_lt_P_equal_odd = partial(dottest_2d_real_operator, (173, 170), 341)

# N < P, Nx != Ny, Px != Py
test_2d_real_operator_dottest_N_even_odd_lt_P_even_odd = partial(dottest_2d_real_operator, (170, 173), (338, 341))
test_2d_real_operator_dottest_N_even_odd_gt_P_odd_even = partial(dottest_2d_real_operator, (170, 173), (341, 338))
test_2d_real_operator_dottest_N_odd_even_lt_P_even_odd = partial(dottest_2d_real_operator, (173, 170), (338, 341))
test_2d_real_operator_dottest_N_odd_even_gt_P_odd_even = partial(dottest_2d_real_operator, (173, 170), (341, 338))

# N > P, Nx != Ny, Px != Py
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, (1200, 1200), (10, 10))
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, (1201, 1201), (11, 11))

# N < P, N=(Nx, Ny), P=(Px, Py)
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, (10, 10),  (1200, 1200))
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, (11, 11), (1201, 1201))

# Test uniqueness of the design matrix.
test_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 100, 10)
test_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 100, 9)
test_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 101, 9)
test_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 101, 10)

test_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 10, 100)
test_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 9, 100)
test_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 9, 101)
test_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 10, 101)

test_2d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 100, 100)
test_2d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 101, 101)

# 3D operator

# N > P
test_3d_real_operator_dottest_N_even_gt_P_odd = partial(dottest_3d_real_operator, 14, 11)
test_3d_real_operator_dottest_N_even_gt_P_even = partial(dottest_3d_real_operator, 14, 10)
test_3d_real_operator_dottest_N_odd_gt_P_odd = partial(dottest_3d_real_operator, 15, 11)
test_3d_real_operator_dottest_N_odd_gt_P_even = partial(dottest_3d_real_operator, 15, 10)
test_3d_real_operator_dottest_N_even_gt_P_oee = partial(dottest_3d_real_operator, 14, (11, 10, 8))
test_3d_real_operator_dottest_N_even_gt_P_eoe = partial(dottest_3d_real_operator, 14, (8, 11, 14))
test_3d_real_operator_dottest_N_even_gt_P_ooe = partial(dottest_3d_real_operator, 14, (10, 14, 11))
test_3d_real_operator_dottest_N_even_gt_P_oee = partial(dottest_3d_real_operator, 27, (11, 10, 8))
test_3d_real_operator_dottest_N_even_gt_P_eoe = partial(dottest_3d_real_operator, 27, (8, 11, 14))
test_3d_real_operator_dottest_N_even_gt_P_ooe = partial(dottest_3d_real_operator, 27, (10, 14, 11))

# N < P
test_3d_real_operator_dottest_N_even_lt_P_odd = partial(dottest_3d_real_operator, 10, 13)
test_3d_real_operator_dottest_N_even_lt_P_even = partial(dottest_3d_real_operator, 10, 14)
test_3d_real_operator_dottest_N_odd_lt_P_odd = partial(dottest_3d_real_operator, 5, 15)
test_3d_real_operator_dottest_N_odd_lt_P_even = partial(dottest_3d_real_operator, 10, 15)
test_3d_real_operator_dottest_N_even_lt_P_oee = partial(dottest_3d_real_operator, 4, (11, 10, 8))
test_3d_real_operator_dottest_N_even_lt_P_eoe = partial(dottest_3d_real_operator, 4, (8, 11, 14))
test_3d_real_operator_dottest_N_even_lt_P_ooe = partial(dottest_3d_real_operator, 4, (10, 14, 11))
test_3d_real_operator_dottest_N_even_lt_P_oee = partial(dottest_3d_real_operator, 7, (11, 10, 8))
test_3d_real_operator_dottest_N_even_lt_P_eoe = partial(dottest_3d_real_operator, 7, (8, 11, 14))
test_3d_real_operator_dottest_N_even_lt_P_ooe = partial(dottest_3d_real_operator, 7, (10, 14, 11))

# N > P, check design matrix uniqueness
test_3d_real_operator_matches_design_matrix_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 14, 11)
test_3d_real_operator_matches_design_matrix_N_even_gt_P_even = partial(check_design_matrix_uniqueness_3d_real_operator, 14, 10)
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 15, 11)
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_3d_real_operator, 15, 10)

# N < P, check design matrix uniqueness
test_3d_real_operator_matches_design_matrix_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 10, 13)
test_3d_real_operator_matches_design_matrix_N_even_lt_P_even = partial(check_design_matrix_uniqueness_3d_real_operator, 10, 14)
test_3d_real_operator_matches_design_matrix_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 5, 15)
test_3d_real_operator_matches_design_matrix_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_3d_real_operator, 10, 15)

# N > P, Px != Py != Pz
test_3d_real_operator_matches_design_matrix_N_even_gt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 14, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_even_gt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 14, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_even_gt_P_odd_even_even = partial(check_design_matrix_uniqueness_3d_real_operator, 14, (10, 14, 11))
test_3d_real_operator_matches_design_matrix_N_even_gt_P_odd_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 27, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_even_gt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 27, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_even_gt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 27, (10, 14, 11))

test_3d_real_operator_matches_design_matrix_N_odd_gt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 14-1, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 14-1, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_odd_even_even = partial(check_design_matrix_uniqueness_3d_real_operator, 14-1, (10, 14, 11))
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_odd_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 27-1, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 27-1, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_odd_gt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 27-1, (10, 14, 11))

# N < P, Px != Py != Pz
test_3d_real_operator_matches_design_matrix_N_even_lt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 6, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_even_lt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 6, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_even_lt_P_odd_even_even = partial(check_design_matrix_uniqueness_3d_real_operator, 6, (10, 14, 11))
test_3d_real_operator_matches_design_matrix_N_even_lt_P_odd_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 7, (11, 10, 8))
test_3d_real_operator_matches_design_matrix_N_even_lt_P_even_odd_even = partial(check_design_matrix_uniqueness_3d_real_operator, 7, (8, 11, 14))
test_3d_real_operator_matches_design_matrix_N_even_lt_P_odd_even_odd = partial(check_design_matrix_uniqueness_3d_real_operator, 7, (10, 14, 11))
"""