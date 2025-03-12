import numpy as np
import itertools
import pytest
from functools import partial
from pylops.utils import dottest

from nifty_solve.operators import FinufftRealOperator, Finufft1DRealOperator, Finufft2DRealOperator, Finufft3DRealOperator, expand_to_dim

EPSILON = 1e-9
IMAG_EPSILON = 1e-6

def design_matrix_as_is(xs, P):
    X = np.ones_like(xs).reshape(len(xs), 1)
    for j in range(1, P):
        if j % 2 == 0:
            X = np.concatenate((X, np.cos(j * xs)[:, None]), axis=1)
        else:
            X = np.concatenate((X, np.sin((j + 1) * xs)[:, None]), axis=1)
    return X


def dottest_1d_real_operator(N, P, dtype=np.float64):
    x = np.linspace(-np.pi, np.pi, N, dtype=dtype)
    A = Finufft1DRealOperator(x, P, eps=EPSILON)
    dottest(A)

    # Check imaginary components are 0
    i = np.max(np.abs(np.imag(A._plan_matvec.execute(A._pre_matvec(np.random.normal(size=P))))))
    assert i < IMAG_EPSILON


def check_is_full_rank(A, tolerance=1e-5):
    if np.linalg.matrix_rank(A, hermitian=(A.shape[0] == A.shape[1])) != min(A.shape):
        diff = np.zeros((min(A.shape), min(A.shape)))
        for i in range(min(A.shape)):
            for j in range(min(A.shape)):
                diff[i, j] = np.linalg.norm(A[:, i] - A[:, j])
        
        assert np.sum(diff < tolerance) == min(A.shape)
            
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(diff)
        plt.colorbar(im)
        assert False
        """
        
def check_design_matrix_uniqueness(Op, points, P, **kwargs):
    A = Op(*points, P, eps=EPSILON, **kwargs)
    A_dense = A.todense()
    A_unique = np.unique(A_dense, axis=1)
    if A_dense.shape != A_unique.shape:
        for i in range(A_dense.shape[1]):
            foo = np.where(np.all(A_dense == A_dense[:, [i]], axis=0))[0]
            if len(foo) > 1:
                print(i, foo, np.unravel_index(foo, A.permute.shape))
        
        assert False

    check_is_full_rank(A_dense)


def check_design_matrix_uniqueness_1d_real_operator(N, P):
    x = np.linspace(0, np.pi, N)
    check_design_matrix_uniqueness(Finufft1DRealOperator, (x, ), P)

def check_design_matrix_uniqueness_2d_real_operator(N, P):
    Nx, Ny = expand_to_dim(N, 2)

    x = np.random.uniform(0.01, np.pi, Nx)
    y = np.random.uniform(0.01, np.pi, Ny)
    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))

    check_design_matrix_uniqueness(Finufft2DRealOperator, (X, Y), P)


def check_design_matrix_uniqueness_3d_real_operator(N, P):
    Nx, Ny, Nz = expand_to_dim(N, 3)

    x = np.random.uniform(0, np.pi, Nx)
    y = np.random.uniform(0, np.pi, Ny)
    z = np.random.uniform(0, np.pi, Nz)
    X, Y, Z = map(lambda x: x.flatten(), np.meshgrid(x, y, z))

    check_design_matrix_uniqueness(Finufft3DRealOperator, (X, Y, Z), P)
    
def get_mode_indices(P):
    mode_indices = np.zeros(P, dtype=int)
    mode_indices[2::2] = np.arange(1, P//2 + (P % 2))
    mode_indices[1::2] = np.arange(P//2 + (P % 2), P)[::-1]
    return mode_indices

def check_1d_real_operator_matches_design_matrix(N, P):
    x = np.linspace(-np.pi, np.pi, N)

    A = Finufft1DRealOperator(x, P, eps=EPSILON)

    local_mode_indices = np.hstack([0, (np.tile(np.arange(1, P // 2 + 1), 2).reshape((2, -1)).T* np.array([-1, 1])).flatten()[:P-1]])
    finufft_mode_indices = np.arange(-P // 2 + 1, P//2 + 1)

    A1 = design_matrix_as_is(x/2, P)[:, np.argsort(local_mode_indices)]
    assert np.allclose(A.todense(), A1)

"""
#def check_2d_real_operator_matches_design_matrix(Nx, Ny, Px, Py):###

    x = np.linspace(-np.pi, np.pi, Nx)
    y = np.linspace(-np.pi, np.pi, Ny)
    xg, yg = np.meshgrid(x, y)
    X, Y = xg.flatten(), yg.flatten()

    A = Finufft2DRealOperator(X, Y, (Px, Py), eps=EPSILON)

    local_mode_indices_x = np.hstack([0, (np.tile(np.arange(1, Px // 2 + 1), 2).reshape((2, -1)).T* np.array([-1, 1])).flatten()[:Px-1]])
    local_mode_indices_y = np.hstack([0, (np.tile(np.arange(1, Py // 2 + 1), 2).reshape((2, -1)).T* np.array([-1, 1])).flatten()[:Py-1]])
    
    #finufft_mode_indices = np.arange(-P // 2 + 1, P//2 + 1)

    Ax = design_matrix_as_is(x/2, Px)[:, np.argsort(local_mode_indices_x)]
    Ay = design_matrix_as_is(y/2, Py)[:, np.argsort(local_mode_indices_y)]

    A_xy = np.kron(Ay, Ax)

    if not np.allclose(A.todense(), A_xy):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 7)
        axes[0].set_title("A")
        axes[1].set_title("Kron(Ay,Ax)")
        axes[2].set_title("Kron(Ax,Ay)")

        axes[0].imshow(A.todense())
        axes[1].imshow(np.kron(Ay.T, Ax.T))
        axes[2].imshow(np.kron(Ax.T, Ay.T))
        axes[3].imshow(Ax)
        axes[4].imshow(Ay)
        axes[5].imshow(Finufft1DRealOperator(x, Px, eps=EPSILON).todense())
        axes[6].imshow(Finufft1DRealOperator(y, Py, eps=EPSILON).todense())

        assert False
"""

def dottest_2d_real_operator(N, P):
    if isinstance(N, int):
        Nx = Ny = N
    else:
        Nx, Ny = N
    x = np.linspace(-np.pi, np.pi, Nx)
    y = np.linspace(-np.pi, np.pi, Ny)

    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))
    A = Finufft2DRealOperator(X, Y, P, eps=EPSILON)
    dottest(A)

    i = np.max(np.abs(np.imag(A._plan_matvec.execute(A._pre_matvec(np.random.normal(size=A.shape[1]))))))
    assert i < IMAG_EPSILON


def dottest_3d_real_operator(N, P):
    X = np.random.uniform(-np.pi, +np.pi, N)
    Y = np.random.uniform(-np.pi, +np.pi, N)
    Z = np.random.uniform(-np.pi, +np.pi, N)
    A = Finufft3DRealOperator(X, Y, Z, P, eps=EPSILON)
    dottest(A)

    i = np.max(np.abs(np.imag(A._plan_matvec.execute(A._pre_matvec(np.random.normal(size=A.shape[1]))))))
    assert i < IMAG_EPSILON


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
    

"""
def test_permute_mask():
    assert permute_mask(10).shape == (10, )
    assert permute_mask(5, ).shape == (5, )
    assert permute_mask(1, 6, 3).shape == (1, 6, 3)
    assert permute_mask(8, 1, 2).shape == (8, 1, 2)
"""

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

# np float 64 vs 32
test_1d_real_operator_dottest_float32 = partial(dottest_1d_real_operator, 80, 10, np.float32)


# N > P
test_1d_real_operator_dottest_N_even_gt_P_even = partial(dottest_1d_real_operator, 80, 10)
test_1d_real_operator_dottest_N_even_gt_P_odd = partial(dottest_1d_real_operator, 80, 11)
test_1d_real_operator_dottest_N_odd_gt_P_odd = partial(dottest_1d_real_operator, 81, 11)
test_1d_real_operator_dottest_N_odd_gt_P_even = partial(dottest_1d_real_operator, 81, 10)

# N < P
test_1d_real_operator_dottest_N_even_lt_P_even = partial(dottest_1d_real_operator, 170, 338)
test_1d_real_operator_dottest_N_even_lt_P_odd = partial(dottest_1d_real_operator, 170, 341)
test_1d_real_operator_dottest_N_odd_lt_P_odd = partial(dottest_1d_real_operator, 171, 341)
test_1d_real_operator_dottest_N_odd_lt_P_even = partial(dottest_1d_real_operator, 171, 338)

# N > P, check design matrix
###test_1d_real_operator_matches_design_matrix_N_even_P_1 = partial(check_1d_real_operator_matches_design_matrix, 10, 1)
###test_1d_real_operator_matches_design_matrix_N_even_P_2 = partial(check_1d_real_operator_matches_design_matrix, 10, 2)
###test_1d_real_operator_matches_design_matrix_N_even_P_3 = partial(check_1d_real_operator_matches_design_matrix, 10, 3)
###test_1d_real_operator_matches_design_matrix_N_even_P_4 = partial(check_1d_real_operator_matches_design_matrix, 10, 4)
###test_1d_real_operator_matches_design_matrix_N_even_P_5 = partial(check_1d_real_operator_matches_design_matrix, 10, 5)
###test_1d_real_operator_matches_design_matrix_N_even_P_6 = partial(check_1d_real_operator_matches_design_matrix, 10, 6)
###test_1d_real_operator_matches_design_matrix_N_even_P_7 = partial(check_1d_real_operator_matches_design_matrix, 10, 7)

###test_1d_real_operator_matches_design_matrix_N_even_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 80, 10)
###test_1d_real_operator_matches_design_matrix_N_even_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 80, 11)
#test_1d_real_operator_matches_design_matrix_N_odd_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 81, 11)
###test_1d_real_operator_matches_design_matrix_N_odd_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 81, 10)

# N < P, check design matrix
###test_1d_real_operator_matches_design_matrix_N_even_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 170, 338)
###test_1d_real_operator_matches_design_matrix_N_even_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 170, 341)
#test_1d_real_operator_matches_design_matrix_N_odd_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 173, 341)
###test_1d_real_operator_matches_design_matrix_N_odd_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 173, 338)

# Test uniqueness of the dense matrix
# Under-parameterised case
test_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 10)
test_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 11)
test_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 11)
test_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 10)

# Over-parameterised case
test_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 10, 100)
test_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 11, 100)
test_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 11, 101)
test_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 10, 101)

test_1d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, 100, 100)
test_1d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, 101, 101)

# 2D operator

# N > P
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, 80, 10)
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_odd = partial(dottest_2d_real_operator, 80, 11)
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, 81, 11) 
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_even = partial(dottest_2d_real_operator, 81, 10) 

# N < P
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, 170, 338)
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_odd = partial(dottest_2d_real_operator, 170, 341)
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, 173, 341)
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_even = partial(dottest_2d_real_operator, 173, 338)

# N > P, Px != Py
test_2d_real_operator_dottest_N_equal_even_gt_P_odd_even = partial(dottest_2d_real_operator, 80, (11, 10))
test_2d_real_operator_dottest_N_equal_even_gt_P_even_odd = partial(dottest_2d_real_operator, 80, (10, 11))
test_2d_real_operator_dottest_N_equal_odd_gt_P_odd_even = partial(dottest_2d_real_operator, 81, (11, 10))
test_2d_real_operator_dottest_N_equal_odd_gt_P_even_odd = partial(dottest_2d_real_operator, 81, (10, 11))

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
test_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, (80, 80), (10, 10))
test_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, (81, 81), (11, 11))

# N < P, N=(Nx, Ny), P=(Px, Py)
test_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, (10, 10),  (80, 80))
test_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, (11, 11), (81, 81))

# Check design matrix computed by hand
###test_2d_operator_matches_design_matrix_N_even_P_even = partial(check_2d_real_operator_matches_design_matrix, 8, 15, 10, 12)
####test_2d_operator_matches_design_matrix_N_even_P_odd = partial(check_2d_real_operator_matches_design_matrix, 10, 11)
####test_2d_operator_matches_design_matrix_N_odd_P_even = partial(check_2d_real_operator_matches_design_matrix, 9, 10)
#test_2d_operator_matches_design_matrix_N_odd_P_odd = partial(check_2d_real_operator_matches_design_matrix, 9, 11)

# Test uniqueness of the design matrix.
test_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 30, 10)
test_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 30, 9)
test_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 31, 9)
test_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 31, 10)

test_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 10, 30)
test_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 9, 30)
test_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 9, 31)
test_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 10, 31)

test_2d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, 30, 30)
test_2d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, 31, 31)

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
