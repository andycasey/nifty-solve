# TODO
import jax.numpy as np
from functools import partial
from pylops.utils import dottest
from nifty_solve.jax_operators import (JaxFinufft1DRealOperator, JaxFinufft2DRealOperator, JaxFinufft3DRealOperator)
from nifty_solve.utils import expand_to_dim
EPSILON = 1e-9

def test_extra_imports():
    import jax
    from jax_finufft import nufft2


def dottest_1d_real_operator(OperatorFactory, N, P, rtol=1e-6):
    x = np.linspace(-np.pi, np.pi, N)
    A = OperatorFactory(x, P, eps=EPSILON)
    dottest(A, rtol=rtol)



def check_is_full_rank(A, rtol=1e-6):
    assert np.linalg.matrix_rank(A, rtol=rtol) == min(A.shape)
    """
        diff = np.zeros((min(A.shape), min(A.shape)))
        for i in range(min(A.shape)):
            for j in range(min(A.shape)):
                diff[i, j] = np.linalg.norm(A[:, i] - A[:, j])
        
        assert np.sum(diff < tolerance) == min(A.shape)
    """
        
def check_design_matrix_uniqueness(Op, points, P, **kwargs):
    A = Op(*points, P, eps=EPSILON, **kwargs)
    A_dense = A.todense()
    """
    A_unique = np.unique(A_dense, axis=1)
    if A_dense.shape != A_unique.shape:
        for i in range(A_dense.shape[1]):
            foo = np.where(np.all(A_dense == A_dense[:, [i]], axis=0))[0]
            if len(foo) > 1:
                print(i, foo, np.unravel_index(foo, A.permute.shape))
        
        assert False
    """
    check_is_full_rank(A_dense)


def check_design_matrix_uniqueness_1d_real_operator(OperatorFactory, N, P):
    x = np.linspace(0, np.pi, N)
    check_design_matrix_uniqueness(OperatorFactory, (x, ), P)

def check_design_matrix_uniqueness_2d_real_operator(OperatorFactory, N, P):
    Nx, Ny = expand_to_dim(N, 2)

    x = np.random.uniform(0.01, np.pi, Nx)
    y = np.random.uniform(0.01, np.pi, Ny)
    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))

    check_design_matrix_uniqueness(OperatorFactory, (X, Y), P)


def check_design_matrix_uniqueness_3d_real_operator(N, P):
    Nx, Ny, Nz = expand_to_dim(N, 3)

    x = np.random.uniform(0, np.pi, Nx)
    y = np.random.uniform(0, np.pi, Ny)
    z = np.random.uniform(0, np.pi, Nz)
    X, Y, Z = map(lambda x: x.flatten(), np.meshgrid(x, y, z))

    check_design_matrix_uniqueness(Finufft3DRealOperator, (X, Y, Z), P)
    

def dottest_2d_real_operator(OperatorFactory, N, P, rtol=1e-6):
    if isinstance(N, int):
        Nx = Ny = N
    else:
        Nx, Ny = N
    x = np.linspace(-np.pi, np.pi, Nx)
    y = np.linspace(-np.pi, np.pi, Ny)

    X, Y = map(lambda x: x.flatten(), np.meshgrid(x, y))
    A = OperatorFactory(X, Y, P, eps=EPSILON)
    dottest(A, rtol=rtol)


def dottest_3d_real_operator(N, P):
    X = np.random.uniform(-np.pi, +np.pi, N)
    Y = np.random.uniform(-np.pi, +np.pi, N)
    Z = np.random.uniform(-np.pi, +np.pi, N)
    A = Finufft3DRealOperator(X, Y, Z, P, eps=EPSILON)
    dottest(A)




# 1D Operator

# N > P
test_jax_1d_real_operator_dottest_N_even_gt_P_even = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 80, 10)
test_jax_1d_real_operator_dottest_N_even_gt_P_odd = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 80, 11)
test_jax_1d_real_operator_dottest_N_odd_gt_P_odd = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 81, 11)
test_jax_1d_real_operator_dottest_N_odd_gt_P_even = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 81, 10)

# N < P
test_jax_1d_real_operator_dottest_N_even_lt_P_even = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 170, 338)
test_jax_1d_real_operator_dottest_N_even_lt_P_odd = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 170, 341)
test_jax_1d_real_operator_dottest_N_odd_lt_P_odd = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 171, 341)
test_jax_1d_real_operator_dottest_N_odd_lt_P_even = partial(dottest_1d_real_operator, JaxFinufft1DRealOperator, 171, 338)

# N > P, check design matrix
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_1 = partial(check_1d_real_operator_matches_design_matrix, 10, 1)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_2 = partial(check_1d_real_operator_matches_design_matrix, 10, 2)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_3 = partial(check_1d_real_operator_matches_design_matrix, 10, 3)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_4 = partial(check_1d_real_operator_matches_design_matrix, 10, 4)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_5 = partial(check_1d_real_operator_matches_design_matrix, 10, 5)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_6 = partial(check_1d_real_operator_matches_design_matrix, 10, 6)
###test_jax_1d_real_operator_matches_design_matrix_N_even_P_7 = partial(check_1d_real_operator_matches_design_matrix, 10, 7)

###test_jax_1d_real_operator_matches_design_matrix_N_even_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 80, 10)
###test_jax_1d_real_operator_matches_design_matrix_N_even_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 80, 11)
#test_jax_1d_real_operator_matches_design_matrix_N_odd_gt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 81, 11)
###test_jax_1d_real_operator_matches_design_matrix_N_odd_gt_P_even = partial(check_1d_real_operator_matches_design_matrix, 81, 10)

# N < P, check design matrix
###test_jax_1d_real_operator_matches_design_matrix_N_even_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 170, 338)
###test_jax_1d_real_operator_matches_design_matrix_N_even_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 170, 341)
#test_jax_1d_real_operator_matches_design_matrix_N_odd_lt_P_odd = partial(check_1d_real_operator_matches_design_matrix, 173, 341)
###test_jax_1d_real_operator_matches_design_matrix_N_odd_lt_P_even = partial(check_1d_real_operator_matches_design_matrix, 173, 338)

# Test uniqueness of the dense matrix
# Under-parameterised case
test_jax_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 100, 10)
test_jax_1d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 100, 11)
test_jax_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 101, 11)
test_jax_1d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 101, 10)

# Over-parameterised case
test_jax_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 10, 100)
test_jax_1d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 11, 100)
test_jax_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 11, 101)
test_jax_1d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 10, 101)

test_jax_1d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 100, 100)
test_jax_1d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_1d_real_operator, JaxFinufft1DRealOperator, 101, 101)


# 2D operator

# N > P
test_jax_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 80, 10)
test_jax_2d_real_operator_dottest_N_equal_even_gt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 80, 11)
test_jax_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 81, 11) 
test_jax_2d_real_operator_dottest_N_equal_odd_gt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 81, 10) 

# N < P
test_jax_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 170, 338)
test_jax_2d_real_operator_dottest_N_equal_even_lt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 170, 341)
test_jax_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 173, 341)
test_jax_2d_real_operator_dottest_N_equal_odd_lt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 173, 338)

# N > P, Px != Py
test_jax_2d_real_operator_dottest_N_equal_even_gt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 80, (11, 10))
test_jax_2d_real_operator_dottest_N_equal_even_gt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 80, (10, 11))
test_jax_2d_real_operator_dottest_N_equal_odd_gt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 81, (11, 10))
test_jax_2d_real_operator_dottest_N_equal_odd_gt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 81, (10, 11))

# N < P, Px != Py
test_jax_2d_real_operator_dottest_N_equal_even_lt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 170, (341, 338))
test_jax_2d_real_operator_dottest_N_equal_even_lt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 170, (338, 341))
test_jax_2d_real_operator_dottest_N_equal_odd_lt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 173, (341, 338))
test_jax_2d_real_operator_dottest_N_equal_odd_lt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, 173, (338, 341))

# N > P, Nx != Ny
test_jax_2d_real_operator_dottest_N_even_odd_lt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (170, 173), 338)
test_jax_2d_real_operator_dottest_N_even_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (170, 173), 341)
test_jax_2d_real_operator_dottest_N_odd_even_lt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (173, 170), 338)
test_jax_2d_real_operator_dottest_N_odd_even_lt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (173, 170), 341)

# N < P, Nx != Ny, Px != Py
test_jax_2d_real_operator_dottest_N_even_odd_lt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (170, 173), (338, 341))
test_jax_2d_real_operator_dottest_N_even_odd_gt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (170, 173), (341, 338))
test_jax_2d_real_operator_dottest_N_odd_even_lt_P_even_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (173, 170), (338, 341))
test_jax_2d_real_operator_dottest_N_odd_even_gt_P_odd_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (173, 170), (341, 338))

# N > P, Nx != Ny, Px != Py
test_jax_2d_real_operator_dottest_N_equal_even_gt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (80, 80), (10, 10))
test_jax_2d_real_operator_dottest_N_equal_odd_gt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (81, 81), (11, 11))

# N < P, N=(Nx, Ny), P=(Px, Py)
test_jax_2d_real_operator_dottest_N_equal_even_lt_P_equal_even = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (10, 10),  (80, 80))
test_jax_2d_real_operator_dottest_N_equal_odd_lt_P_equal_odd = partial(dottest_2d_real_operator, JaxFinufft2DRealOperator, (11, 11), (81, 81))


# Test uniqueness of the design matrix.
test_jax_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 30, 10)
test_jax_2d_real_operator_design_matrix_uniqueness_N_even_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 30, 9)
test_jax_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 31, 9)
test_jax_2d_real_operator_design_matrix_uniqueness_N_odd_gt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 31, 10)

test_jax_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 10, 30)
test_jax_2d_real_operator_design_matrix_uniqueness_N_even_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 9, 30)
test_jax_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 9, 31)
test_jax_2d_real_operator_design_matrix_uniqueness_N_odd_lt_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 10, 31)

test_jax_2d_real_operator_design_matrix_uniqueness_N_equal_P_even = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 30, 30)
test_jax_2d_real_operator_design_matrix_uniqueness_N_equal_P_odd = partial(check_design_matrix_uniqueness_2d_real_operator, JaxFinufft2DRealOperator, 31, 31)
