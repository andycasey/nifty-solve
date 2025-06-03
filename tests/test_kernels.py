import numpy as np
from pylops import Diagonal
from scipy.spatial.distance import cdist

from nifty_solve.kernels import SquaredExponential, Matern32, BaseKernel
from nifty_solve.operators import FinufftRealOperator, FinufftOperator

def matern_32_kernel(r, l, v=1):
    r_l = np.abs(r)/l
    return v * (1 + r_l) * np.exp(-r_l)

def se_kernel(d, lengthscale, variance=1):
    return variance * np.exp(-0.5 * (d/lengthscale)**2)

np.random.seed(8)


def test_squared_exponential_kernel_accuracy_1d(
    length_scale=0.1,
    N=100,
    t_min=0,
    t_max=np.pi,
    epsilon=1e-3,
):
    t = np.sort(np.random.uniform(t_min, t_max, N))

    K = SquaredExponential(length_scale)
    P = K._modes_required(epsilon)

    frequencies = np.arange(-P//2, P//2)
    A = FinufftOperator(t, n_modes=P)
    Λ = Diagonal(K.spectral_density(frequencies).astype(A.dtype))

    Cov = (A @ Λ @ A.H).todense().real
    Cov_naive = se_kernel(t[:, None] - t[None, :], length_scale)
    
    assert np.allclose(Cov, Cov_naive, atol=1e-2)
