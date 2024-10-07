import finufft
import numpy as np
import numpy.typing as npt
from typing import Optional, Union
from scipy.sparse.linalg import lsqr
from pylops import LinearOperator, Diagonal, Identity

class Finufft1DOperator(LinearOperator):

    def __init__(
        self,
        x: npt.ArrayLike,
        P: int,
        dtype: Optional[Union[np.dtype, str]] = np.complex128,
        **kwargs
    ):
        super().__init__(dtype=dtype, shape=(len(x), P))
        self.explicit = False
        self._plan_matvec = finufft.Plan(2, (P, ), **kwargs)
        self._plan_rmatvec = finufft.Plan(1, (P, ), **kwargs)
        for plan in (self._plan_rmatvec, self._plan_matvec):
            plan.setpts(x)
        return None
    
    def _matvec(self, c):
        #return self._plan_matvec.execute(c * self.Λ) * self.C_inv_sqrt
        return self._plan_matvec.execute(c)
    
    def _rmatvec(self, f):
        return self._plan_rmatvec.execute(f)


if __name__ == "__main__":



    np.random.seed(8675309) # MAGIC to enforce reproducibility
    RCOND = 1e-14 # MAGIC input to `np.linalg.lstsq()`

    def truefunction(ts):
        """
        This function produces the TRUTH, whi ch does not lie in the basis spanned by
        the Fourier modes.
        """
        return 1.0 * np.sin(11. * ts) / ts + 1.0 * ts * np.cos(37. * ts) + 0.5 * np.sin(400. * ts)

    n = 23
    T = 1.5 # range
    deltaomega = 0.5 * np.pi / T # frequency spacing for Fourier series

    # generate like a Ward
    # ts = T * np.arange(0.5 / n, 1., 1. / n)

    # generate like a Hogg (harder problem!)
    # ts = np.sort(T * np.random.uniform(size=n))

    # generate intermediate between Hogg and Ward
    ts = (T * (np.arange(0.5 / n, 1., 1. / n)
        + (0.75 / n) * (np.random.uniform(size=n) - 0.5))) # magic 0.75

    xi = np.linspace(0, T, 1024)
    yi = truefunction(xi)

    Y = truefunction(ts)
    ts = (ts / T) * 2 * np.pi - np.pi
    xi = xi / T * 2 * np.pi - np.pi
    xi = np.sort(np.hstack([xi, ts]))


    x_true, f_true = (ts, Y)
    #f_obs, f_err = (f_true, None)
    f_err = np.random.normal(0, 50, size=f_true.size)
    f_obs = f_true + np.random.normal(0, 1, size=f_err.size) * f_err #+ np.random.normal(0, 0.01, size=f_obs.shape)
    f_err = np.abs(f_err)

    P = 512

    def ω(P):
        K = np.arange(P) - P // 2
        δω = (2 * np.pi) / P
        return K * δω


    def construct_matern_32_weight_vector(n_modes: int, s: float = 1) -> npt.ArrayLike:
        return 1/((s*ω(n_modes))**2 + 1)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(ts, f_obs, c="k", s=50, zorder=100)

    lo = Finufft1DOperator(ts, P=P)
    x_est = lo / f_obs.astype(complex)


    width = 100
    Λ = (construct_matern_32_weight_vector(P, width)**-2).astype(complex)


    li = Finufft1DOperator(xi, P=P)
    f_pred = li @ x_est
    f_pred = np.sign(f_pred) * np.abs(f_pred)
    ax.plot(xi, f_pred, c="tab:red")

    A = lo

    #x_est2 = (A.H @ A + Λ) / (A.H * f_obs.astype(complex))
    Λ_inv_sqrt = Diagonal(Λ**-0.5)
    x_est2 = (A * Λ_inv_sqrt) / (f_obs.astype(complex))
    x_est2 *= Λ**-0.5
    
    f_pred2 = li @ x_est2
    f_pred2 = np.sign(f_pred2) * np.abs(f_pred2)
    ax.plot(xi, f_pred2, c="tab:blue")


    # Now with errors
    P = 15
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(ts, f_obs, c="k", s=50, zorder=100)
    ax.errorbar(ts, f_obs, yerr=f_err, fmt="none", c="k", zorder=99)

    lo = Finufft1DOperator(ts, P=P)
    li = Finufft1DOperator(xi, P=P)

    x_est = lo / f_obs.astype(complex)

    A = lo
    C_inv = Diagonal(1/f_err**2)
    Y = f_obs.astype(complex)

    #x_est2, *_ = lsqr(A.H @ C_inv @ A, A.H @ C_inv @ Y)
    x_est2 = (A.H @ C_inv @ A) / (A.H @ Cinv @ Y)
    f_pred = li @ x_est2
    f_pred = np.sign(f_pred) * np.abs(f_pred)
    ax.plot(xi, f_pred, c="tab:red")



