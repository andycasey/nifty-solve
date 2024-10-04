from __future__ import annotations

import finufft
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as sp
import warnings

from dataclasses import dataclass
from typing import Literal, Optional, Union
from functools import cached_property
from timeit import default_timer as timer

warnings.filterwarnings('ignore')

    

class Fourier1DBasis:

    """A linear 1D Fourier model."""

    def __init__(self, x: npt.ArrayLike, n_modes: int, finufft_kwargs: Optional[dict] = None) -> None:
        """
        A linear 1D Fourier model.

        :param x:
            The noise-less positions of sampled data.
        
        :param n_modes:
            The number of Fourier modes to fit.

        :param finufft_kwargs: [Optional]   
            The keyword arguments to pass to the `finufft.Plan()` constructor.        
        """
        self.x = x
        self.n_modes = n_modes
        self.finufft_kwargs = finufft_kwargs or {}
        self.finufft_kwargs.setdefault("eps", 1e-8)    
        self._f1_plan = finufft.Plan(1, (n_modes, ), **self.finufft_kwargs) # A.T
        self._f2_plan = finufft.Plan(2, (n_modes, ), **self.finufft_kwargs) # A
        for plan in (self._f1_plan, self._f2_plan):
            plan.setpts(x)
        return None

    def _validate_inputs(
        self, 
        f: npt.ArrayLike,
        f_err: Optional[npt.ArrayLike] = None,
        Λ: Optional[npt.ArrayLike] = None
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        if len(f) != len(self.x):
            raise ValueError(f"Input array `f` must have the same length as `x` ({len(self.x)}), but has length {len(f)}.")
        
        f = np.array(f, dtype=np.complex128)
        if f_err is None:
            inv_f_err = 1
        else:
            if len(f_err) != len(self.x):
                raise ValueError(f"Input array `f_err` must have the same length as `x` ({len(self.x)}), but has length {len(f_err)}.")
            inv_f_err = 1/f_err
        
        if Λ is None:
            Λ_inv_sqrt = 1
        else:
            if len(Λ) != self.n_modes:
                raise ValueError(f"Feature weighting `Λ` must have the same length as `n_modes` ({self.n_modes}), but has length {len(Λ)}.")
            Λ_inv_sqrt = Λ**-0.5

        return (f, inv_f_err, Λ_inv_sqrt)


    def solve(
        self,
        f: npt.ArrayLike,
        f_err: Optional[npt.ArrayLike] = None,
        Λ: Optional[npt.ArrayLike] = None,
        **kwargs
    ):
        """
        Solve the linear system of equations given the (complex) data `f` sampled at `x`.
        
        :param f: 
            The complex-valued data sampled at `x`.
        
        :param f_err: [Optional]
            The error in the data `f`.
        
        :param Λ: [Optional]
            The feature-weighting (regularization) matrix as defined in Hogg & Villar ().

            An example of Λ can be set knowing that every mode has some characteristic frequency ω, 
            where we can choose

                f(ω) = \frac{1}{(sω)^2 + 1}
            
            where s is a hyper-parameter that controls ths (inverse) width of the weighting function
            in frequency space. Given this function we can compute the inverse of the feature-weighting
            matrix:

                Λ^(-1) = f(ω)^2
            
        :param **kwargs:
            Additional keyword arguments to pass to `scipy.linalg.sparse.lsqr`.        
        """

        t_plan = -timer()
        f, f_inv_err, Λ_inv_sqrt = self._validate_inputs(f, f_err, Λ)

        lo = sp.LinearOperator(
            (len(self.x), self.n_modes),
            matvec=lambda c: self._f2_plan.execute(c * Λ_inv_sqrt) * f_inv_err,
            rmatvec=lambda f: self._f1_plan.execute(f * f_inv_err) * Λ_inv_sqrt,
            dtype=complex
        )

        Y = (f * f_inv_err).astype(complex)        

        lsqr_kwargs = dict(atol=0, btol=0, conlim=0, x0=lo.rmatvec(Y), calc_var=True)
        lsqr_kwargs.update(**kwargs)
        t_plan += timer()

        t_solve = -timer()
        self.c, *v = sp.lsqr(lo, Y, **lsqr_kwargs)
        t_solve += timer()
        self.c *= Λ_inv_sqrt
        self.meta = dict(
            lsqr_kwargs=lsqr_kwargs,
            finufft_kwargs=self.finufft_kwargs, 
            t_plan=t_plan,           
            t_solve=t_solve, 
            lsqr_meta=dict(zip(
                ("istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", "xnorm", "var"), 
                v
            ))
        )
        self.c_var = self.meta["lsqr_meta"].get("var", None)
        return (self.c, self.c_var)
    
    def predict(self, x: npt.ArrayLike | None = None, c: npt.ArrayLike | None = None) -> npt.ArrayLike:
        return finufft.nufft1d2(x if x is not None else self.x, c if c is not None else self.c)

    def predict_magnitude(self, x: npt.ArrayLike | None = None, c: npt.ArrayLike | None = None) -> npt.ArrayLike:
        v = self.predict(x, c)
        return np.sign(v) * np.abs(v)


def ω(P):
    K = np.arange(P) - P // 2
    δω = (2 * np.pi) / P
    return K * δω

def construct_matern_32_weight_vector(n_modes: int, s: float = 1) -> npt.ArrayLike:
    return 1/((s*ω(n_modes))**2 + 1)


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

    xi = np.linspace(0, T, 2048)
    yi = truefunction(xi)

    Y = truefunction(ts)
    ts = (ts / T) * 2 * np.pi - np.pi
    xi = xi / T * 2 * np.pi - np.pi

    x_true, f_true = (ts, Y)
    f_obs, f_err = (f_true, None)

    """
    p = 1024
    X, omegas = fourier_design_matrix(ts, deltaomega, p)
    print(ts.shape, X.shape, Y.shape, omegas.shape)
    beta_default = train_feature_weighted_ols(X, Y)
    beta_smooth = train_feature_weighted_ols(X, Y, weight_function(omegas, width))
    beta_smooth2 = train_feature_weighted_ols(X, Y, weight_function(omegas, width) ** 0.25)
    Lambdadiag = 1. / (0.07 * weight_function(omegas, width) ** 2) # MAGIC 0.07
    beta_CLambda = train_feature_weighted_wls(X, Y, Lambdadiag, Cdiag)
    """


    N = 512

    fig, ax = plt.subplots()
    v = np.logspace(-10, 0, 100)
    g = [np.ptp(construct_matern_32_weight_vector(N, vi)) for vi in v]
    ax.plot(v, g)
    ax.semilogx()



    model = Fourier1DBasis(x_true, N)
    nnn = 2048
    #xi = np.arange(0.5 * T / nnn, T, T / nnn)

    regularization_args = [
        None,
        (100, ),
    ]
    def generate_complex_mvn_samples(mu, Sigma, num_samples):
        """
        Generates samples from a multivariate normal distribution with complex mean and covariance matrix.

        Args:
            mu: Complex mean vector.
            Sigma: Complex covariance matrix.
            num_samples: Number of samples to generate.

        Returns:
            Complex samples.
        """

        # Separate real and imaginary parts
        mu_real = np.concatenate([np.real(mu), np.imag(mu)])
        Sigma_real = np.block([[np.real(Sigma), -np.imag(Sigma)],
                            [np.imag(Sigma), np.real(Sigma)]])

        # Generate real-valued samples
        real_samples = np.random.multivariate_normal(mu_real, Sigma_real, num_samples)

        # Separate real and imaginary parts
        real_part = real_samples[:, :len(mu)]
        imag_part = real_samples[:, len(mu):]

        # Combine real and imaginary parts
        complex_samples = real_part + 1j * imag_part

        return complex_samples    

    colors = ("tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "cyan", "magenta", "yellow", "black")

    fig, axes = plt.subplots(2, 1, figsize=(15, 6))#, sharex=True, sharey=True)
    
    for i, args in enumerate(regularization_args):
        # (args, ax) in enumerate(zip(regularization_args, axes)):

        if args is None:
            kwargs = {}
            label = "No regularization"
        else:
            width, *_ = args
            kwargs = dict(Λ=construct_matern_32_weight_vector(N, width)**-2)
            label = f"$s = {width}$"

        μ, Σ = model.solve(f_obs, 0.01 * np.ones_like(f_obs), **kwargs)
        
        
        f_pred = model.predict_magnitude(x_true)

        fi = model.predict_magnitude(xi)

        if i == 0:
            #axes[0].plot(xi, yi, c="#666666", lw=2, label="Truth", ms=0, zorder=-1)
            axes[0].scatter(x_true, f_obs, c="k", s=100, zorder=1000, label=f"Data ($N = 23$)")

        c = colors[i]
        axes[0].scatter(x_true, f_pred, c=c, s=30, zorder=50 + i)
        draws = generate_complex_mvn_samples(μ, np.diag(Σ), 30)
        for draw in draws:
            axes[0].plot(xi, model.predict_magnitude(xi, draw), c=c, alpha=0.1)
        
        axes[0].plot(xi, fi, c=c, lw=1, ms=0, label=label)
        axes[0].set_xlim(-np.pi, +np.pi)
        axes[0].set_xlabel(r"$t$")
        axes[1].set_ylabel(r"$f$")
        v = np.sqrt(np.abs(Σ))
        m = np.sqrt(np.abs(μ.real)**2 + np.abs(μ.imag)**2)
        axes[1].plot(m, c=c)
        axes[1].fill_between(np.arange(m.size), m - v, m + v, facecolor=c, alpha=0.5)
        axes[1].set_xlabel(r"Fourier mode index")
        axes[1].set_ylabel(r"Strength (magnitude)")
        axes[1].set_xlim(0, N)
        # axes[2].plot(np.abs(result.imag), c=c)


        #ax_error.plot(xi, fi - f(xi), c="tab:orange", ls=":", lw=1, label="Error")
        #axes[0].legend()
    axes[0].set_xlabel(f"$x$")
    axes[0].set_ylabel(f"$y$")
    axes[0].legend()
    fig.tight_layout()
        
