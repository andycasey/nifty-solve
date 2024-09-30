from __future__ import annotations

import finufft
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as sp
from dataclasses import dataclass
from typing import Literal, Optional, Union
from functools import cached_property
from timeit import default_timer as timer

def construct_matern_32_weight_vector(n_modes: int, s: float = 1, λ: float = 1) -> npt.ArrayLike:
    if λ <= 0:
        raise ValueError("The regularization parameter `λ` must be non-negative.")    
    ω = (np.arange(n_modes) - n_modes // 2)
    a = (λ/((s*ω)**2 + 1))
    return 1/a #(a + a * 1j)

        

class Fourier1DBasis:
    def __init__(self, x: npt.ArrayLike, n_modes: int) -> None:
        self.x = x
        self.n_modes = n_modes
        return None
    
    def _check_sizes(self, f: npt.ArrayLike) -> None:
        if len(f) != len(self.x):
            raise ValueError(f"Input array `f` must have the same length as `x` ({len(self.x)}), but has length {len(f)}.")

    def solve(
        self,
        f: npt.ArrayLike,
        f_err: Optional[npt.ArrayLike] = None,
        Λ: Optional[npt.ArrayLike] = None,
        finufft_kwargs: Optional[dict] = None, 
        **kwargs
    ):
        self._check_sizes(f)
        
        if np.iscomplex(f).any():
            matvec_wrapper = lambda v: v
        else:
            matvec_wrapper = lambda v: v
        
        kwds = dict(eps=1e-16)
        kwds.update(finufft_kwargs or {})
        A = finufft.Plan(2, (self.n_modes, ), **kwds)
        AT = finufft.Plan(1, (self.n_modes, ), **kwds)
        A.setpts(self.x)
        AT.setpts(self.x)

        inv_err = 1 if f_err is None else 1/f_err
        if Λ is not None:
            if True:
                op = np.multiply if Λ.ndim == 1 else np.dot
                matvec = lambda c: matvec_wrapper(np.hstack([A.execute(c) * inv_err, op(Λ, c)]))
                rmatvec = lambda f: AT.execute(f[:len(self.x)] * inv_err) + op(Λ, f[len(self.x):])
                Y = np.hstack([f * inv_err, np.zeros(self.n_modes)])
                shape = (len(self.x) + self.n_modes, self.n_modes)
            else:
                matvec = lambda c: matvec_wrapper(A.execute(c * Λ) * inv_err)
                rmatvec = lambda f: AT.execute(f * inv_err) / Λ
                shape = (len(self.x), self.n_modes)
                Y = f * inv_err
        else:
            Y = f * inv_err
            matvec = lambda c: matvec_wrapper((A.execute(c) * inv_err))
            rmatvec = lambda f: AT.execute(f * inv_err)
            shape = (len(self.x), self.n_modes)
        
        lo = sp.LinearOperator(
            shape,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=complex
        )

        Y = Y.astype(complex)
        solve_time = -timer()
        x0 = lo.rmatvec(Y)
        self.c, *v = sp.lsqr(lo, Y, show=True, atol=0, btol=0, conlim=0, x0=x0, **kwargs)
        solve_time += timer()
        self.meta = dict(
            kwargs=kwargs,
            finufft_kwargs=finufft_kwargs, 
            solve_time=solve_time,            
            lsqr_meta=dict(zip(
                ("istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", "xnorm", "var"), 
                v
            ))
        )
        return self.c
    
    def predict(self, x: npt.ArrayLike | None = None, c: npt.ArrayLike | None = None) -> npt.ArrayLike:
        v = finufft.nufft1d2(x if x is not None else self.x, c if c is not None else self.c)
        return np.sign(v) * np.abs(v)



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
        #(N, 0.01, 1), # to match Hogg & Villar deltaomega choice
        #(N, 0.5, 1),
        #(N, 0.1, 1),
        #(N, 0.05, 1),
        #(N, 0.5, 1),
        #(N, 5e-3, 1),
        (N, 0.05, 1),
        #(N, 2, 1),
        #(N, 0.5, 1),
        #(N, 10.0, 1),
        #(N, 100, 1),
        #(N, 1e-1, 1),
        #(N, 1e-2, 1),
        #(N, 10 * 0.002364489412645407, 1),
        #(N, 1e-3, 1),
        #(N, 0.05 * np.pi/3, 1e-5), # to match Hogg & Villar deltaomega choice\
        #(N, 0.05, 1),
        #(N, 0.025, 1),
        (N, 0.086, 1),
        (N, 0.1, 1),
        #(N, 0.1, 1e-1),
        #(N, 0.09, 1),
        #(N, 0.085, 1),        
        #(N, 0.08, 1),
        #(N, 1e-6, 1),
        #(N, 0.1, 1),
        #(N, 0.01, 100),
        #(N, 1e-4, 1),
        #(N, 1e-5, 1e-3),

        
#        (N, 1e-4, 1e-6),
        #(N, 1e-4, 100),
        #(N, 1e-6, 100),
    ]

    colors = ("tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))#, sharex=True, sharey=True)
    
    for i, args in enumerate(regularization_args):
        # (args, ax) in enumerate(zip(regularization_args, axes)):

        if args is None:
            kwargs = {}
            label="None"
        else:
            kwargs = dict(Λ=construct_matern_32_weight_vector(*args))
            label = f"{args[1]:.2e} {args[2]:.2e}"

        result = model.solve(f_obs, f_err, **kwargs)
        f_pred = model.predict(x_true)

        fi = model.predict(xi)

        if i == 0:
            #axes[0].plot(xi, yi, c="#666666", lw=2, label="Truth", ms=0, zorder=-1)
            axes[0].scatter(x_true, f_obs, c="k", s=100, label=f"Noiseless data samples")

        c = colors[i]
        axes[0].scatter(x_true, f_pred, c=c, s=30)
        axes[0].plot(xi, fi, c=c, lw=1, ms=0, label=label)
        axes[1].plot(np.sqrt(np.abs(result.real)**2 + np.abs(result.imag)**2), c=c)
       # axes[2].plot(np.abs(result.imag), c=c)


        #ax_error.plot(xi, fi - f(xi), c="tab:orange", ls=":", lw=1, label="Error")
        #axes[0].legend()
    axes[0].set_xlabel(f"$x$")
    axes[0].set_ylabel(f"$y$")
    axes[0].legend()
    fig.tight_layout()
        
