from __future__ import annotations

import finufft
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as sp
from dataclasses import dataclass
from typing import Literal, Optional, Union
from functools import cached_property
from timeit import default_timer as timer

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
        finufft_kwargs: Optional[dict] = None, 
        **kwargs
    ):
        self._check_sizes(f)
        
        finufft_kwargs = finufft_kwargs or {}        
        A = finufft.Plan(2, (self.n_modes, ), **finufft_kwargs)
        AT = finufft.Plan(1, (self.n_modes, ), **finufft_kwargs)
        A.setpts(self.x)
        AT.setpts(self.x)

        

        if f_err is None:            
            Y = f
            lo = sp.LinearOperator(
                (len(self.x), self.n_modes),
                matvec=A.execute,
                rmatvec=AT.execute,
                dtype=complex
            )            
        else:
            # X = (A.T @ C^(-1) @ A)^(-1) @ A.T @ C^(-1) @ Y
            # Solve:
            #   (A.T @ C^(-1) @ A) @ X = A.T @ C^(-1) @ Y
            Cinv = (f_err**-2)
            L = np.linalg.cholesky(np.diag(Cinv + 1e-12))

            Y = f @ L
            assert np.isfinite(Cinv).all()
            
            def matvec(c):
                return A.execute(c) @ L

            def rmatvec(f):
                return AT.execute(f @ L)
            
            lo = sp.LinearOperator(
                (len(self.x), self.n_modes),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=complex
            )
        
        solve_time = -timer()
        self.c, *v = sp.lsqr(lo, Y, **kwargs)
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
    
    def predict(self, x: npt.ArrayLike | None = None) -> npt.ArrayLike:
        return finufft.nufft1d2(x if x is not None else self.x, self.c)



if __name__ == "__main__":


    N = 20 # number of modes
    M = 300 # number of data points
    random = np.random.default_rng(3)
    x_true = np.sort(2. * np.pi * random.uniform(size=M) - np.pi)
    def f(x):
        return np.cos(x) + np.cos(2*x) + np.sin(4*x)

    f_true = f_obs = f(x_true).astype(complex)
    f_err = 0.1 * np.ones(x_true.size)
    f_err[-50:] = np.inf
    #f_err = 0.03 * np.ones_like(f_true)
    #f_err = 0.5 * np.random.rand(x_true.size)
    #f_obs = f_true + f_err * np.random.rand(x_true.size)
    #f_err = np.abs(f_err)

    model = Fourier1DBasis(x_true, N)
    result = model.solve(f_obs, f_err)

    f_pred = model.predict(x_true)

    xi = np.linspace(-np.pi, np.pi, 1000)
    fi = model.predict(xi)

    import matplotlib.pyplot as plt


    fig, (ax_error, ax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 4]))

    ax.plot(xi, f(xi), c="tab:blue", lw=2, label="Truth", ms=0)
    ax.scatter(x_true, f_obs, c="k", s=100, label=f"Noiseless (non-uniform) data samples ($M={M}$)")
    ax.scatter(x_true, f_pred, c="tab:orange", s=30, label=f"Predicted samples ($N={N}$)")
    #ax.plot(xi, fi, c="tab:orange", ls=":", lw=1, label="Predicted model", ms=0)

    #ax_error.plot(xi, fi - f(xi), c="tab:orange", ls=":", lw=1, label="Error")
    ax_error.set_xticklabels([])
    ax_error.set_ylabel(r"$\Delta{}y$")

    ylim_max = np.max(np.abs(ax_error.get_ylim()))
    ax_error.set_ylim(-ylim_max, ylim_max)
    ax_error.axhline(0, c="#666666", ls=":", lw=0.5, ms=0)

    ax.legend()
    ax.set_xlabel(f"$x$")
    ax.set_ylabel(f"$y$")
    #ax_error.set_title(f"$s = {s:.3f}$ $\lambda = {λ:.1f}$")
    fig.tight_layout()


    # X = (A.T @ C^(-1) @ A)^(-1) @ A.T @ C^(-1) @ f
    # (A.T @ Cinv @ A) @ X = A.T @ Cinv @ f
    # A.T @ Cinv @ f = (A.T @ Cinv @ A) @ X
    # Y = B @ X
    #   Y = A.T @ Cinv @ f
    #   B = A.T @ Cinv @ A
    #   B = D @ A   where D = A.T @ Cinv
    #   B.T = A.T @ D.T 