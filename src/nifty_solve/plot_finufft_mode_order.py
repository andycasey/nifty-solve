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
    values = λ/(s**2 * np.arange(-n_modes // 2, n_modes // 2 + 1)[:n_modes]**2 + 1)
    #if np.unique(values).size == 1:
    #    raise ValueError("The regularization parameter `s` is too small.")
    return values

        

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
            matvec_wrapper = lambda v: v.real
        
        kwds = dict(eps=1e-6)
        kwds.update(finufft_kwargs or {})
        A = finufft.Plan(2, (self.n_modes, ), **kwds)
        AT = finufft.Plan(1, (self.n_modes, ), **kwds)
        A.setpts(self.x)
        AT.setpts(self.x)

        inv_err = 1 if f_err is None else 1/f_err
        if Λ is not None:
            assert False
            if True:
                op = np.multiply if Λ.ndim == 1 else np.dot
                matvec = lambda c: np.hstack([A.execute(c) * inv_err, op(Λ, c)])
                rmatvec = lambda f: AT.execute(f[:len(self.x)] * inv_err) + op(Λ, f[len(self.x):])
                Y = np.hstack([f * inv_err, np.zeros(self.n_modes)])
                shape = (len(self.x) + self.n_modes, self.n_modes)
            else:
                matvec = lambda c: A.execute(c * np.sqrt(Λ)) * inv_err
                rmatvec = lambda f: AT.execute(f * inv_err) * np.sqrt(Λ)
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
        return finufft.nufft1d2(x if x is not None else self.x, c if c is not None else self.c)



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    lb, ub = (-np.pi + 0.1, np.pi - 0.1)
    x_true = np.linspace(lb, ub, 10)
    xi = np.linspace(lb, ub, 2048)

    for K in (1, 2, 3, 4, 5):
        model = Fourier1DBasis(x_true, K)

        fig, axes = plt.subplots(K, figsize=(3, 3 * K))
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes):
            ax.plot(xi, np.sin(xi), c="tab:blue", label="sin(x)")
            ax.plot(xi, np.cos(xi), c="tab:red", label="cos(x)")
            v = np.zeros(K).astype(np.complex128)
            v[i] = 1
            ax.plot(xi, model.predict(xi, v), ls=":", label="1+0j")
            v[i] = 0 + 1j
            ax.plot(xi, model.predict(xi, v), ls=":", label="0+1j")
            ax.set_title(f"{i}th entry with K = {K}")    
            if i == 0:
                ax.legend()

        ax.legend()
        fig.tight_layout()
        fig.savefig(f"finufft_mode_order_{K}.png")

