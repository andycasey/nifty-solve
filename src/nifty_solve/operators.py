import finufft
import numpy as np
import numpy.typing as npt
from functools import cached_property
from pylops import LinearOperator
from typing import Optional, Union

from .utils import odd_ceiling, expand_to_dim

class FinufftRealOperator(LinearOperator):
    def __init__(self, *points, n_modes: Union[int, tuple[int]], **kwargs):
        if len(set(map(len, points))) != 1:
            raise ValueError("All point arrays must have the same length.")
        
        if points[0].dtype == np.float64:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float64, np.complex128)
        else:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float32, np.complex64)

        # This is the requested number of modes, but may not be the actual.
        n_requested_modes = expand_to_dim(n_modes, len(points))
        super().__init__(
            dtype=self.DTYPE_REAL,
            shape=(len(points[0]), int(np.prod(n_requested_modes))),
        )
        self.explicit = False
        self.n_modes = tuple(map(odd_ceiling, n_requested_modes))
        # We store the finufft kwds on the object in case we want to create 
        # another operator to evalaute at different points.
        self.finufft_kwds = dict(
            n_modes_or_dim=self.n_modes,
            n_trans=1,
            eps=1e-6,
            isign=None,
            dtype=self.DTYPE_COMPLEX.__name__,
            modeord=0,
        )
        self.finufft_kwds.update(kwargs)
        self._plan_matvec = finufft.Plan(2, **self.finufft_kwds)
        self._plan_rmatvec = finufft.Plan(1, **self.finufft_kwds)
        self._plan_matvec.setpts(*points)
        self._plan_rmatvec.setpts(*points)

    def _pre_matvec(self, c):
        m, h, p = self._shape_half_p
        f = (
            0.5  * np.hstack([c[:h+1],   np.zeros(p-h-1)])
        +   0.5j * np.hstack([np.zeros(p-m+h+1), c[h+1:]])
        )
        f = f.reshape(self.n_modes)
        f += np.conj(np.flip(f))
        return f.astype(self.DTYPE_COMPLEX)

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_matvec(c)).real

    def _post_rmatvec(self, f):
        m, h, _ = self._shape_half_p
        f_flat = f.flatten()
        return np.hstack([f_flat[:h+1].real, f_flat[-(m-h-1):].imag])

    def _rmatvec(self, f):
        return self._post_rmatvec(
            self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX))
        )

    @cached_property
    def _shape_half_p(self):
        return (self.shape[1], self.shape[1] // 2, int(np.prod(self.n_modes)))

class Finufft1DRealOperator(FinufftRealOperator):
    def __init__(self, x: npt.ArrayLike, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and
        cosine functions using the Flatiron Institute Non-Uniform Fast Fourier 
        Transform.

        :param x:
            The x-coordinates of data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, n_modes=n_modes, **kwargs)

class Finufft2DRealOperator(FinufftRealOperator):
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        n_modes: Union[int, tuple[int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 2D signals with sine and
        cosine functions using the Flatiron Institute Non-Uniform Fast Fourier 
        Transform.

        :param x:
            The x-coordinates of data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, y, n_modes=n_modes, **kwargs)

class Finufft3DRealOperator(FinufftRealOperator):
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        z: npt.ArrayLike,
        n_modes: Union[int, tuple[int, int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 3D signals with sine and
        cosine functions using the Flatiron Institute Non-Uniform Fast Fourier 
        Transform.

        :param x:
            The x-coordinates of data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of data. This should be within the domain [0, 2π).

        :param z:
            The z-coordinates of data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, y, z, n_modes=n_modes, **kwargs)