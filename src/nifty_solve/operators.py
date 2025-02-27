from typing import Optional, Union

import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator

ensure_odd = lambda x: x + ((x + 1) % 2)

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
        self.n_modes = tuple(map(ensure_odd, n_requested_modes))
        # We store the finufft kwds on the object in case we want to create another operator to evalaute at different points.
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

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_matvec(c))

    def _rmatvec(self, f):
        return self._post_rmatvec(self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX)))


class Finufft1DRealOperator(FinufftRealOperator):
    def __init__(self, x: npt.ArrayLike, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, n_modes=n_modes, **kwargs)

    def _pre_matvec(self, c):
        h = self.shape[1] // 2
        f = np.zeros(self.n_modes, dtype=self.DTYPE_COMPLEX)
        f[:h+1] = 0.5 * (c[:h+1] + np.hstack([c[h+1:], np.zeros(2*h+2-c.size)]) * 1j)
        f += np.conj(np.flip(f))
        return f

    def _post_rmatvec(self, f):
        h = len(f) // 2
        c = np.hstack([f[:h+1].real, f[:h].imag])
        return c[:self.shape[1]] # Ignore hidden modes
 

class Finufft2DRealOperator(FinufftRealOperator):
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        n_modes: Union[int, tuple[int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 2D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, y, n_modes=n_modes, **kwargs)
        return None

    def _pre_matvec(self, c):        
        p = np.prod(self.n_modes)
        h = self.shape[1] // 2
        f = (
            0.5  * np.hstack([c[:h+1], np.zeros(p - h - 1)]) 
        +   0.5j * np.hstack([np.zeros(p - c.size + h + 1), c[h+1:]])
        )
        f = f.reshape(self.n_modes)
        return f + np.conj(np.flip(f))


    def _post_rmatvec(self, f):
        h = self.shape[1] // 2
        f_flat = f.flatten()
        v = np.hstack([
            f_flat[:h].real,
            f_flat[h].real,
            -np.flip(f_flat[:h-(1 - self.shape[1] % 2)].imag)
        ])[:self.shape[1]]
        return v



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
        A linear operator to fit a model to real-valued 3D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param z:
            The z-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor.
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, y, z, n_modes=n_modes, **kwargs)
        self._Hx, self._Hy, self._Hz = tuple(map(_halfish, self.n_modes))

    def _pre_matvec(self, c):
        c = c.reshape(self.n_modes)
        f = -1j * c.astype(self.DTYPE_COMPLEX)
        f[self._Hx :, self._Hy :, self._Hz :] = c[self._Hx :, self._Hy :, self._Hz :]
        return f

    def _post_rmatvec(self, f):
        c = -f.imag
        c[self._Hx :, self._Hy :, self._Hz :] = f[
            self._Hx :, self._Hy :, self._Hz :
        ].real
        return c

def expand_to_dim(n_modes, n_dims):
    if isinstance(n_modes, int):
        return (n_modes,) * n_dims
    else:
        if isinstance(n_modes, (tuple, list, np.ndarray)):
            if len(n_modes) == n_dims:
                return tuple(n_modes)
            else:
                raise ValueError(
                    f"Number of modes must be an integer or a tuple of length {n_dims}."
                )
        else:
            raise TypeError(
                f"Number of modes must be an integer or a tuple of integers."
            )


def _halfish(P: int):
    return P // 2
