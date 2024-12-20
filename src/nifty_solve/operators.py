import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator, Diagonal
from typing import Optional, Union

class FinufftRealOperator(LinearOperator):
    def __init__(self, *points, n_modes: Union[int, tuple[int]], **kwargs):
        if len(set(map(len, points))) != 1:
            raise ValueError("All point arrays must have the same length.")
        self.n_modes = expand_to_dim(n_modes, len(points))

        if points[0].dtype == np.float64:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float64, np.complex128)
        else:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float32, np.complex64)

        super().__init__(dtype=self.DTYPE_REAL, shape=(len(points[0]), int(np.prod(self.n_modes))))
        self.explicit = False
        # We store the finufft kwds on the object in case we want to create another operator to evalaute at different points.
        self.finufft_kwds = dict(
            n_modes_or_dim=self.n_modes,
            n_trans=1,
            eps=1e-6,
            isign=None,
            dtype=self.DTYPE_COMPLEX.__name__,
            modeord=0
        )
        self.finufft_kwds.update(kwargs) 
        self._plan_matvec = finufft.Plan(2, **finufft_kwds)
        self._plan_rmatvec = finufft.Plan(1, **finufft_kwds)
        self._plan_matvec.setpts(*points)
        self._plan_rmatvec.setpts(*points)

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_process_matvec(c)).real

    def _rmatvec(self, f):
        return self._post_process_rmatvec(self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX)))


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
        self._Hx = _halfish(n_modes)

    def _pre_process_matvec(self, c):
        return np.hstack([-1j * c[:self._Hx], c[self._Hx:]], dtype=self.DTYPE_COMPLEX)

    def _post_process_rmatvec(self, f):
        return np.hstack([-f[:self._Hx].imag, f[self._Hx:].real], dtype=self.DTYPE_REAL)


class Finufft2DRealOperator(FinufftRealOperator):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, n_modes: Union[int, tuple[int, int]], **kwargs):
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
        self._Hx, self._Hy = tuple(map(_halfish, self.n_modes))

    def _pre_process_matvec(self, c):
        c = c.reshape(self.n_modes)
        f = -1j * c.astype(self.DTYPE_COMPLEX)
        f[self._Hx:, self._Hy:] = c[self._Hx:, self._Hy:]
        return f

    def _post_process_rmatvec(self, f):
        c = -f.imag
        c[self._Hx:, self._Hy:] = f[self._Hx:, self._Hy:].real
        return c


class Finufft3DRealOperator(FinufftRealOperator):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, n_modes: Union[int, tuple[int, int, int]], **kwargs):
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

    def _pre_process_matvec(self, c):
        c = c.reshape(self.n_modes)
        f = -1j * c.astype(self.DTYPE_COMPLEX)
        f[self._Hx:, self._Hy:, self._Hz:] = c[self._Hx:, self._Hy:, self._Hz:]
        return f

    def _post_process_rmatvec(self, f):
        c = -f.imag
        c[self._Hx:, self._Hy:, self._Hz:] = f[self._Hx:, self._Hy:, self._Hz:].real
        return c


def expand_to_dim(n_modes, n_dims):
    if isinstance(n_modes, int):
        return (n_modes, ) * n_dims
    else:
        if isinstance(n_modes, (tuple, list, np.ndarray)):
            if len(n_modes) == n_dims:
                return tuple(n_modes)
            else:
                raise ValueError(f"Number of modes must be an integer or a tuple of length {n_dims}.")
        else:
            raise TypeError(f"Number of modes must be an integer or a tuple of integers.")

def _halfish(P: int):
    return P // 2