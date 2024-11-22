import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator, Diagonal
from typing import Optional, Union

class FinufftRealOperator(LinearOperator):

    def __init__(self, *points, n_modes: Union[int, tuple[int]], **kwargs):
        if len(set(map(len, points))) != 1:
            raise ValueError("All point arrays must have the same length.")
        n_modes = expand_to_dim(n_modes, len(points))
        super().__init__(dtype=np.float64, shape=(len(points[0]), int(np.prod(n_modes))))
        self.explicit = False
        self._plan_matvec = finufft.Plan(2, n_modes, modeord=1, **kwargs)
        self._plan_rmatvec = finufft.Plan(1, n_modes, modeord=1, **kwargs)
        self._plan_matvec.setpts(*points)
        self._plan_rmatvec.setpts(*points)
        self._permute_mask = permute_mask(*n_modes)
        self._permute_mask_flat = self._permute_mask.flatten()
        return None
    
    def _pre_process_matvec(self, c):
        f = np.zeros(self._permute_mask.shape, dtype=np.complex128)
        f[self._permute_mask] = c[self._permute_mask_flat]
        f[~self._permute_mask] = -1j * c[~self._permute_mask_flat]
        return f        
    
    def _post_process_rmatvec(self, f):
        p = np.zeros(f.shape, dtype=np.float64)
        p[self._permute_mask] = f[self._permute_mask].real
        p[~self._permute_mask] = -f[~self._permute_mask].imag
        return p

    def _matvec(self, c):
        return self._plan_matvec.execute(self._pre_process_matvec(c)).real

    def _rmatvec(self, f):
        return self._post_process_rmatvec(self._plan_rmatvec.execute(f.astype(np.complex128)))

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
        return None
    
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
        return None

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
        return None

def permute_mask(*P):
    args = [1] + list(map(_permute_mask, P))
    while len(args) > 2:
        args[-2:] = [np.kron(args[-2], args[-1])]
    return np.kron(*args).reshape(P).astype(bool)

def _permute_mask(P):
    H = P // 2 + (P % 2)
    return np.hstack([np.ones(H), np.zeros(P - H)])

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