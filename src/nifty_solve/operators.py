import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator, Diagonal
from typing import Optional, Union


class Finufft1DRealOperator(LinearOperator):

    def __init__(self, x: npt.ArrayLike, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain (0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(dtype=np.float64, shape=(len(x), n_modes))
        self.n_modes = n_modes
        self.explicit = False
        self._plan_matvec = finufft.Plan(2, (n_modes, ), modeord=1, **kwargs)
        self._plan_rmatvec = finufft.Plan(1, (n_modes, ), modeord=1, **kwargs)
        self._plan_matvec.setpts(x)
        self._plan_rmatvec.setpts(x)
        self._re_im_mask = real_imag_mask_1d(n_modes)
        return None
    
    def _matvec(self, c):
        return self._plan_matvec.execute(to_complex(c, self._re_im_mask)).real
    
    def _rmatvec(self, f):
        return to_real(self._plan_rmatvec.execute(f.astype(np.complex128)), self._re_im_mask)


class Finufft2DRealOperator(LinearOperator):
    
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, n_modes: Union[int, tuple[int, int]], **kwargs):
        """
        A linear operator to fit a model to real-valued 2D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain (0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain (0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        p_x, p_y = expand_to_dim(n_modes, 2)
        if len(x) != len(y):
            raise ValueError("The number of x and y coordinates must be the same.")        
        super().__init__(dtype=np.float64, shape=(len(x), p_x * p_y))
        self.n_modes = (p_x, p_y)
        self.explicit = False
        self._plan_matvec = finufft.Plan(2, self.n_modes, modeord=1, **kwargs)
        self._plan_rmatvec = finufft.Plan(1, self.n_modes, modeord=1, **kwargs)
        self._plan_matvec.setpts(x, y)
        self._plan_rmatvec.setpts(x, y)
        self._re_im_mask = real_imag_mask_2d(p_x, p_y)
        return None

    def _matvec(self, c):
        return self._plan_matvec.execute(to_complex(c.reshape(self.n_modes), self._re_im_mask)).real
    
    def _rmatvec(self, f):
        return to_real(self._plan_rmatvec.execute(f.astype(np.complex128)), self._re_im_mask)


def real_imag_mask_1d(P):
    H = P // 2 + (P % 2)
    return np.hstack([np.ones(H), np.zeros(P - H)]).astype(bool)

def real_imag_mask_2d(Px, Py):
    return np.outer(real_imag_mask_1d(Px), real_imag_mask_1d(Py))

def to_complex(c, is_real):
    f = np.zeros(c.shape, dtype=np.complex128)
    f[is_real] = c[is_real]
    f[~is_real] = -1j * c[~is_real]
    return f

def to_real(f, is_real):
    p = np.zeros(f.shape, dtype=np.float64)
    p[is_real] = f[is_real].real
    p[~is_real] = -f[~is_real].imag
    return p


def expand_to_dim(n_modes, n_dims):
    if isinstance(n_modes, int):
        return (n_modes, ) * n_dims
    else:
        if isinstance(n_modes, (tuple, list, np.array)):
            if len(n_modes) == n_dims:
                return tuple(n_modes)
            else:
                raise ValueError(f"Number of modes must be an integer or a tuple of length {n_dims}.")
        else:
            raise TypeError(f"Number of modes must be an integer or a tuple of integers.")

