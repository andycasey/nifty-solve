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
        self._halfish_n_modes = self.n_modes // 2 + (self.n_modes % 2)
        return None
    
    def _matvec(self, c):
        return self._plan_matvec.execute(
            np.hstack([
                c[:self._halfish_n_modes],
                c[self._halfish_n_modes:] * -1j
            ])
        ).real
    
    def _rmatvec(self, f):
        v = self._plan_rmatvec.execute(f.astype(np.complex128))
        return np.hstack([v.real[:self._halfish_n_modes], -v.imag[self._halfish_n_modes:]])


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
        
        # Generate a permutation matrix to put parameters in real/imaginary form.
        p_min = min(p_x, p_y)
        self.permute = np.ones(self.n_modes).astype(np.complex128)
        self.permute[np.triu_indices(p_min, k=1)] = -1j
        self.permute[np.diag_indices(p_min)] = 1
        self.permute[np.diag_indices(p_min)] = np.hstack([np.ones(p_min // 2 + (p_min % 2)), -1j * np.ones(p_min // 2)])
        self.mask_permute_real = (self.permute.real == 1)
        return None

    def _matvec(self, c):
        return self._plan_matvec.execute(c.reshape(self.n_modes) * self.permute).real
    
    def _rmatvec(self, f):
        r = self._plan_rmatvec.execute(f.astype(np.complex128))
        v = np.zeros(self.n_modes)
        v[self.mask_permute_real] = r[self.mask_permute_real].real
        v[~self.mask_permute_real] = -r[~self.mask_permute_real].imag
        return v


class Finufft3DRealOperator(LinearOperator):

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, n_modes: Union[int, tuple[int, int, int]], **kwargs):
        """
        A linear operator to fit a model to real-valued 3D signals with sine and cosine functions
        using the Flatiron Institute Non-Uniform Fast Fourier Transform.

        :param x:
            The x-coordinates of the data. This should be within the domain (0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain (0, 2π).

        :param z:
            The z-coordinates of the data. This should be within the domain (0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to the `finufft.Plan()` constructor. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        p_x, p_y, p_z = expand_to_dim(n_modes, 3)
        if len(x) != len(y) or len(x) != len(z) or len(y) != len(z):
            raise ValueError("The number of x, y, and z coordinates must be the same.")        
        super().__init__(dtype=np.float64, shape=(len(x), p_x * p_y * p_z))
        self.n_modes = (p_x, p_y, p_z)
        self.explicit = False
        self._plan_matvec = finufft.Plan(2, self.n_modes, modeord=1, **kwargs)
        self._plan_rmatvec = finufft.Plan(1, self.n_modes, modeord=1, **kwargs)
        self._plan_matvec.setpts(x, y, z)
        self._plan_rmatvec.setpts(x, y, z)
        
        # Generate a permutation matrix to put parameters in real/imaginary form.
        p_min = min(p_x, p_y, p_z)
        self.permute = np.ones(self.n_modes).astype(np.complex128)
        self.permute[np.triu_indices(p_min, k=1)] = -1j
        self.permute[np.diag_indices(p_min)] = 1
        self.permute[np.diag_indices(p_min)] = np.hstack([np.ones(p_min // 2 + (p_min % 2)), -1j * np.ones(p_min // 2)])
        self.mask_permute_real = (self.permute.real == 1)
        return None

    def _matvec(self, c):
        return self._plan_matvec.execute(c.reshape(self.n_modes) * self.permute).real
    
    def _rmatvec(self, f):
        r = self._plan_rmatvec.execute(f.astype(np.complex128))
        v = np.zeros(self.n_modes)
        v[self.mask_permute_real] = r[self.mask_permute_real].real
        v[~self.mask_permute_real] = -r[~self.mask_permute_real].imag
        return v



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

