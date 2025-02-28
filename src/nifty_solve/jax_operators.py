# jax_operators.py

import warnings
import numpy as np
from typing import Union
from functools import cached_property

try:
    import jax.numpy as jnp
    from jax_finufft import nufft1, nufft2
except ImportError:
    warnings.warn(
        "jax or jax_finufft not installed, required for jax operators.",
        ImportWarning,
    )
    # Need to assign jnp as np because jnp is used in type annotations.
    jnp = np
    import finufft
    
    FINUFFT_MAPPING = {
        1: (finufft.nufft1d1, finufft.nufft1d2),
        2: (finufft.nufft2d1, finufft.nufft2d2),
        3: (finufft.nufft3d1, finufft.nufft3d2),
    }

    def nufft1(c, *p, **kwargs):
        f1, f2 = FINUFFT_MAPPING[c.ndim]
        return f1(c, *p, **kwargs)

    def nufft2(c, *p, **kwargs):
        f1, f2 = FINUFFT_MAPPING[c.ndim]
        return f2(c, *p, **kwargs)

from pylops import JaxOperator, LinearOperator

from .utils import odd_ceiling, expand_to_dim


class BaseMeta(type):
    """
    Metaclass to easily wrap our custom operators with pylops' JaxOperator class.
    """

    def __call__(cls, *args, **kwargs):
        # Create the instance normally
        instance = super().__call__(*args, **kwargs)
        # Return the wrapped instance
        return JaxOperator(instance)


class CombinedMeta(type(LinearOperator), BaseMeta):
    """
    Combine our metaclass with LinearOperator for compatibility.
    """

    pass


# Base class
class JaxFinufftRealOperator(LinearOperator, metaclass=CombinedMeta):
    """
    Base class for jax operators.
    """

    def __init__(self, *points, n_modes: Union[int, tuple[int]], **kwargs):
        if len(set(map(len, points))) != 1:
            raise ValueError("All point arrays must have the same length.")
        
        self.DTYPE_REAL = jnp.array(0.0).dtype
        self.DTYPE_COMPLEX = jnp.array(0.0 + 0.0j).dtype

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
            # TODO: Move these to `opts`? Check jax finufft documentation.
            #n_modes_or_dim=self.n_modes,
            #n_trans=1,
            eps=1e-6,
            #isign=None,
            #dtype=self.DTYPE_COMPLEX.__name__,
            #modeord=0,
        )
        self.finufft_kwds.update(kwargs)
        self.points = points

    def _matvec(self, c):
        return nufft2(self._pre_matvec(c), *self.points, **self.finufft_kwds).real

    def _rmatvec(self, f):
        return self._post_rmatvec(
            nufft1(
                self.n_modes, 
                f.astype(self.DTYPE_COMPLEX), 
                *self.points, 
                **self.finufft_kwds
            )
        )

    def _pre_matvec(self, c):
        m, h, p = self._shape_half_p
        f = (
            0.5  * jnp.hstack([c[:h+1],   jnp.zeros(p-h-1)])
        +   0.5j * jnp.hstack([jnp.zeros(p-m+h+1), c[h+1:]])
        )
        f = f.reshape(self.n_modes)
        return f + jnp.conj(jnp.flip(f))

    def _post_rmatvec(self, f):
        m, h, _ = self._shape_half_p
        f_flat = f.flatten()
        return jnp.hstack([f_flat[:h+1].real, f_flat[-(m-h-1):].imag])

    @cached_property
    def _shape_half_p(self):
        return (self.shape[1], self.shape[1] // 2, int(np.prod(self.n_modes)))

class JaxFinufft1DRealOperator(JaxFinufftRealOperator):

    def __init__(self, x: jnp.ndarray, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and
        cosine functions using jax bindings to the Flatiron Institute Non-Uniform 
        Fast Fourier Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, n_modes=n_modes, **kwargs)

class JaxFinufft2DRealOperator(JaxFinufftRealOperator):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        n_modes: Union[int, tuple[int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 2D signals with sine and
        cosine functions using jax bindings to the Flatiron Institute Non-Uniform 
        Fast Fourier Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, y, n_modes=n_modes, **kwargs)

class JaxFinufft3DRealOperator(JaxFinufftRealOperator):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
        n_modes: Union[int, tuple[int, int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 3D signals with sine and
        cosine functions using jax bindings to the Flatiron Institute Non-Uniform 
        Fast Fourier Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param z:
            The z-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. 
            Note that the mode ordering keyword `modeord` cannot be supplied.
        """
        return super().__init__(x, y, z, n_modes=n_modes, **kwargs)