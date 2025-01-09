# jax_operators.py

import warnings
from typing import Union

import numpy as np

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

from pylops import JaxOperator, LinearOperator

from .operators import _halfish, expand_to_dim


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
        self.n_modes = expand_to_dim(n_modes, len(points))

        # We want the data types jax is using, typically float32, complex64,
        # But the user might have specified double precision manually so we should check
        self.DTYPE_REAL = jnp.array(0.0).dtype
        self.DTYPE_COMPLEX = jnp.array(0.0 + 0.0j).dtype

        # Set the operator properties
        super().__init__(
            dtype=self.DTYPE_REAL, shape=(len(points[0]), int(np.prod(self.n_modes)))
        )
        self.explicit = False
        self.points = points
        self.kwargs = kwargs

    def _matvec(self, c):
        return nufft2(self._pre_process_matvec(c), *self.points, **self.kwargs).real

    def _rmatvec(self, f):
        return self._post_process_rmatvec(
            nufft1(
                self.n_modes, f.astype(self.DTYPE_COMPLEX), *self.points, **self.kwargs
            )
        )


class JaxFinufft1DRealOperator(JaxFinufftRealOperator):

    def __init__(self, x: jnp.ndarray, n_modes: int, **kwargs):
        """
        A linear operator to fit a model to real-valued 1D signals with sine and cosine
        functions using jax bindings to the Flatiron Institute Non-Uniform Fast Fourier
        Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. Note that the mode
            ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, n_modes=n_modes, **kwargs)
        self._Hx = _halfish(n_modes)

    def _pre_process_matvec(self, c):
        return jnp.hstack([1j * c[: self._Hx], c[self._Hx :]], dtype=self.DTYPE_COMPLEX)

    def _post_process_rmatvec(self, f):
        return jnp.hstack(
            [f[: self._Hx].imag, f[self._Hx :].real], dtype=self.DTYPE_REAL
        )


class JaxFinufft2DRealOperator(JaxFinufftRealOperator):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        n_modes: Union[int, tuple[int, int]],
        **kwargs,
    ):
        """
        A linear operator to fit a model to real-valued 2D signals with sine and cosine
        functions using jax bindings to the Flatiron Institute Non-Uniform Fast Fourier
        Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. Note that the mode
            ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, y, n_modes=n_modes, **kwargs)
        self._H = _halfish(np.prod(self.n_modes))

    def _pre_process_matvec(self, c):
        f = c.astype(self.DTYPE_COMPLEX)
        f = f.at[:self._H].set(-1j * f.at[:self._H].get())
        return f.reshape(self.n_modes)

    def _post_process_rmatvec(self, f):
        return np.hstack([-f[:self._H].get().imag, f[self._H:].get().real], dtype=self.DTYPE_REAL)



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
        A linear operator to fit a model to real-valued 3D signals with sine and cosine
        functions using jax bindings to the Flatiron Institute Non-Uniform Fast Fourier
        Transform (fiNUFFT).

        :param x:
            The x-coordinates of the data. This should be within the domain [0, 2π).

        :param y:
            The y-coordinates of the data. This should be within the domain [0, 2π).

        :param z:
            The z-coordinates of the data. This should be within the domain [0, 2π).

        :param n_modes:
            The number of Fourier modes to use.

        :param kwargs: [Optional]
            Keyword arguments are passed to fiNUFFT via jax_finufft. Note that the mode
            ordering keyword `modeord` cannot be supplied.
        """
        super().__init__(x, y, z, n_modes=n_modes, **kwargs)
        self._Hx, self._Hy, self._Hz = tuple(map(_halfish, self.n_modes))

    def _pre_process_matvec(self, c):
        c = c.reshape(self.n_modes)
        f = 1j * c.astype(self.DTYPE_COMPLEX)
        f = f.at[self._Hx :, self._Hy :, self._Hz :].set(
            c.at[self._Hx :, self._Hy :, self._Hz :].get()
        )
        return f

    def _post_process_rmatvec(self, f):
        c = f.imag
        c = c.at[self._Hx :, self._Hy :, self._Hz :].set(
            f.at[self._Hx :, self._Hy :, self._Hz :].get().real
        )
        return c
