from typing import Optional, Union

import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator


class FinufftRealOperator(LinearOperator):
    def __init__(self, *points, n_modes: Union[int, tuple[int]], **kwargs):
        if len(set(map(len, points))) != 1:
            raise ValueError("All point arrays must have the same length.")
        self.n_modes = expand_to_dim(n_modes, len(points))

        if points[0].dtype == np.float64:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float64, np.complex128)
        else:
            self.DTYPE_REAL, self.DTYPE_COMPLEX = (np.float32, np.complex64)

        super().__init__(
            dtype=self.DTYPE_REAL,
            shape=(len(points[0]), int(np.prod(self.n_modes))),
        )
        self.explicit = False
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
        return self._plan_matvec.execute(self._pre_process_matvec(c))

    def _rmatvec(self, f):
        return self._post_process_rmatvec(
            self._plan_rmatvec.execute(f.astype(self.DTYPE_COMPLEX))
        )


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
        super().__init__(
            x, 
            n_modes=n_modes, 
            # Ensure FINUFFT has an odd number of modes for symmetry so predicted values are all real
            n_modes_or_dim=(n_modes + 1 - n_modes % 2, ),
            **kwargs
        )

    def _pre_matvec(self, c):
        p, = self.n_modes
        h = p // 2
        f = np.zeros(p + 1 - p % 2, dtype=self.DTYPE_COMPLEX)
        f[:h+1] = 0.5 * (c[:h+1] + np.hstack([c[h+1:], np.zeros(2 - p % 2)]) * 1j)
        f += np.conj(np.flip(f))
        return f

    def _post_rmatvec(self, f):
        p, = self.n_modes
        h = p // 2
        c = np.hstack([f[:h+1].real, f[:h].imag])
        return c[:p] # Ignore hidden mode for even number modes


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

        # For assembling basis vector into full complex (but conjugate symmetric) modes
        if self.n_modes[0] != self.n_modes[1]:
            raise NotImplementedError(
                "Differing numbers of modes in each dimensions is currently not supported."
            )
        if self.n_modes[0] % 2 != 1:
            raise NotImplementedError(
                "Even numbers of modes is currently not supported."
            )
        self.inds_tril = np.tril_indices(n=self.n_modes[0], m=self.n_modes[1], k=-1)

    def _reverse_reordering(self, f, coefs=True):
        # Repackages mode weights in reverse of matvec ASSUMING conjugate symmetry in f
        # Shape
        N_x, N_y = self.n_modes
        f_ = f.reshape((N_x, N_y))
        # Diagonal
        real_comp_diag = np.flip(np.diag(f_).real[: N_x // 2 + 1])
        imag_comp_diag = np.flip(np.diag(f_).imag[: N_x // 2])
        # Lower triangle
        real_comp_tril = f_[self.inds_tril].real
        imag_comp_tril = f_[self.inds_tril].imag
        # Join
        if coefs:
            return np.concatenate(
                [real_comp_diag, real_comp_tril, imag_comp_diag, imag_comp_tril]
            )
        # This is just so we can reuse the function to get the mode order
        else:
            return np.concatenate(
                [real_comp_diag, real_comp_tril, real_comp_diag[1:], real_comp_tril]
            )

    def get_mode_freqs(self):
        """
        Get the x and y frequencies for the modes of the basis, ordered as per the input
        vector to matvec. We do some processing and reordering to enforce symmetries
        before giving the input to fiNUFFT, and so the fiNUFFT mode ordering does not
        correspond 1-to-1 with the input to matvec. Use the weights from this function
        if you want to regularise the mode weights by frequency.
        """
        N_x, N_y = self.n_modes
        # Start in fiNUFFT order
        xmodes = np.arange(N_x) - ((N_x - 1) / 2)
        ymodes = np.arange(N_x) - ((N_y - 1) / 2)
        xmodes, ymodes = np.meshgrid(xmodes, ymodes, indexing="ij")
        # Reorder with reverse method
        xmodes = self._reverse_reordering(xmodes, coefs=False)
        ymodes = self._reverse_reordering(ymodes, coefs=False)
        return xmodes, ymodes

    def _pre_process_matvec(self, c):
        N_x, N_y = self.n_modes
        # Split the input into halves, treating the entries as either purely real or
        # purely imaginary
        real_comp = c[: N_x * N_y // 2 + 1].real
        imag_comp = c[N_x * N_y // 2 + 1 :].real
        # Construct the diagonals using symmetry (real) and antisymmetry (imagianry)
        modes_coefs_diag_real = np.zeros((N_x), dtype=self.DTYPE_REAL)
        modes_coefs_diag_real[: N_x // 2 + 1] = 0.5 * np.flip(real_comp[: N_x // 2 + 1])
        modes_coefs_diag_real += np.flip(modes_coefs_diag_real)
        modes_coefs_diag_imag = np.zeros((N_x), dtype=self.DTYPE_REAL)
        modes_coefs_diag_imag[: N_x // 2] = 0.5 * np.flip(imag_comp[: N_x // 2])
        modes_coefs_diag_imag += -np.flip(modes_coefs_diag_imag)
        # Do the same for non-diagonals
        mode_coefs_real = np.zeros((N_x, N_y), dtype=self.DTYPE_REAL)
        mode_coefs_real[self.inds_tril] = 0.5 * real_comp[N_x // 2 + 1 :]
        mode_coefs_real += np.flip(mode_coefs_real)
        mode_coefs_imag = np.zeros((N_x, N_y), dtype=self.DTYPE_REAL)
        mode_coefs_imag[self.inds_tril] = 0.5 * imag_comp[N_x // 2 :]
        mode_coefs_imag += -np.flip(mode_coefs_imag)
        # Combine real + imag
        return (
            mode_coefs_real.astype(self.DTYPE_COMPLEX)
            + mode_coefs_imag.astype(self.DTYPE_COMPLEX) * 1j
            + np.diag(modes_coefs_diag_real).astype(self.DTYPE_COMPLEX)
            + np.diag(modes_coefs_diag_imag).astype(self.DTYPE_COMPLEX) * 1j
        )  # / (np.pi * np.sqrt(N_x * N_y))

    def _post_process_rmatvec(self, f):
        return self._reverse_reordering(f).astype(self.DTYPE_REAL)


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

    def _pre_process_matvec(self, c):
        c = c.reshape(self.n_modes)
        f = -1j * c.astype(self.DTYPE_COMPLEX)
        f[self._Hx :, self._Hy :, self._Hz :] = c[self._Hx :, self._Hy :, self._Hz :]
        return f

    def _post_process_rmatvec(self, f):
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
