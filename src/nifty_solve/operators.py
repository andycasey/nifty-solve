import finufft
import numpy as np
import numpy.typing as npt
from pylops import LinearOperator, Diagonal
from typing import Optional

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

