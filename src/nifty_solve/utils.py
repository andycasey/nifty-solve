import numpy as np

def odd_ceiling(x):
    return x + ((x + 1) % 2)

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

