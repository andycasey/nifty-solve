

def test_1d_real_operator():

    import numpy as np
    from pylops.utils import dottest

    from nifty_solve.operators import Finufft1DRealOperator
    

    def design_matrix_as_is(xs, P):
        """
        Take in a set of x positions and return the Fourier design matrix.

        ## Bugs:
        - Needs comment header.
        
        ## Comments:
        - The code looks different from the paper because Python zero-indexes.
        - This could be replaced with something that makes use of finufft.
        """
        X = np.ones_like(xs).reshape(len(xs), 1)
        for j in range(1, P):
            if j % 2 == 0:
                X = np.concatenate((X, np.cos(j * xs)[:, None]), axis=1)
            else:
                X = np.concatenate((X, np.sin((j + 1) * xs)[:, None]), axis=1)
        return X
        
    NPs = [
        (1200, 11),
        (1200, 12),
        (1201, 11),
        (1201, 12),
        (171, 338),
        (171, 339),
        (170, 338),
        (170, 339)
    ]

    for N, P in NPs:
        x = np.linspace(-np.pi, np.pi, N)
        A = Finufft1DRealOperator(x, P)

        dottest(A)

        mode_indices = np.zeros(P, dtype=int)
        mode_indices[2::2] = np.arange(1, P//2 + (P % 2))
        mode_indices[1::2] = np.arange(P//2 + (P % 2), P)[::-1]

        A1 = design_matrix_as_is(x/2, P)
        assert np.allclose(A.todense()[:, mode_indices], A1)