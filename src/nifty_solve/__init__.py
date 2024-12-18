try:
    from .jax_operators import (
        JaxFinufft1DRealOperator,
        JaxFinufft2DRealOperator,
        JaxFinufft3DRealOperator,
    )
except ImportError:
    # No JAX option installed. Maybe we should create dummy classes and raise NotAvailableError on init?
    None

from .operators import (
    Finufft1DRealOperator,
    Finufft2DRealOperator,
    Finufft3DRealOperator,
)
