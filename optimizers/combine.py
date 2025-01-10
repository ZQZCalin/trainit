"""Extends/simplifies some combine methods in optax."""

import jax
import optax

def multi_transform() -> optax.GradientTransformation:
    """Simplifies `optax.multi_transform`.
    
    
    """