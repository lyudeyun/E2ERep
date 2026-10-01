"""Search algorithms used by Arachne to optimize the localized weights."""

from .de_optimizer import DEOptimizer
from .pso_optimizer import PSOOptimizer

__all__ = ['PSOOptimizer', 'DEOptimizer', 'get_optimizer']

ALGORITHM_REGISTRY = {
    'PSO': PSOOptimizer,
    'DE': DEOptimizer,
}


def get_optimizer(algorithm_name):
    """Return the optimizer class registered under `algorithm_name` (case-insensitive)."""
    algorithm_name = algorithm_name.upper()
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")
    return ALGORITHM_REGISTRY[algorithm_name]
