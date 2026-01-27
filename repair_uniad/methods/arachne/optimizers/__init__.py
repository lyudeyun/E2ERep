"""
Optimization algorithms for Arachne neural network repair.

This module provides various optimization algorithms that can be used
to optimize neural network weights during repair.
"""

# Handle both relative and absolute imports
try:
    from .pso_optimizer import PSOOptimizer
    from .de_optimizer import DEOptimizer
except ImportError:
    # When used as standalone module (methods/arachne added to sys.path)
    from pso_optimizer import PSOOptimizer
    from de_optimizer import DEOptimizer

__all__ = ['PSOOptimizer', 'DEOptimizer']

# Algorithm registry for easy lookup
ALGORITHM_REGISTRY = {
    'PSO': PSOOptimizer,
    'DE': DEOptimizer,
}

def get_optimizer(algorithm_name):
    """
    Get optimizer class by name.
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm ('PSO', 'DE', etc.)
    
    Returns
    -------
    optimizer_class : class
        Optimizer class
    
    Raises
    ------
    ValueError
        If algorithm name is not found
    """
    algorithm_name = algorithm_name.upper()
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")
    return ALGORITHM_REGISTRY[algorithm_name]

