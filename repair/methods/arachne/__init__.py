# Import PyTorch version (main version)
from .arachne_pytorch import ArachnePyTorch

# TensorFlow version - import only when needed to avoid dependency issues
# from .arachne import Arachne

__all__ = ("ArachnePyTorch",)
# __all__ = ("Arachne", "ArachnePyTorch")  # Uncomment if TensorFlow version is needed
