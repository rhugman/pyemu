from .base import Emulator
from .dsi import DSI
from .transformers import (
    BaseTransformer,
    Log10Transformer,
    RowWiseMinMaxScaler,
    StandardScalerTransformer,
    NormalScoreTransformer,
    TransformerPipeline,
    AutobotsAssemble
)

# Try to import TensorFlow-dependent classes
# Users will need to install tensorflow with:
# pip install pyemu[emulators-dsiae] or pip install pyemu[emulators-ldfa] or pip install pyemu[emulators-all]
try:
    import tensorflow
    from .dsiae import DSIAE
    from .ldfa import LDFA
    _HAS_TENSORFLOW = True
except ImportError:
    # Create placeholder classes that raise informative errors when instantiated
    def _tensorflow_not_installed_error(class_name, extra_dependency):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"The {class_name} class requires TensorFlow, which is not installed. "
                f"Install it with 'pip install pyemu[{extra_dependency}]' or 'pip install tensorflow'."
            )
        return type(class_name, (), {"__init__": __init__})

    DSIAE = _tensorflow_not_installed_error("DSIAE", "emulators-dsiae")
    LDFA = _tensorflow_not_installed_error("LDFA", "emulators-ldfa")
    _HAS_TENSORFLOW = False

__all__ = [
    'Emulator',
    'DSI',
    'LDFA',  # Will be a placeholder class if tensorflow is not installed
    'DSIAE',  # Will be a placeholder class if tensorflow is not installed
    'BaseTransformer',
    'Log10Transformer',
    'RowWiseMinMaxScaler',
    'StandardScalerTransformer',
    'NormalScoreTransformer',
    'TransformerPipeline',
    'AutobotsAssemble'
]
