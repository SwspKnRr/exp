"""Machine learning modules for signal prediction and filtering."""
from .model import MomentumClassifier
from .filter import MLSignalFilter

__all__ = [
    'MomentumClassifier',
    'MLSignalFilter'
]
