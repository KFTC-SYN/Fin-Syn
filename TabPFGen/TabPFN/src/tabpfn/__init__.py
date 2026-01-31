from importlib.metadata import version
import sys
from pathlib import Path

# Add TabPFN/src to Python path so that tabpfn module can be imported
# This allows imports like "from TabPFN.src.tabpfn import TabPFNClassifier"
_tabpfn_src_path = Path(__file__).parent.parent
if str(_tabpfn_src_path) not in sys.path:
    sys.path.insert(0, str(_tabpfn_src_path))

# Use relative imports to work with local directory structure
from .classifier import TabPFNClassifier
from .misc.debug_versions import display_debug_info
from .model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from .regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
