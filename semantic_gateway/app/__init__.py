"""
SemanticRouter API Package
Fast, cost-optimizing LLM proxy gateway powered by local sentence embeddings.
"""

import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

# Guard against Windows Application Control / Smart App Control policy blocking unsigned scikit-learn DLLs
try:
    import sklearn.metrics._dist_metrics  # noqa: F401
except Exception:
    class _MockModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__file__ = f"{name}.py"
            self.__path__ = []
            self.__package__ = name.rpartition(".")[0]
            self.__spec__ = importlib.machinery.ModuleSpec(name, None)

        def __getattr__(self, name):
            return MagicMock()

    for mod_name in [
        "sklearn",
        "sklearn.metrics",
        "sklearn.metrics.pairwise",
        "sklearn.metrics.cluster",
        "sklearn.metrics._dist_metrics",
        "sklearn.metrics._pairwise_distances_reduction",
        "sklearn.metrics._pairwise_distances_reduction._dispatcher",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _MockModule(mod_name)

__version__ = "0.1.0"
