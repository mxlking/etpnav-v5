# Compatibility shim: provide get_env_class expected by older code.
# Returns env class by name. First try project vlnce_baselines, then habitat_baselines.
import importlib


def get_env_class(name: str):
    # try project-specific envs
    try:
        mod = importlib.import_module('vlnce_baselines.common.environments')
        if hasattr(mod, name):
            return getattr(mod, name)
    except Exception:
        pass

    # try habitat_baselines env factory modules for common patterns
    try:
        mod = importlib.import_module('habitat_baselines.common.env_factory')
        # no direct mapping, return None to indicate not found
    except Exception:
        pass

    raise ImportError(f"Environment class '{name}' not found in vlnce_baselines or habitat_baselines.")
