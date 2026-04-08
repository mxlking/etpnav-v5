from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import types
from typing import Any, Iterable

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf


def _convert_value(value: Any) -> Any:
    if isinstance(value, CompatCN):
        return value.clone()
    if isinstance(value, DictConfig):
        return CompatCN(OmegaConf.to_container(value, resolve=False))
    if isinstance(value, dict):
        return CompatCN(value)
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    return value


class CompatCN:
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_deprecated_keys", set())
        if initial:
            for key, value in initial.items():
                self._data[key] = _convert_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __delitem__(self, key: str) -> None:
        if self._frozen:
            raise AttributeError(f"Config is frozen, cannot delete {key}")
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(key) from None
        if key in data:
            return data[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return

        # torch.load may materialize objects before __init__ runs (pickle path).
        # Ensure internal fields always exist before normal attribute handling.
        try:
            data = object.__getattribute__(self, "_data")
            frozen = object.__getattribute__(self, "_frozen")
        except AttributeError:
            object.__setattr__(self, "_data", {})
            object.__setattr__(self, "_frozen", False)
            object.__setattr__(self, "_deprecated_keys", set())
            data = object.__getattribute__(self, "_data")
            frozen = False

        if frozen and key not in data:
            raise AttributeError(f"Config is frozen, cannot set {key}")
        data[key] = _convert_value(value)

    def __repr__(self) -> str:
        return f"CompatCN({self._data!r})"

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def clone(self) -> "CompatCN":
        return CompatCN(self.to_dict())

    def __deepcopy__(self, memo: dict[int, Any]) -> "CompatCN":
        existing = memo.get(id(self))
        if existing is not None:
            return existing

        copied = CompatCN()
        memo[id(self)] = copied
        object.__setattr__(
            copied,
            "_data",
            deepcopy(object.__getattribute__(self, "_data"), memo),
        )
        object.__setattr__(
            copied,
            "_frozen",
            object.__getattribute__(self, "_frozen"),
        )
        object.__setattr__(
            copied,
            "_deprecated_keys",
            deepcopy(object.__getattribute__(self, "_deprecated_keys"), memo),
        )
        return copied

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, CompatCN):
                result[key] = value.to_dict()
            else:
                result[key] = deepcopy(value)
        return result

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, CompatCN):
                value.freeze()

    def defrost(self) -> None:
        object.__setattr__(self, "_frozen", False)
        for value in self._data.values():
            if isinstance(value, CompatCN):
                value.defrost()

    def register_deprecated_key(self, key: str) -> None:
        self._deprecated_keys.add(key)

    def merge_from_other_cfg(self, other: Any) -> None:
        if isinstance(other, CompatCN):
            other_dict = other.to_dict()
        elif isinstance(other, DictConfig):
            other_dict = deepcopy(other)
        elif isinstance(other, dict):
            other_dict = deepcopy(other)
        else:
            raise TypeError(f"Unsupported config type: {type(other)!r}")
        self._merge_mapping(other_dict)

    def merge_from_file(self, path: str) -> None:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self._merge_mapping(data)

    def merge_from_list(self, opts: Iterable[Any]) -> None:
        opts = list(opts)
        if len(opts) % 2 != 0:
            raise ValueError("opts should contain alternating KEY VALUE pairs")
        for key, value in zip(opts[0::2], opts[1::2]):
            parsed = yaml.safe_load(value) if isinstance(value, str) else value
            self._set_by_path(str(key), parsed)

    def _merge_mapping(self, other: Any) -> None:
        if isinstance(other, DictConfig):
            other = deepcopy(other)
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self._data or not isinstance(self._data[key], CompatCN):
                    self._data[key] = CompatCN()
                self._data[key]._merge_mapping(value)
            else:
                self._data[key] = _convert_value(value)

    def _set_by_path(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node: CompatCN = self
        for part in parts[:-1]:
            if part not in node._data:
                node._data[part] = CompatCN()
            elif not isinstance(node._data[part], CompatCN):
                node._data[part] = _convert_value(node._data[part])
                if not isinstance(node._data[part], CompatCN):
                    node._data[part] = CompatCN()
            node = node._data[part]
        node._data[parts[-1]] = _convert_value(value)


_PATCHED = False


# 新加: 统一补齐 NumPy 1.24+ 移除的旧别名，避免旧代码在运行期零散触发兼容错误。
def _patch_numpy_legacy_aliases() -> None:
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "object": object,
        "long": int,
    }
    for alias_name, alias_value in legacy_aliases.items():
        if alias_name not in np.__dict__:
            setattr(np, alias_name, alias_value)


# 新加: 兼容新版 HabitatSimActions 默认使用小写动作名，补回旧代码依赖的大写别名。
def _patch_habitat_action_aliases() -> None:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    legacy_action_aliases = {
        "STOP": "stop",
        "MOVE_FORWARD": "move_forward",
        "TURN_LEFT": "turn_left",
        "TURN_RIGHT": "turn_right",
        "LOOK_UP": "look_up",
        "LOOK_DOWN": "look_down",
    }
    for legacy_name, new_name in legacy_action_aliases.items():
        if legacy_name not in HabitatSimActions._known_actions and new_name in HabitatSimActions._known_actions:
            HabitatSimActions._known_actions[legacy_name] = HabitatSimActions._known_actions[new_name]

# 新加: 兼容新版 habitat-baselines 中缺失的 common.environments 模块。
def _install_habitat_baselines_environments_alias() -> None:
    module_name = "habitat_baselines.common.environments"
    if module_name in sys.modules:
        return

    proxy_module = types.ModuleType(module_name)

    def get_env_class(name: str):
        from habitat_baselines.common.baseline_registry import baseline_registry

        env_cls = baseline_registry.get_env(name)
        if env_cls is not None:
            return env_cls

        from vlnce_baselines.common import environments as local_envs

        if hasattr(local_envs, name):
            return getattr(local_envs, name)

        raise ImportError(f"Environment class '{name}' is not registered")

    proxy_module.get_env_class = get_env_class
    sys.modules[module_name] = proxy_module


# 新加: 统一注入 Habitat/Habitat-Baselines 兼容补丁。
def patch_habitat_config_compat() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_numpy_legacy_aliases()

    import habitat
    import habitat.config.default as habitat_default

    habitat.Config = CompatCN
    habitat_default.Config = CompatCN
    _patch_habitat_action_aliases()
    _install_habitat_baselines_environments_alias()
    _PATCHED = True


# 新加: 对外保留单一入口，后续扩展兼容逻辑时只需改这里。
def patch_habitat_compat() -> None:
    patch_habitat_config_compat()
