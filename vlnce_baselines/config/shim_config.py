import copy
import yaml

class ConfigNode:
    def __init__(self, init_dict=None):
        object.__setattr__(self, "_data", {})
        if init_dict:
            for k, v in init_dict.items():
                if isinstance(v, dict):
                    self._data[k] = ConfigNode(v)
                else:
                    self._data[k] = v

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        # create nested node on access
        node = ConfigNode()
        data[name] = node
        return node

    def __setattr__(self, name, value):
        if name == "_data":
            object.__setattr__(self, name, value)
            return
        data = object.__getattribute__(self, "_data")
        if isinstance(value, dict):
            data[name] = ConfigNode(value)
        else:
            data[name] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def to_dict(self):
        def _rec(node):
            if not isinstance(node, ConfigNode):
                return node
            out = {}
            for k, v in node._data.items():
                out[k] = _rec(v)
            return out
        return _rec(self)

    def clone(self):
        return copy.deepcopy(self)

    def merge_from_file(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self._merge_dict(self, data)

    def merge_from_list(self, opts):
        # opts expected as list: ["A.B", value, ...]
        if not opts:
            return
        i = 0
        while i < len(opts):
            key = opts[i]
            val = opts[i+1]
            parts = key.split('.')
            node = self
            for p in parts[:-1]:
                node = getattr(node, p)
            setattr(node, parts[-1], val)
            i += 2

    def _merge_dict(self, node, data):
        for k, v in data.items():
            if isinstance(v, dict):
                if not isinstance(getattr(node, k), ConfigNode):
                    setattr(node, k, ConfigNode())
                self._merge_dict(getattr(node, k), v)
            else:
                setattr(node, k, v)

    def freeze(self):
        pass

    def defrost(self):
        pass

Config = ConfigNode
