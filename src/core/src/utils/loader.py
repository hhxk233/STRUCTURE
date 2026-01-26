from copy import deepcopy
from pathlib import Path

import yaml


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        if hasattr(stream, "name"):
            self._root = Path(stream.name).resolve().parent
        else:
            self._root = Path.cwd()


def _construct_include(loader: Loader, node):
    filename = loader.construct_scalar(node)
    include_path = (loader._root / filename).resolve()
    with open(include_path, "r") as f:
        return yaml.load(f, Loader)


Loader.add_constructor("!include", _construct_include)


def merge_dicts(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
