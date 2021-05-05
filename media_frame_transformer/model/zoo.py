# a model zoo, for quick definition / retrieval of model by name

from typing import Any, Dict

_MODELS = {}


def get_model(config: Dict[str, Any]):
    arch = config["arch"]
    return _MODELS[arch](config)


def get_model_names():
    return sorted(list(_MODELS.keys()))


def register_model(model: object):
    arch = model.__name__
    _MODELS[arch] = model
    return model
