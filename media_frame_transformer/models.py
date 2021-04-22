# a model zoo, for quick definition / retrieval of model by name

_MODELS = {}


def get_model(arch: str):
    return _MODELS[arch]()


def get_model_names():
    return sorted(list(_MODELS.keys()))


def register_model(arch: str):
    def _register(f):
        assert arch not in _MODELS
        _MODELS[arch] = f
        return f

    return _register
