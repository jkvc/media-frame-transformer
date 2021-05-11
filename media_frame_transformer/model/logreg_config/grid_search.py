import itertools

from media_frame_transformer.model.logreg_config.base import load_logreg_model_config

_LOGREG_ARCH_PREFIX = "logreg"
_CONFIG_OVERRIDES = [
    ("+sn", "use_source_individual_norm"),
    ("+kb", "use_log_labelprop_bias"),
    ("+lr", "use_learned_residualization"),
    ("+gr", "use_gradient_reversal"),
]


def load_logreg_model_config_all_archs(n_classes, n_sources):
    base_config = load_logreg_model_config(_LOGREG_ARCH_PREFIX, n_classes, n_sources)

    arch2configs = {}
    combinations = itertools.product([False, True], repeat=len(_CONFIG_OVERRIDES))
    for comb in combinations:
        arch = _LOGREG_ARCH_PREFIX
        config_copy = {**base_config}
        for (prefix, key), value in zip(_CONFIG_OVERRIDES, comb):
            if value:
                arch += prefix
            config_copy[key] = value

        arch2configs[arch] = config_copy

    return arch2configs
