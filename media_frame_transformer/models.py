from transformers import RobertaForSequenceClassification

_MODELS = {}


def get_model(arch: str):
    return _MODELS[arch]()


def register_model(arch: str):
    def _register(f):
        assert arch not in _MODELS
        _MODELS[arch] = f
        return f

    return _register


@register_model("roberta_base")
def roberta_base():
    return RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=15,
        output_attentions=False,
        output_hidden_states=False,
    )


@register_model("roberta_base_half")
def roberta_base_half():
    return _freeze_roberta_top_half(roberta_base())


@register_model("roberta_highdrop")
def roberta_highdrop():
    return RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=15,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=0.2,
    )


@register_model("roberta_highdrop_half")
def roberta_highdrop_half():
    return _freeze_roberta_top_n_layers(roberta_highdrop(), 6)


@register_model("roberta_highdrop_third")
def roberta_highdrop_third():
    return _freeze_roberta_top_n_layers(roberta_highdrop(), 8)


def _freeze_roberta_top_n_layers(model, n):
    # pretrained roberta = embeddings -> encoder.laysers -> classfier
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i, module in enumerate(model.roberta.encoder.layer):
        if i < n:
            for param in module.parameters():
                param.requires_grad = False
    return model
