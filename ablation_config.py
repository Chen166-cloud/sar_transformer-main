"""Controlled single-factor ablation definitions."""

ABLATION_PRESETS = {
    "full": {"msf": True, "frequency": True, "gate": True, "bottleneck": True, "fusion": True},
    "wout_frequency": {"msf": True, "frequency": False, "gate": True, "bottleneck": True, "fusion": True},
    "wout_gate": {"msf": True, "frequency": True, "gate": False, "bottleneck": True, "fusion": True},
    "wout_fusion": {"msf": True, "frequency": True, "gate": True, "bottleneck": True, "fusion": False},
    "wout_msf": {"msf": False, "frequency": True, "gate": True, "bottleneck": True, "fusion": True},
    "wout_bottleneck": {"msf": True, "frequency": True, "gate": True, "bottleneck": False, "fusion": True},
}


def resolve_ablation_preset(name):
    if name not in ABLATION_PRESETS:
        raise ValueError(
            "Unknown ablation preset {!r}; expected one of {}".format(
                name, sorted(ABLATION_PRESETS)
            )
        )
    return dict(ABLATION_PRESETS[name])
