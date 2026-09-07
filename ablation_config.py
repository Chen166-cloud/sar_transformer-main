"""Model variants used by the historical and ICSPS 2026 protocols.

``ABLATION_PRESETS`` is intentionally kept unchanged: the legacy runners import
it directly and reproduce the experiments that pre-date the ICSPS protocol.
The paper-facing variants live in ``FORMAL_VARIANTS`` and use names that encode
the representation and both Fourier-refinement locations explicitly.
"""

ABLATION_PRESETS = {
    "full": {"msf": True, "frequency": True, "gate": True, "bottleneck": True, "fusion": True},
    "wout_frequency": {"msf": True, "frequency": False, "gate": True, "bottleneck": True, "fusion": True},
    "wout_gate": {"msf": True, "frequency": True, "gate": False, "bottleneck": True, "fusion": True},
    "wout_fusion": {"msf": True, "frequency": True, "gate": True, "bottleneck": True, "fusion": False},
    "wout_msf": {"msf": False, "frequency": True, "gate": True, "bottleneck": True, "fusion": True},
    "wout_bottleneck": {"msf": True, "frequency": True, "gate": True, "bottleneck": False, "fusion": True},
}


FORMAL_VARIANTS = {
    "full": {
        "representation": "log",
        "compensation": True,
        "msf": True,
        "decoder_fdr": True,
        "gate": True,
        "bottleneck_local": True,
        "bottleneck_fdr": True,
    },
    "intensity_only": {
        "representation": "intensity",
        "compensation": False,
        "msf": True,
        "decoder_fdr": True,
        "gate": True,
        "bottleneck_local": True,
        "bottleneck_fdr": True,
    },
    "log_only": {
        "representation": "log",
        "compensation": False,
        "msf": True,
        "decoder_fdr": True,
        "gate": True,
        "bottleneck_local": True,
        "bottleneck_fdr": True,
    },
    "wout_all_fdr": {
        "representation": "log",
        "compensation": True,
        "msf": True,
        "decoder_fdr": False,
        "gate": True,
        "bottleneck_local": True,
        "bottleneck_fdr": False,
    },
}


def resolve_ablation_preset(name):
    if name not in ABLATION_PRESETS:
        raise ValueError(
            "Unknown ablation preset {!r}; expected one of {}".format(
                name, sorted(ABLATION_PRESETS)
            )
        )
    return dict(ABLATION_PRESETS[name])


def resolve_formal_variant(name):
    """Return a copy of an ICSPS paper-facing variant definition."""
    if name not in FORMAL_VARIANTS:
        raise ValueError(
            "Unknown formal variant {!r}; expected one of {}".format(
                name, sorted(FORMAL_VARIANTS)
            )
        )
    return dict(FORMAL_VARIANTS[name])


def legacy_preset_to_variant(name):
    """Translate a historical preset without changing its original behavior."""
    preset = resolve_ablation_preset(name)
    return {
        "representation": "log",
        "compensation": preset["fusion"],
        "msf": preset["msf"],
        "decoder_fdr": preset["frequency"],
        "gate": preset["gate"],
        # The historical ``wout_bottleneck`` removed the whole block, while
        # ``wout_frequency`` left the bottleneck FDR enabled.  Preserve both
        # details here; the formal w/o-all-FDR variant is defined separately.
        "bottleneck_local": preset["bottleneck"],
        "bottleneck_fdr": preset["bottleneck"],
    }
