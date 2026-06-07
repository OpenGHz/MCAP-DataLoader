"""Backward-compatible shim.

The configuration framework now lives in the standalone ``cfgable`` package.
This module re-exports it so existing ``mcap_data_loader.basis.cfgable`` imports keep
working unchanged. New code should import from ``cfgable`` directly.
"""

from cfgable import (  # noqa: F401
    NoConfig,
    OtherCfgType,
    ConfigType,
    AllConfigType,
    DataClassProto,
    InitConfigMeta,
    InitConfigABCMeta,
    InitConfigMixinBasis,
    InitConfigMixin,
    InitConfigABCMixin,
    ConfigurableBasis,
    dump_omegaconf,
    dump_or_repr,
    fetch_config,
    import_string,
    get_fully_qualified_class_name,
)
