"""Backward-compatible shim.

The Hydra bridge now lives in ``cfgable.hydra_utils`` (install the
``cfgable[hydra]`` extra). Re-exported here so existing
``mcap_data_loader.utils.hydra_utils`` imports keep working.
"""

from cfgable.hydra_utils import (  # noqa: F401
    relative_path_between,
    init_hydra_config,
    hydra_instance,
    hydra_instance_from_config_path,
    hydra_instance_from_dict,
)
