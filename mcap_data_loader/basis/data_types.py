"""Backward-compatible shim.

``StrEnum`` / ``ReprEnum`` now live in ``cfgable.enums``. Re-exported here so
existing ``mcap_data_loader.basis`` (and ``...basis.data_types``) imports keep working.
"""

from cfgable.enums import ReprEnum, StrEnum  # noqa: F401
