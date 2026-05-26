"""Cross-cutting utilities — config, logging, seeding, device, paths.

Populated incrementally :
- Sprint 11.B : :mod:`maps.utils.seeding`
- Sprint 12.B : :mod:`maps.utils.config`, :mod:`maps.utils.logging_setup`,
  :mod:`maps.utils.device`, :mod:`maps.utils.paths`
- Sprint 16 (planned, DETTE-4) : :mod:`maps.utils.energy_tracker`
  (may end up in :mod:`maps.domains.marl` instead)
"""

from maps.utils.config import CONFIG_ROOT, load_config, parse_overrides
from maps.utils.device import DevicePref, get_device, pick_best_available
from maps.utils.logging_setup import DEFAULT_FORMAT, LogLevel, configure_logging
from maps.utils.paths import Paths, get_paths
from maps.utils.seeding import LAB_DEFAULT_SEED, set_all_seeds

__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_FORMAT",
    "LAB_DEFAULT_SEED",
    "DevicePref",
    "LogLevel",
    "Paths",
    "configure_logging",
    "get_device",
    "get_paths",
    "load_config",
    "parse_overrides",
    "pick_best_available",
    "set_all_seeds",
]
