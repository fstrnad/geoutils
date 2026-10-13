"""Deprecated aggregator: import from ``geoutils.plotting`` instead."""
import warnings

from geoutils.plotting import *  # noqa: F401,F403
from geoutils.plotting.style import *  # noqa: F401,F403
from geoutils.plotting.axes import *  # noqa: F401,F403
from geoutils.plotting.colors import *  # noqa: F401,F403
from geoutils.plotting.figure import *  # noqa: F401,F403
from geoutils.plotting.xy import *  # noqa: F401,F403
from geoutils.plotting.map_plots import *  # noqa: F401,F403

warnings.warn(
    "geoutils.plotting.plots is deprecated; import from geoutils.plotting",
    DeprecationWarning,
    stacklevel=2,
)
