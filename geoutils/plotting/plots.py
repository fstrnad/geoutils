"""Deprecated aggregator: import from ``geoutils.plotting`` instead."""
import warnings

from geoutils.plotting import *  # noqa: F401,F403
from geoutils.plotting.style import *  # noqa: F401,F403
from geoutils.plotting.axes import *  # noqa: F401,F403
from geoutils.plotting.colors import *  # noqa: F401,F403
from geoutils.plotting.figure import *  # noqa: F401,F403
from geoutils.plotting.xy import *  # noqa: F401,F403
from geoutils.plotting.maps.projections import *  # noqa: F401,F403
from geoutils.plotting.maps.basemap import *  # noqa: F401,F403
from geoutils.plotting.maps.fields import *  # noqa: F401,F403
from geoutils.plotting.maps.overlays import *  # noqa: F401,F403

warnings.warn(
    "geoutils.plotting.plots is deprecated; import from geoutils.plotting",
    DeprecationWarning,
    stacklevel=2,
)
