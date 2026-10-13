"""Guards the step-1 packaging work: version exposure and importability."""
import importlib

import geoutils


def test_version_exposed():
    assert isinstance(geoutils.__version__, str) and geoutils.__version__


def test_core_subpackages_importable():
    for mod in (
        "geoutils.utils.spatial_utils",
        "geoutils.utils.time_utils",
        "geoutils.plotting.map_plots",
        "geoutils.plotting.axes",
        "geoutils.plotting.colors",
        "geoutils.plotting.figure",
        "geoutils.plotting.style",
        "geoutils.plotting.xy",
        "geoutils.geodata.base_dataset",
    ):
        assert importlib.import_module(mod) is not None
