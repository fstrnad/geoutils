"""Guards the step-1 packaging work: version exposure and importability."""
import importlib

import geoutils


def test_version_exposed():
    assert isinstance(geoutils.__version__, str) and geoutils.__version__


def test_core_subpackages_importable():
    for mod in (
        "geoutils.utils.spatial_utils",
        "geoutils.utils.time_utils",
        "geoutils.plotting.maps.fields",
        "geoutils.plotting.axes",
        "geoutils.plotting.colors",
        "geoutils.plotting.figure",
        "geoutils.plotting.style",
        "geoutils.plotting.xy",
        "geoutils.geodata.base_dataset",
    ):
        assert importlib.import_module(mod) is not None


def test_inter_font_is_shipped():
    import tomllib
    from pathlib import Path

    fonts = Path(geoutils.__file__).parent / "plotting" / "fonts"
    for name in ("Inter-Regular.ttf", "Inter-Italic.ttf", "Inter-SemiBold.ttf",
                 "Inter-Bold.ttf", "OFL.txt"):
        assert (fonts / name).is_file()
    root = Path(geoutils.__file__).parent.parent
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    assert cfg["tool"]["setuptools"]["package-data"]["geoutils.plotting"] == ["fonts/*"]
