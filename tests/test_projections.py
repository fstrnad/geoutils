import cartopy.crs as ccrs
import pytest

from geoutils.plotting.maps.basemap import central_longitude_of
from geoutils.plotting.maps.projections import get_projection


@pytest.mark.parametrize("name, crs", [
    ("Mollweide", ccrs.Mollweide), ("EqualEarth", ccrs.EqualEarth),
    ("Robinson", ccrs.Robinson), ("PlateCarree", ccrs.PlateCarree),
    ("Nearside", ccrs.NearsidePerspective),
    ("LambertConformal", ccrs.LambertConformal),
])
def test_known_projections(name, crs):
    assert isinstance(get_projection(name), crs)


def test_dateline_sets_central_longitude_180():
    proj = get_projection("PlateCarree", dateline=True)
    assert central_longitude_of(proj) == 180


def test_nearside_uses_central_latitude():
    proj = get_projection("Nearside", central_latitude=40)
    assert proj.proj4_params["lat_0"] == 40


def test_unknown_projection_lists_valid_names():
    with pytest.raises(ValueError, match="Robinson") as err:
        get_projection("Mercatorish")
    assert "Mercatorish" in str(err.value)


def test_non_numeric_central_longitude_rejected():
    with pytest.raises(ValueError, match="central_longitude"):
        get_projection("PlateCarree", central_longitude="0")
