"""Map projection lookup."""
import numbers

import cartopy.crs as ccrs

_PROJECTIONS = {
    "Mollweide": lambda lon, lat: ccrs.Mollweide(central_longitude=lon),
    "EqualEarth": lambda lon, lat: ccrs.EqualEarth(central_longitude=lon),
    "Robinson": lambda lon, lat: ccrs.Robinson(central_longitude=lon),
    "PlateCarree": lambda lon, lat: ccrs.PlateCarree(central_longitude=lon),
    "Nearside": lambda lon, lat: ccrs.NearsidePerspective(
        central_longitude=lon, central_latitude=lat),
    "LambertConformal": lambda lon, lat: ccrs.LambertConformal(
        central_longitude=lon),
}


def get_projection(projection, central_longitude=None, central_latitude=None,
                   dateline=False):
    central_longitude = 0 if central_longitude is None else central_longitude
    central_latitude = 0 if central_latitude is None else central_latitude
    if dateline:
        central_longitude = 180

    if not isinstance(central_longitude, numbers.Real):
        raise ValueError(
            f'central_longitude is not of type{type(central_longitude)}!'
        )
    if projection not in _PROJECTIONS:
        raise ValueError(
            f"Projection {projection!r} is not available; "
            f"choose one of {sorted(_PROJECTIONS)}!")
    return _PROJECTIONS[projection](central_longitude, central_latitude)
