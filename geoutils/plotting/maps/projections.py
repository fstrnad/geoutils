"""Map projection lookup."""
# import matplotlib.cm as cm
import cartopy.crs as ccrs





def get_projection(projection, central_longitude=None, central_latitude=None,
                   dateline=False):
    central_longitude = 0 if central_longitude is None else central_longitude
    central_latitude = 0 if central_latitude is None else central_latitude
    if dateline:
        central_longitude = 180

    if not isinstance(central_longitude, float) and not isinstance(central_longitude, int):
        raise ValueError(
            f'central_longitude is not of type{type(central_longitude)}!'
        )
    if projection == "Mollweide":
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    elif projection == "EqualEarth":
        proj = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == "Robinson":
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == "PlateCarree":
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    elif projection == "Nearside":
        proj = ccrs.NearsidePerspective(
            central_longitude=central_longitude, central_latitude=central_latitude)
    elif projection == "LambertConformal":
        proj = ccrs.LambertConformal(central_longitude=central_longitude)
    else:
        raise ValueError(f"This projection {projection} is not available yet!")

    return proj
