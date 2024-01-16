import geoutils.utils.general_utils as gut
import metpy as mpy
import metpy.constants as mpconstants
import metpy.calc as mpcalc


def n_point_smoother(data, p=9, n_reps=10):
    # Apply the 9-point smoother to the data
    data_s = mpcalc.smooth_n_point(data, p, n_reps)
    return data_s


def define_qg_constants():
    # Set default static stability value
    sigma = 2.0e-6 * mpy.units.units('m^2 Pa^-2 s^-2')

    # Set f-plane at typical synoptic f0 value
    f0 = 1e-4 * mpy.units.units('s^-1')

    # Use dry gas constant from MetPy constants
    Rd = mpconstants.Rd

    return sigma, f0, Rd


def get_geostrophic_winds(z):
    """Compute geostrophic winds based on geopotential height at a given level

    Args:
        z (xarray.DataArray): dataarray containing geopotential height with metpy units

    Returns:
        xr.DataArray: tuple of geostrphic u and v winds.
    """
    gut.myprint("Compute geostrophic winds")
    geostrophic_u, geostrophic_v = mpcalc.geostrophic_wind(z)
    return geostrophic_u, geostrophic_v


def get_absolute_vorticity_geostrophic(geostrophic_u, geostrophic_v):
    avor = mpcalc.absolute_vorticity(
        u=geostrophic_u, v=geostrophic_v)
    return avor


def get_vorticity_advection(avor,
                            geostrophic_u,
                            geostrophic_v):
    # Vorticity advection
    vortadv = mpcalc.advection(
        avor, u=geostrophic_u, v=geostrophic_v).metpy.convert_to_base_units()

    return vortadv


def get_qg_circulation(vortadvlow, vortadvhigh, dp=100):
    # First term of QG forcing
    sigma, f0, Rd = define_qg_constants()
    diff_avor = (vortadvhigh-vortadvlow)/(dp*mpy.units.units.hPa)
    term1 = (diff_avor * (-f0/sigma)).metpy.convert_to_base_units()

    return term1


def get_tadv(Tlev, zlev=None, geostrophic_u=None, geostrophic_v=None):
    if zlev is None and geostrophic_u is None and geostrophic_v is None:
        raise ValueError("Need either zlev or geostrophic winds")
    if zlev is not None:
        geostrophic_u, geostrophic_v = get_geostrophic_winds(z=zlev)

    gut.myprint("Compute temperature advection")
    tadvlev = mpcalc.advection(Tlev,
                               u=geostrophic_u,
                               v=geostrophic_v)
    return tadvlev


def get_temperature_advection_term(tadv=None, zlev=None, Tlev=None,
                                   p=100):
    if zlev is None and tadv is None:
        raise ValueError("Need either zlev or tadv")
    if zlev is not None and tadv is None:
        tadv = get_tadv(Tlev=Tlev, zlev=zlev)
    elif tadv is None:
        raise ValueError("Need either zlev or tadv")
    # Second term of QG forcing
    sigma, f0, Rd = define_qg_constants()
    gut.myprint("Compute temperature advection laplacian")
    lap_tadv = mpcalc.laplacian(tadv,
                                # because time is given as additional dimension
                                axes=('lat', 'lon'),
                                )
    term2 = -(lap_tadv * Rd/(sigma * (p * mpy.units.units.hPa))
              ).metpy.convert_to_base_units()

    return term2


def get_qg_forcing(zlow, zhigh, zlev, Tlev, dp, p, smoother=False):
    """Compute QG forcing based on geopotential height and temperature for a given level

    Args:
        zlow (xr.DataArray): lover level of geopotential height for term1
        zhigh (xr.DataArray): upper level of geopotential height for term1
        zlev (xr.DataArray): level to compute QG forcing
        Tlev (xr.DataArray): temperature at level to compute QG forcing
        dp (float): pressure difference between zlow and zhigh
        p (float): pressure at level to compute QG forcing

    Returns:
        dict: dictionary containing QG forcing, term1, and term2
    """

    gut.myprint("==========Compute QG forcing===========")
    # Grab lat/lon values from file as unit arrays
    lats = zlev['lat'].metpy.unit_array
    lons = zlev['lon'].metpy.unit_array

    # Calculate distance between grid points
    # will need for computations later
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

    gut.myprint("Compute geostrophic winds")
    u_geo_low, v_geo_low = get_geostrophic_winds(z=zlow)
    u_geo_high, v_geo_high = get_geostrophic_winds(z=zhigh)
    u_geo_lev, v_geo_lev = get_geostrophic_winds(z=zlev)
    Tlev = Tlev.metpy.convert_units('degC')
    if smoother:
        gut.myprint("Apply 9-point smoother")
        u_geo_low_s = n_point_smoother(u_geo_low)
        v_geo_low_s = n_point_smoother(v_geo_low)
        u_geo_high_s = n_point_smoother(u_geo_high)
        v_geo_high_s = n_point_smoother(v_geo_high)
        u_geo_lev_s = n_point_smoother(u_geo_lev)
        v_geo_lev_s = n_point_smoother(v_geo_lev)
        Tlev_s = n_point_smoother(Tlev)
    else:
        u_geo_low_s = u_geo_low
        v_geo_low_s = v_geo_low
        u_geo_high_s = u_geo_high
        v_geo_high_s = v_geo_high
        u_geo_lev_s = u_geo_lev
        v_geo_lev_s = v_geo_lev
        Tlev_s = Tlev

    gut.myprint("Compute absolute vorticity")
    avorlow = get_absolute_vorticity_geostrophic(
        geostrophic_u=u_geo_low_s, geostrophic_v=v_geo_low_s)
    avorhigh = get_absolute_vorticity_geostrophic(
        geostrophic_u=u_geo_high_s, geostrophic_v=v_geo_high_s)

    gut.myprint("Compute vorticity advection")
    vortadvlow = get_vorticity_advection(
        avor=avorlow, geostrophic_u=u_geo_low_s, geostrophic_v=v_geo_low_s)
    vortadvhigh = get_vorticity_advection(
        avor=avorhigh, geostrophic_u=u_geo_high_s, geostrophic_v=v_geo_high_s)

    gut.myprint("Compute vorticity advection term")
    term1 = get_qg_circulation(vortadvlow=vortadvlow, vortadvhigh=vortadvhigh,
                               dp=dp)

    gut.myprint("Compute temperature advection term")
    tadv = get_tadv(Tlev=Tlev_s, geostrophic_u=u_geo_lev_s,
                    geostrophic_v=v_geo_lev_s)
    term2 = get_temperature_advection_term(tadv=tadv, p=p,
                                           )

    gut.myprint("Finish QG forcing")
    qg_forcing = term1 + term2

    return {"QG forcing": qg_forcing,
            "term1": term1,
            "term2": term2,
            }
