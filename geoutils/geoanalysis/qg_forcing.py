import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import geoutils.geodata.base_dataset as bds
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import numpy as np
import metpy as mpy
import metpy.constants as mpconstants
import metpy.calc as mpcalc


def define_qg_constants():
    # Set default static stability value
    sigma = 2.0e-6 * mpy.units.units('m^2 Pa^-2 s^-2')

    # Set f-plane at typical synoptic f0 value
    f0 = 1e-4 * mpy.units.units('s^-1')

    # Use dry gas constant from MetPy constants
    Rd = mpconstants.Rd

    return sigma, f0, Rd


def get_geostrophic_winds(z):
    geostrophic_u, geostrophic_v = mpcalc.geostrophic_wind(
        height=z)
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
        avor, u=geostrophic_u, v=geostrophic_v)

    return vortadv


def get_qg_circulation(vortadvlow, vortadvhigh, dp=100):
    # First term of QG forcing
    sigma, f0, Rd = define_qg_constants()
    diff_avor = (vortadvhigh-vortadvlow)/(dp*mpy.units.units.hPa)
    term1 = (-f0/sigma * diff_avor).metpy.convert_to_base_units()

    return term1


def get_tadv(Tlev, geostrophic_u, geostrophic_v):
    tadvlev = mpcalc.advection(Tlev,
                               u=geostrophic_u,
                               v=geostrophic_v)
    return tadvlev


def get_temperature_advection_term(tadv, p=100):
    # Second term of QG forcing
    sigma, f0, Rd = define_qg_constants()
    lap_tadv = mpcalc.laplacian(tadv, axes=['lat', 'lon'],
                                #    deltas=[dy, dx]
                                )
    term2 = -(lap_tadv * Rd/(sigma * (p * mpy.units.units.hPa))
              ).metpy.convert_to_base_units()

    return term2


def get_qg_forcing(zlow, zhigh, zlev, Tlev, dp, p):

    gut.myprint("==========Compute QG forcing===========")

    gut.myprint("Compute geostrophic winds")
    u_geo_low, v_geo_low = get_geostrophic_winds(z=zlow)
    u_geo_high, v_geo_high = get_geostrophic_winds(z=zhigh)
    u_geo_lev, v_geo_lev = get_geostrophic_winds(z=zlev)

    gut.myprint("Compute absolute vorticity")
    avorlow = get_absolute_vorticity_geostrophic(
        geostrophic_u=u_geo_low, geostrophic_v=v_geo_low)
    avorhigh = get_absolute_vorticity_geostrophic(
        geostrophic_u=u_geo_high, geostrophic_v=v_geo_high)

    gut.myprint("Compute vorticity advection")
    vortadvlow = get_vorticity_advection(
        avor=avorlow, geostrophic_u=u_geo_low, geostrophic_v=v_geo_low)
    vortadvhigh = get_vorticity_advection(
        avor=avorhigh, geostrophic_u=u_geo_high, geostrophic_v=v_geo_high)

    gut.myprint("Compute vorticity advection term")
    term1 = get_qg_circulation(vortadvlow=vortadvlow, vortadvhigh=vortadvhigh,
                               dp=dp)

    gut.myprint("Compute temperature advection term")
    tadv = get_tadv(Tlev=Tlev, geostrophic_u=u_geo_lev,
                    geostrophic_v=v_geo_lev)
    term2 = get_temperature_advection_term(tadv=tadv, p=p)

    gut.myprint("Finish QG forcing")
    qg_forcing = term1 + term2

    return {"QG forcing": qg_forcing,
            "term1": term1,
            "term2": term2,
            }
