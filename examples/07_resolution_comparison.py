"""What a coarser grid loses: the Alps at 0.25° and 0.5°.

The 0.5° files hold 6-hourly data for 1-3 July 2024 only. Their grid points
coincide with every second 0.25° point, so the comparison is made by
interpolating the coarse field back onto the fine grid.

Run:  python examples/07_resolution_comparison.py
"""
import geoutils.plotting as gp
from common import FIGURES, label, load

when = "2024-07-01T12:00"
alps = {"lat_range": (43, 49), "lon_range": (4, 17)}
area = {"lat": slice(43, 49), "lon": slice(4, 17)}
fine = load("t2m", 7, "0.25deg").sel(time=when).compute()
coarse = load("t2m", 7, "0.5deg").sel(time=when).compute()
lost = coarse.interp(lat=fine.lat, lon=fine.lon) - fine

res = gp.create_multi_plot(1, 3, projection="PlateCarree", figsize=(17, 3.9),
                           title=f"2 m temperature, {when.replace('T', ' ')} UTC")
shown = fine.sel(**area)
for ax, data, name in [(res["ax"][0], fine, "0.25°"), (res["ax"][1], coarse, "0.5°")]:
    gp.plot_map(data, ax=ax, title=name, cmap="Spectral_r", vmin=float(shown.min()),
                vmax=float(shown.max()), label=label(data), **alps)
gp.plot_map(lost, ax=res["ax"][2], title="0.5° interpolated minus 0.25°",
            cmap="RdBu_r", vmin=-3, vmax=3, label="Difference [K]", **alps)

gp.save_fig(FIGURES / "07_resolution_comparison", fig=res["fig"])
