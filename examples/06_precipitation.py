"""Precipitation: monthly totals and the number of wet days.

Run:  python examples/06_precipitation.py
"""
import geoutils.plotting as gp
from common import FIGURES, load

res = gp.create_multi_plot(1, 3, projection="PlateCarree", figsize=(11.9, 3.1),
                           title="ERA5 total precipitation, 2024")
jan, jul, wet = res["ax"]

for ax, month, name in [(jan, 1, "January"), (jul, 7, "July")]:
    total = load("tp", month).sum("time").compute()
    gp.plot_map(total, ax=ax, plot_type="contourf", levels=10, cmap="YlGnBu",
                vmin=0, vmax=250, title=f"{name} total", label="Precipitation [mm]",
                unset_sci=True)

daily = load("tp", 7).resample(time="1D").sum()
wet_days = (daily >= 1).sum("time").compute()  # days with at least 1 mm
gp.plot_map(wet_days, ax=wet, plot_type="contourf", levels=10, cmap="BuPu",
            vmin=0, vmax=30, title="July wet days (≥ 1 mm)", label="Days")

gp.save_fig(FIGURES / "06_precipitation", fig=res["fig"])
