"""Line plots from hourly data: a year of daily means, the diurnal cycle, and
the north-south profile in winter and summer.

Run:  python examples/04_profiles_and_distribution.py
"""
import geoutils.plotting as gp
from common import FIGURES, at_city, load

t2m = load("t2m")  # every month that is downloaded
cities = ["Oslo", "Berlin", "Madrid"]

res = gp.create_plot(1, 3, figsize=(18, 4.6))
ax_year, ax_day, ax_lat = res["ax"]

# daily means over the year; months that are not downloaded show as gaps
daily = [at_city(t2m, c).resample(time="1D").mean().compute() for c in cities]
gp.plot_2d(y=daily, x=daily[0].time, ax=ax_year, label=cities, lw_arr=[1.2],
           ylabel="2 m temperature [°C]", title="Daily mean, 2024")

# mean diurnal cycle in July (UTC)
july = load("t2m", 7)
cycles = [at_city(july, c).groupby("time.hour").mean().compute() for c in cities]
gp.plot_2d(y=cycles, x=cycles[0].hour, ax=ax_day, label=cities,
           xlabel="Hour of day [UTC]", ylabel="2 m temperature [°C]",
           title="Diurnal cycle, July")

# zonal mean profile, January against July
profiles = [load("t2m", m).mean(("time", "lon")).compute() for m in (1, 7)]
gp.plot_2d(y=profiles, x=profiles[0].lat, ax=ax_lat, label=["January", "July"],
           color_arr=["#0072B2", "#D55E00"], xlabel="Latitude [°N]",
           ylabel="2 m temperature [°C]", title="Zonal mean")

gp.save_fig(FIGURES / "04_profiles_and_distribution", fig=res["fig"])
