"""Spread instead of a single line: hourly temperature distributions in July
and the daily min-max range as a shaded band.

Run:  python examples/08_distribution_and_range.py
"""
import geoutils.plotting as gp
from common import FIGURES, at_city, load

july = load("t2m", 7)
cities = ["Oslo", "Madrid"]

res = gp.create_plot(1, 2, figsize=(8.4, 3.2))
ax_hist, ax_band = res["ax"]

# distribution of all hourly values in July
hourly = [at_city(july, c).values for c in cities]
gp.plot_hist(hourly, ax=ax_hist, density=True, nbins=18, label=cities,
             xlabel="2 m temperature [°C]", ylabel="Density",
             title="Hourly values, July")

# Berlin: daily mean with the daily minimum-maximum range
berlin = at_city(july, "Berlin").resample(time="1D")
mean, lo, hi = (berlin.mean().compute(), berlin.min().compute(),
                berlin.max().compute())
gp.plot_2d(y=[mean], x=mean.time.dt.day, y_lb=[lo], y_ub=[hi], ax=ax_band,
           label=["Daily mean"], xlabel="Day of July", ylabel="2 m temperature [°C]",
           title="Berlin, daily range in July")

gp.save_fig(FIGURES / "08_distribution_and_range", fig=res["fig"])
