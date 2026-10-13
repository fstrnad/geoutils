"""Line plots from the same field: profiles with a spread band, and a histogram.

Run:  python examples/04_profiles_and_distribution.py [path/to/era5.nc]
"""
import geoutils.plotting as gp
from common import FIGURES, load_t2m

t2m = load_t2m()

res = gp.create_plot(1, 3, figsize=(16, 4.2))
ax_lat, ax_lon, ax_hist = res["ax"]

# north-south profile: mean over longitude with min-max band
gp.plot_2d(y=t2m.mean("lon"), x=t2m.lat, ax=ax_lat, label="zonal mean",
           y_lb=t2m.min("lon").values, y_ub=t2m.max("lon").values,
           xlabel="Latitude [°N]", ylabel="2 m temperature [°C]",
           title="Zonal mean and range")

# west-east profiles at three latitudes
lats = [48, 51, 54]
gp.plot_2d(y=[t2m.sel(lat=lat, method="nearest") for lat in lats],
           x=t2m.lon, ax=ax_lon, label=[f"{lat}°N" for lat in lats],
           xlabel="Longitude [°E]", ylabel="2 m temperature [°C]",
           title="West-east profiles")

# distribution of all grid points
gp.plot_hist(t2m.values.ravel(), ax=ax_hist, fig=res["fig"], nbins=20,
             xlabel="2 m temperature [°C]", ylabel="Density", density=True,
             title="Distribution")

gp.save_fig(FIGURES / "04_profiles_and_distribution", fig=res["fig"])
