"""Anomalies on a diverging colormap, with the default symmetric colour range.

Run:  python examples/02_anomalies.py [path/to/era5.nc]
"""
import geoutils.plotting as gp
from common import FIGURES, load_t2m

t2m = load_t2m()
anom = t2m - t2m.mean()          # departure from the area mean
zonal_anom = t2m - t2m.mean("lon")  # departure from each latitude's mean

res = gp.create_multi_plot(1, 2, projection="PlateCarree", figsize=(11, 4.5),
                           title=f"2 m temperature anomalies (area mean {float(t2m.mean()):.1f} °C)")
# diverging colormaps get colour limits symmetric about zero automatically
gp.plot_map(anom, ax=res["ax"][0], cmap="RdBu_r", plot_type="contourf",
            levels=10, label="Anomaly from area mean [K]", title="Area mean removed")
gp.plot_map(zonal_anom, ax=res["ax"][1], cmap="RdBu_r", plot_type="contourf",
            levels=10, label="Anomaly from zonal mean [K]",
            title="Zonal mean removed")

gp.save_fig(FIGURES / "02_anomalies", fig=res["fig"])
