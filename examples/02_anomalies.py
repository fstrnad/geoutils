"""Weather anomalies: one day's mean against its month, on a diverging colormap.

Diverging colormaps get colour limits symmetric about zero automatically.

Run:  python examples/02_anomalies.py
"""
import geoutils.plotting as gp
from common import FIGURES, load

res = gp.create_multi_plot(1, 2, projection="PlateCarree", figsize=(9.1, 3.6),
                           title="Daily-mean 2 m temperature anomaly from the monthly mean")
for ax, (month, day) in zip(res["ax"], [(1, "2024-01-15"), (7, "2024-07-15")]):
    t2m = load("t2m", month)
    anomaly = (t2m.sel(time=day).mean("time") - t2m.mean("time")).compute()
    gp.plot_map(anomaly, ax=ax, cmap="RdBu_r", plot_type="contourf", levels=12,
                title=day, label="Anomaly [K]")

gp.save_fig(FIGURES / "02_anomalies", fig=res["fig"])
