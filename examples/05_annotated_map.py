"""Annotate a map: mark cities with their July mean and outline a region.

Run:  python examples/05_annotated_map.py
"""
import geoutils.plotting as gp
from common import CITIES, FIGURES, at_city, label, load

t2m = load("t2m", 7).mean("time").compute()

res = gp.plot_map(
    t2m, plot_type="contourf", levels=12, cmap="Spectral_r", label=label(t2m),
    title="July 2024 mean 2 m temperature", plot_borders=True,
    color_bounds="minmax", figsize=(10, 7.5))
ax = res["ax"]

gp.plot_rectangle(ax, lon_range=(5, 16), lat_range=(44, 48),
                  text="Alps", color="k", lw=1.2)
for city, (lat, lon) in CITIES.items():
    value = float(at_city(t2m, city))
    ax.plot(lon, lat, "o", mfc="white", mec="k", markersize=6,
            transform=gp.get_projection("PlateCarree"))
    gp.plot_text(ax, f"{city} {value:.0f} °C", xpos=lon, ypos=lat + 0.9,
                 weight="bold")

gp.save_fig(FIGURES / "05_annotated_map", fig=res["fig"])
