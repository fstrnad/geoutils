"""Annotate a map: mark the temperature extremes and a region of interest.

Regional maps are drawn in the PlateCarree projection, which is the one that
crops the view to the extent of the data.

Run:  python examples/05_annotated_map.py [path/to/era5.nc]
"""
import geoutils.plotting as gp
from common import FIGURES, load_t2m

t2m = load_t2m()

res = gp.plot_map(
    t2m, plot_type="contourf", levels=12, cmap="Spectral_r",
    label="2 m temperature [°C]", title="Extremes and a region of interest",
    plot_borders=True, land_ocean=True, color_bounds="minmax", figsize=(8, 6.5))
ax = res["ax"]

gp.plot_rectangle(ax, lon_range=(11, 14.5), lat_range=(50, 52),
                  text="Region of interest", color="k", lw=1.5)

# location of the warmest and coldest grid point
stacked = t2m.stack(point=("lat", "lon"))
for point, tag in [(stacked.idxmax("point"), "max"),
                   (stacked.idxmin("point"), "min")]:
    lat, lon = point.item()
    value = float(t2m.sel(lat=lat, lon=lon))
    ax.plot(lon, lat, "o", mfc="white", mec="k", markersize=8,
            transform=gp.get_projection("PlateCarree"))
    gp.plot_text(ax, f"{tag} {value:.1f} °C", xpos=lon, ypos=lat + 0.3,
                 color="k", fsize=12, weight="bold")

gp.save_fig(FIGURES / "05_annotated_map", fig=res["fig"])
