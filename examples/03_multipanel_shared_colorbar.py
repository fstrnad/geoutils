"""A multi-panel figure sharing one colorbar, with a different map extent each.

All panels use the same colour limits, so the colours can be compared directly.

Run:  python examples/03_multipanel_shared_colorbar.py [path/to/era5.nc]
"""
import geoutils.plotting as gp
from common import FIGURES, load_t2m

t2m = load_t2m()
regions = {
    "North Sea coast": dict(lat_range=(52, 56), lon_range=(5, 11)),
    "Eastern Germany": dict(lat_range=(50, 54), lon_range=(10, 16)),
    "Central Germany": dict(lat_range=(49, 53), lon_range=(7, 13)),
    "Alpine foreland": dict(lat_range=(47, 51), lon_range=(8, 14)),
}

res = gp.create_multi_plot(2, 2, projection="PlateCarree", figsize=(10, 8),
                           title="Regional views, common colour scale")
for ax, (name, extent) in zip(res["ax"], regions.items()):
    im = gp.plot_map(t2m, ax=ax, plot_type="contourf", levels=8, title=name,
                     vmin=-7, vmax=9, cmap="viridis", **extent)
gp.add_colorbar(im, res["fig"], "2 m temperature [°C]", multi_plots=True)

gp.save_fig(FIGURES / "03_multipanel_shared_colorbar", fig=res["fig"])
