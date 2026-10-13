"""One field, three ways to draw it: mesh, filled contours and contour lines.

The default colour range is the robust 2nd-98th percentile of the data;
``color_bounds="minmax"`` shows the full range instead.

Run:  python examples/01_map_basics.py [path/to/era5.nc]
"""
import geoutils.plotting as gp
from common import FIGURES, load_t2m

t2m = load_t2m()
label = "2 m temperature [°C]"

res = gp.create_multi_plot(1, 3, projection="PlateCarree", figsize=(15, 4.5),
                           title="ERA5 2 m temperature, 2024-01-01 12 UTC")
mesh, filled, lines = res["ax"]

gp.plot_map(t2m, ax=mesh, title="Mesh", label=label, color_bounds="minmax")
gp.plot_map(t2m, ax=filled, plot_type="contourf", title="Filled contours",
            label=label, color_bounds="minmax", levels=8)

# contour lines drawn on top of a pale mesh; plot_array adds to an existing map
gp.plot_map(t2m, ax=lines, title="Contour lines", cmap="YlOrBr", label=label,
            color_bounds="minmax", alpha=0.3)
gp.plot_array(t2m, ax=lines, plot_type="contour", projection="PlateCarree",
              color="k", levels=5, color_bounds="minmax", lw=0.9, clabel=True,
              clabel_fmt="%.0f")

gp.save_fig(FIGURES / "01_map_basics", fig=res["fig"])
