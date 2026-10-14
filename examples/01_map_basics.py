"""One field, three ways to draw it: mesh, filled contours and contour lines.

The default colour range is the robust 2nd-98th percentile of the data;
``color_bounds="minmax"`` shows the full range instead.

Run:  python examples/01_map_basics.py
"""
import geoutils.plotting as gp
from common import FIGURES, label, load

t2m = load("t2m", 1).sel(time="2024-01-15T12:00").compute()

res = gp.create_multi_plot(1, 3, projection="PlateCarree", figsize=(11.9, 3.1),
                           title="ERA5 2 m temperature, 2024-01-15 12 UTC")
mesh, filled, lines = res["ax"]

gp.plot_map(t2m, ax=mesh, title="Mesh", cmap="RdYlBu_r", label=label(t2m),
            color_bounds="minmax")
gp.plot_map(t2m, ax=filled, plot_type="contourf", title="Filled contours",
            cmap="RdYlBu_r", label=label(t2m), color_bounds="minmax", levels=10)

# contour lines on a pale mesh; plot_array adds to an existing map
gp.plot_map(t2m, ax=lines, title="Contour lines", cmap="Greys", label=label(t2m),
            color_bounds="minmax", alpha=0.25)
gp.plot_array(t2m, ax=lines, plot_type="contour", projection="PlateCarree",
              color="k", levels=14, color_bounds="minmax", lw=0.6)

gp.save_fig(FIGURES / "01_map_basics", fig=res["fig"])
