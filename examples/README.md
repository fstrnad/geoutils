# Plotting examples

Scripts using `geoutils.plotting` on the ERA5 test file
(`/Users/fstrnad/projects/era5/data/era5_test.nc`, 2 m temperature over Central
Europe). Pass another file as the first argument; it needs a `t2m` variable.
Run from anywhere, figures are written to `examples/figures/`.

| Script | Shows |
|---|---|
| `01_map_basics.py` | mesh, filled contours, contour lines; `color_bounds` |
| `02_anomalies.py` | diverging colormap with automatic symmetric limits |
| `03_multipanel_shared_colorbar.py` | `create_multi_plot`, regional extents, one shared colorbar |
| `04_profiles_and_distribution.py` | `plot_2d` with a spread band, several series, `plot_hist` |
| `05_annotated_map.py` | `plot_rectangle`, `plot_text`, markers on a map |

`common.py` loads the data and renames the coordinates to the `lat`/`lon`
names geoutils expects.
