# Plotting examples

Scripts using `geoutils.plotting` on the ERA5 single-level data in
`/Users/fstrnad/projects/era5/data/era5-single-levels` (hourly 2 m temperature
and total precipitation for Europe, 2024, 0.25°; a few 6-hourly 0.5° fields).
Set `ERA5_DATA` to use another copy. Run from anywhere, e.g.
`python examples/03_multipanel_shared_colorbar.py`; figures go to
`examples/figures/`.

| Script | Shows |
|---|---|
| `01_map_basics.py` | mesh, filled contours, contour lines; `color_bounds` |
| `02_anomalies.py` | daily anomalies on a diverging colormap, symmetric limits |
| `03_multipanel_shared_colorbar.py` | four seasons, `create_multi_plot`, one shared colorbar |
| `04_time_series_and_profiles.py` | `plot_2d`: year of daily means, diurnal cycle, zonal profile |
| `05_annotated_map.py` | city markers and labels, `plot_rectangle` |
| `06_precipitation.py` | monthly totals and wet-day counts, fixed colour limits |
| `07_resolution_comparison.py` | 0.25° against 0.5° over the Alps, and what interpolation loses |
| `08_distribution_and_range.py` | `plot_hist` of hourly values, `plot_2d` with a shaded min-max band |

`common.py` finds and loads the files lazily (renaming the coordinates to the
`time`/`lat`/`lon` that geoutils expects, converting to °C and mm). When no month is
given, only months whose `.sig.json` marker exists are loaded, so a download
that is still running is safe; incomplete months show up as gaps. Scripts that
ask for a specific month need that file to be complete.
