# Plotting overhaul: design

Date: 2026-10-13. Scope: `geoutils/plotting/` only.

## Goal

Make the plotting package correct, leaner, easier to use and better looking,
without breaking the public names exported by `geoutils.plotting`.

Success criteria:
- All known bugs below are fixed, each with a regression test.
- Plotting code is smaller than today's ~3,800 lines, with no file over ~600 lines.
- Importing `geoutils.plotting` no longer modifies global matplotlib state.
- `from geoutils.plotting import ...` and `import geoutils.plotting.plots as cplt` keep working.
- Existing tests plus new ones pass after every step.

## Decisions (agreed with the user)

- Files may be moved, split or merged.
- The default look may change.
- Order: (1) bug fixes, (2) file moves with no behaviour change, (3) code slimming, (4) new look.
  Each step is its own commit and leaves the test suite green.

## Known bugs (step 1)

Verified on matplotlib 3.10 / cartopy 0.25.

| Where | Bug | Fix |
|---|---|---|
| `xy_plots.plot_2d` density | `sns` imported only in the `bar` branch; `alpha` undefined | import at top of branch; use `alpha_arr` |
| `xy_plots.plot_2d` scatter with `z_arr` | `ax.scatter(markersize=...)` invalid; `plt.cm.get_cmap` removed in mpl 3.9 | use `s=`; use `mpl.colormaps[...]` |
| `xy_plots.plot_2d` | `marker_size` overwritten by `mk_size`; mutable default args; `lcmap` colorbar hardcodes `jet` | merge into one option; `None` defaults; use `lcmap` |
| `map_plots.create_map` | `ax.outline_patch` removed from cartopy | `ax.spines['geo'].set_visible(False)` |
| `map_plots.plot_array` | `vmin`/`vmax` type check rejects numpy scalars | accept `numbers.Real` / `np.number` |
| `plots.save_fig` / `mk_plot_dir` | `os.makedirs('')` for a bare filename; extension check is a substring match on the whole path | use `pathlib`; check the suffix |
| `plots.create_plot` | single panel returns `ax=[]` | create the axes and return it |
| `plotting_utils.prepare_axis` | `yscale='symlog'` overwritten by later `pop`; `ysymlog` flow broken | resolve scale once |
| `plotting_utils.create_cmap` | `not isinstance(...) or not isinstance(...)` always raises for Colormap objects | accept any `mpl.colors.Colormap` |

## Target structure (step 2)

```
geoutils/plotting/
  __init__.py      public API (same names as today)
  style.py         from plot_settings.py: sizes, palette, set_style(), change_fsize/reset_fsize
  result.py        from plot_result.py
  figure.py        from plots.py + create_multi_plot + enumerate_subplots
  axes.py          prepare_axis, set_title, set_legend, plot_line/vline/hline/arrow/text, text_box
  colors.py        get_cmap, create_cmap, norms/levels, make_colorbar(_discrete), add_colorbar
  xy.py            from xy_plots.py
  maps/
    projections.py   get_projection (dict lookup)
    basemap.py       create_map, set_extent, set_grid, scale_bar
    fields.py        plot_map, plot_array, _draw_* drawers, significance hatching
    overlays.py      plot_edges, plot_wind_field, plot_trajectory, plot_rectangle, plot_ring, plot_horizontal_line_at_latitude
  dendrogram.py
  videos.py        unchanged (optional `video` extra, not imported by __init__)
  plots.py         deprecation shim: re-exports the public API, warns once
```

Moves use `git mv` so history is kept. No logic changes in this step except import paths.
Internal imports of `geoutils.plotting.*` elsewhere in the repo (e.g. `videos.py`, tutorials, tests)
are updated in the same commit.

## Slimming (step 3)

- `plot_array`, `plot_map`, `prepare_axis`: replace the unconsumed `**kwargs` soup with explicit
  keyword arguments. Recurring groups become small frozen dataclasses (`ColorbarSpec`, `AxisSpec`).
- Old keyword names stay accepted for one release and emit `DeprecationWarning`, so notebooks keep working.
- Collapse repeated per-series fallback logic in `plot_2d` into one helper.
- Collapse the vmin/vmax/levels/round/sci/extend logic, which is currently split across
  `plot_array`, `set_cb_boundaries` and `make_colorbar`, into a single place in `colors.py`.
- Split the significance-hatching branch out of `plot_map` into its own function.
- `get_projection` becomes a dict; `dendrogram.py` module functions that take `self` are cleaned up.
- Drop the dead code my changes expose. Pre-existing dead code is listed for the user, not deleted unasked.

## New look (step 4)

- `set_style()` applies all rc settings; nothing runs at import. `__init__` does not call it,
  but plotting functions use a `plt.rc_context` with the geoutils style, so user state is untouched.
- Base style: white background, light grid, outward ticks, no top/right spines, `savefig` dpi 300
  and `bbox='tight'` set in rc rather than in `save_fig`.
- Palette: colourblind-safe categorical cycle (Okabe-Ito) set through `axes.prop_cycle`;
  the `y`/`c`/`m` entries go away. More series than colours cycles instead of repeating colour 0.
- Colour bounds: default `color_bounds` changes from the 10th-90th percentile to a robust
  2nd-98th percentile with extend arrows. Diverging colormaps get symmetric bounds. `'quantile'` remains available.
- Colorbars: use `constrained_layout` / `fig.colorbar(shrink, pad)` instead of `inset_axes` with
  magic `borderpad` values; titles use layout instead of `y_title` offsets.

## Testing

- Step 1: one failing test per bug first, then the fix.
- Step 2: the full suite before and after; plus a check that every name in `__all__` and the old
  `plots` module resolves.
- Steps 3-4: image-free assertions on returned `PlotResult` content, axis properties and colorbar
  limits (matplotlib `Agg` backend). No pixel-comparison baselines, since the look is meant to change.

## Out of scope

Non-plotting modules, the optional dependency set, and any new plot types.

## Open assumptions to confirm

- Okabe-Ito and the 2-98% default are my choices; say if you prefer others.
- A one-release deprecation window for old kwargs is acceptable.
