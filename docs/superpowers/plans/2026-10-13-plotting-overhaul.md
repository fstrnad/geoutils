# Plotting Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix known plotting bugs, restructure `geoutils/plotting/` into focused modules, slim the code, and modernise the default look.

**Architecture:** Four sequential phases, each ending green on the full suite: bug fixes in place, pure file moves, kwargs-to-explicit-arguments slimming, new look. The public names exported by `geoutils.plotting` never change; `geoutils.plotting.plots` stays importable as a deprecation shim.

**Tech Stack:** Python >=3.12, matplotlib 3.10, cartopy 0.25, xarray, seaborn, pytest (Agg backend via `tests/conftest.py`), ruff.

**Spec:** `docs/superpowers/specs/2026-10-13-plotting-overhaul-design.md`

## Global Constraints

- Work on branch `plotting-overhaul`, never on `main`. One commit per task; commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Never use shell `rm`; file moves use `git mv`. If a file must be deleted, ask the user.
- No new dependencies. Lint/format with `ruff` on changed files only.
- Every name in `geoutils.plotting.__all__` and `import geoutils.plotting.plots as cplt` keep working at every commit.
- Old keyword arguments removed in Phase 3 keep working for one release and raise `DeprecationWarning`.
- After each task: `python -m pytest -q` is green (25 tests pass today).
- Do not touch non-plotting modules except to update their `geoutils.plotting.*` imports.
- Pre-existing dead code: list it for the user, do not delete it.
- Tests use the `latlon_da` / `spacetime_da` fixtures in `tests/conftest.py`; new regression tests go in `tests/test_plotting_regressions.py`.

## Review Focus

1. `plot_map` on an all-NaN array: expect a clear `ValueError`, not a numpy warning storm or silent empty plot (Task 2).
2. `plot_map` on a constant array (`vmin == vmax`): expect a drawn plot with a widened colour range (Task 2).
3. `plot_2d` with more series than palette colours: expect cycling, distinct neighbours, no `IndexError` (Task 1).
4. Plotting into a user-supplied `ax`: expect the user's figure untouched except that axis, and global `rcParams` unchanged after the call (Tasks 9, 11).
5. Diverging data (positive and negative) with a diverging cmap: expect symmetric colour limits; all-positive data with a sequential cmap must not be forced symmetric (Task 12).

---

## Phase 1: Bug fixes (files stay where they are)

### Task 1: `plot_2d` bugs

**Files:**
- Modify: `geoutils/plotting/xy_plots.py` (`plot_2d`)
- Test: `tests/test_plotting_regressions.py` (create)

**Interfaces:**
- Produces: `plot_2d(y, x=None, z_arr=None, lw_arr=None, mk_arr=None, ls_arr=None, alpha_arr=None, color_arr=None, color=None, lcmap=None, ax=None, all_x=False, norm=False, standardize=False, plot_type='xy', **kwargs) -> PlotResult` (list defaults become `None`, filled with the old values inside).

- [ ] **Step 1: Create branch `plotting-overhaul`, then write failing tests** in `tests/test_plotting_regressions.py`: `test_plot_2d_density_runs`, `test_plot_2d_scatter_with_z_arr_runs` (`z_arr=[z]`, `plot_type="scatter"`), `test_plot_2d_default_lists_not_shared` (call with `lw_arr` omitted twice; assert `plot_2d.__defaults__` contains no `list`), `test_plot_2d_more_series_than_colors` (`len(pst.colors) + 3` series; assert `len(ax.lines) == n` and no exception), `test_plot_2d_lcmap_colorbar_uses_given_cmap` (assert colorbar mappable's `cmap.name == "plasma"` for `lcmap="plasma"`), `test_plot_2d_marker_size_kwarg_used` (`marker_size=9` gives `ax.lines[0].get_markersize() == 9`).
- [ ] **Step 2: Run** `python -m pytest tests/test_plotting_regressions.py -q`; expect all six to FAIL.
- [ ] **Step 3: Fix in `plot_2d`:** import seaborn at the top of the function; the density branch uses `alpha_arr` not the undefined `alpha`; scatter uses `s=mk_size` and `mpl.colormaps[cmap]`; read `marker_size` and `mk_size` into one variable (either name works); `lcmap` builds the colorbar `ScalarMappable` from itself; palette index `idx % len(pst.colors)`; `None` list defaults.
- [ ] **Step 4: Run** the new tests plus `python -m pytest -q`; expect PASS.
- [ ] **Step 5: Commit** `fix(plotting): plot_2d density/scatter/lcmap/marker_size/defaults`.

### Task 2: `create_map` frame and `plot_array` input handling

**Files:**
- Modify: `geoutils/plotting/map_plots.py` (`create_map`, `plot_array`)
- Test: `tests/test_plotting_regressions.py`

- [ ] **Step 1: Add failing tests:** `test_create_map_rem_frame` (`create_map(rem_frame=True)["ax"].spines["geo"].get_visible() is False`), `test_plot_map_accepts_numpy_scalar_limits` (`vmin=np.float32(0), vmax=np.float64(5)`), `test_plot_map_all_nan_raises_valueerror` (match `"all-NaN"`), `test_plot_map_constant_array_draws` (`latlon_da * 0 + 3`; result `im` not None, `im.norm.vmin < im.norm.vmax` when a mesh).
- [ ] **Step 2: Run;** the first three FAIL, the fourth documents current behavior.
- [ ] **Step 3: Fix:** replace `ax.outline_patch.set_visible(False)` with `ax.spines["geo"].set_visible(False)`; vmin/vmax type check uses `numbers.Real` (covers numpy scalars); at the top of `plot_array`, if `z` is not None and every value is NaN, raise `ValueError("z is all-NaN, nothing to plot")`.
- [ ] **Step 4: Run** full suite; PASS.
- [ ] **Step 5: Commit** `fix(plotting): rem_frame, numpy-scalar limits, all-NaN input`.

### Task 3: `save_fig`, `mk_plot_dir`, `create_plot`

**Files:**
- Modify: `geoutils/plotting/plots.py`
- Test: `tests/test_plotting_regressions.py`

**Interfaces:**
- Produces: `mk_plot_dir(savepath: str | Path) -> None`, `save_fig(savepath, fig=None, extension='png', dpi='figure') -> Path`, `create_plot(nrows=1, ncols=1, **kwargs) -> PlotResult` whose `ax` is a single `Axes` for 1x1 and an array otherwise.

- [ ] **Step 1: Failing tests** (use `tmp_path` and `monkeypatch.chdir(tmp_path)`): `test_save_fig_bare_filename`, `test_save_fig_adds_extension_only_when_missing` (path inside a directory named `png_plots/` still gets `.png`; `out.pdf` stays `out.pdf`), `test_create_plot_single_returns_axes` (`res["ax"]` is a `matplotlib.axes.Axes`), `test_create_plot_multi_returns_flat_array` (shape `(4,)` for 2x2).
- [ ] **Step 2: Run;** expect FAIL for the first three.
- [ ] **Step 3: Fix** with `pathlib`: `mk_plot_dir` creates `Path(savepath).parent` with `parents=True, exist_ok=True`; extension check is `Path(savepath).suffix.lower()` in the known set; the 1x1 `create_plot` uses `plt.subplots`; `dpi=300` default for `png` regardless of whether `fig` is passed.
- [ ] **Step 4: Run** full suite; PASS.
- [ ] **Step 5: Commit** `fix(plotting): save_fig paths, create_plot single axes`.

### Task 4: `prepare_axis` scale and `create_cmap` Colormap objects

**Files:**
- Modify: `geoutils/plotting/plotting_utils.py` (`prepare_axis`, `create_cmap`)
- Test: `tests/test_plotting_regressions.py`

- [ ] **Step 1: Failing tests:** `test_yscale_symlog_applied` (`plot_2d(y=..., yscale="symlog").ax.get_yscale() == "symlog"`), `test_ysymlog_flag_applied`, `test_ylog_and_ysymlog_conflict_raises`, `test_create_cmap_accepts_colormap_object` (`create_cmap(mpl.colormaps["viridis"], levels=[0, 1, 2])` returns `(cmap, BoundaryNorm)`), `test_create_cmap_rejects_other_types` (`ValueError`).
- [ ] **Step 2: Run;** expect FAIL for symlog and Colormap tests.
- [ ] **Step 3: Fix:** resolve `ylog`/`ysymlog` once, after popping both, then map `yscale`; `create_cmap` raises only when `cmap` is neither `str` nor `mpl.colors.Colormap`.
- [ ] **Step 4: Run** full suite; PASS; run `ruff check` on the four edited files.
- [ ] **Step 5: Commit** `fix(plotting): symlog scale, Colormap objects in create_cmap`.

---

## Phase 2: File moves (no behavior change)

### Task 5: Non-map modules into the new layout

**Files (all via `git mv`):**
- `plot_settings.py` -> `style.py`; `plot_result.py` -> `result.py`; `xy_plots.py` -> `xy.py`
- Split `plotting_utils.py` into `axes.py` (`prepare_axis`, `has_labels`, `set_title`, `set_legend`, `plot_line`, `plot_vline`, `plot_hline`, `plot_arrow`, `plot_text`, `text_box`, `get_lon_lat_ticklabels`) and `colors.py` (everything cmap/norm/colorbar: `get_cmap`, `create_cmap`, `truncate_colormap`, `get_available_*`, `make_colorbar*`, `add_colorbar`, `set_cb_boundaries`, `discrete_norm_ticks`, `get_arr_colorbar`, `cbarfmt`, `place_colorbar_offset_text`). Remaining helpers (`check_geoaxis`, `check_plot_type`, `check_projection`, `get_grid_figure`, `remove_colorbars`, `full_reset_ax`, `enumerate_subplots`) go to the module that uses them; a helper used by several goes to `axes.py`.
- `plots.py` content (`create_plot`, `save_fig`, `mk_plot_dir`, `create_twinaxis`) -> `figure.py`, together with `enumerate_subplots`.
- Modify: every in-repo import of the old module names (grep `geoutils.plotting.` in `geoutils/`, `tests/`, `tutorials/*.py`), and `tests/test_packaging.py`'s module list.
- Test: `tests/test_plotting_api.py` (create)

**Interfaces:**
- Produces: the module layout in the spec; `plots.py` becomes a shim.

- [ ] **Step 1: Write `tests/test_plotting_api.py`:** `test_all_public_names_resolve` (every name in `geoutils.plotting.__all__` is `getattr`-able), `test_plots_shim_exposes_public_api` (`import geoutils.plotting.plots as cplt`; same names present on it), `test_plots_shim_warns_once` (`pytest.warns(DeprecationWarning)` on first fresh import via `importlib.reload`). Run: the first two PASS, the third FAILS.
- [ ] **Step 2:** `git mv` each file; move functions between `axes.py`/`colors.py`/`figure.py` with cut-and-paste only (no edits to bodies). Update imports.
- [ ] **Step 3:** Rewrite `plots.py` as a shim: `from geoutils.plotting import *` plus `warnings.warn("geoutils.plotting.plots is deprecated; import from geoutils.plotting", DeprecationWarning, stacklevel=2)`. Keep `__all__` identical to the package's. Internal repo code (`videos.py`, `tsa/*.py`, tutorials, `cutouts/test_cutout.py`) switches to `import geoutils.plotting as cplt`.
- [ ] **Step 4: Run** `python -m pytest -q`; all green, and grep shows no remaining import of `plotting_utils`, `plot_settings`, `plot_result`, `xy_plots`.
- [ ] **Step 5: Commit** `refactor(plotting): split utils, rename modules, plots shim`.

### Task 6: `maps/` package

**Files:**
- Create: `geoutils/plotting/maps/__init__.py` (empty docstring only)
- `git mv` `map_plots.py` -> `maps/fields.py`, then move out by cut-and-paste: `maps/projections.py` (`get_projection`), `maps/basemap.py` (`create_map`, `set_extent`, `set_grid`, `scale_bar`, `utm_from_lon`, `estimate_distance`, `get_grid_steps`, `get_grid_dist`), `maps/overlays.py` (`plot_edges`, `plot_wind_field`, `plot_trajectory`, `plot_rectangle`, `plot_ring`, `plot_horizontal_line_at_latitude`, `plt_text_map`), `figure.py` gets `create_multi_plot`. `maps/fields.py` keeps `plot_map`, `plot_array`, `_DrawContext`, `_draw_*`, `_FIELD_DRAWERS`, `_resolve_drawer`.
- Modify: `geoutils/plotting/__init__.py` import sources; `tests/test_plotting.py` (`import geoutils.plotting.maps.fields as mp`); `tests/test_packaging.py` module list.

- [ ] **Step 1:** Add `test_maps_modules_importable` to `tests/test_plotting_api.py` listing the five new modules; run, expect FAIL.
- [ ] **Step 2:** Do the moves and import updates. Resolve cycles: `fields` may import `basemap` and `projections`; `basemap` must not import `fields`.
- [ ] **Step 3: Run** full suite; PASS. Verify no file in `geoutils/plotting/` exceeds ~600 lines (`wc -l`); if `fields.py` or `colors.py` does, note it for Phase 3 rather than splitting further here.
- [ ] **Step 4: Commit** `refactor(plotting): split map_plots into maps/ package`.

---

## Phase 3: Slimming

### Task 7: One colour-scale resolver

**Files:**
- Modify: `geoutils/plotting/colors.py`, `geoutils/plotting/maps/fields.py`
- Test: `tests/test_colors.py` (create)

**Interfaces:**
- Produces: `resolve_color_scale(z, vmin=None, vmax=None, levels=None, color_bounds="quantile", extend=None) -> ColorScale`, where `ColorScale` is a frozen dataclass `(vmin: float, vmax: float, levels: np.ndarray | None, extend: str, sci: int | None, round_dec: int | None)`. Absorbs the vmin/vmax/levels/rounding/sci block from `plot_array` and the logic of `set_cb_boundaries`.

- [ ] **Step 1: Write characterization tests** in `tests/test_colors.py` against current results, using values taken by running the current `plot_array` on fixed arrays: `test_quantile_bounds`, `test_minmax_bounds`, `test_absmax_symmetric`, `test_extend_both_when_outliers`, `test_extend_neither_when_forced`, `test_levels_have_no_duplicates`, `test_constant_array_widened`.
- [ ] **Step 2:** Implement `resolve_color_scale`; run the tests to confirm they pass.
- [ ] **Step 3:** Replace the inline block in `plot_array` and the body of `set_cb_boundaries` with calls to it; delete `set_cb_boundaries` if nothing else uses it (grep first; if it is exported, keep it as a one-line wrapper).
- [ ] **Step 4: Run** full suite; PASS. **Commit** `refactor(plotting): single colour-scale resolver`.

### Task 8: `plot_2d` simplification

**Files:** Modify `geoutils/plotting/xy.py`; Test: `tests/test_xy_plots.py` (extend).

**Interfaces:**
- Produces: private `_pick(values: Sequence, idx: int)` returning `values[idx]` or the last element when `idx` is out of range, replacing the five repeated fallbacks; a short numpy-style docstring on `plot_2d` listing every supported option.

- [ ] **Step 1:** Add `test_pick_falls_back_to_last` and `test_plot_2d_per_series_styles` (3 series, `lw_arr=[1, 2]`: line widths `[1, 2, 2]`). Run: the second passes against the current code, confirming the helper preserves behavior.
- [ ] **Step 2:** Introduce `_pick`, replace the repeated blocks, split `plot_2d` into `_draw_xy`, `_draw_bar`, `_draw_density` private functions called by a short dispatcher.
- [ ] **Step 3: Run** full suite; PASS. **Commit** `refactor(plotting): simplify plot_2d`.

### Task 9: Explicit arguments, specs and deprecations for maps

**Files:**
- Modify: `geoutils/plotting/maps/fields.py`, `geoutils/plotting/maps/projections.py`, `geoutils/plotting/colors.py`
- Test: `tests/test_plotting.py` (extend)

**Interfaces:**
- Produces: `get_projection` as a `dict[str, Callable]` lookup, raising `ValueError` listing valid names for unknown input.
- Produces: frozen dataclass `ColorbarSpec(label: str | None, orientation: str = "horizontal", tick_step: int = 2, sci: int | None = None, shift_ticks: bool | None = None, pad: float | None = None, fontsize: float | None = None)` in `colors.py`; `make_colorbar` accepts `spec: ColorbarSpec`.
- Produces: `plot_array(z=None, x=None, y=None, *, fig=None, ax=None, plot_type="colormesh", projection=None, vmin=None, vmax=None, cmap="viridis", color_bounds="quantile", label=None, title=None, significance_mask=None, colorbar: ColorbarSpec | None = None, **kwargs)`; legacy keywords (`orientation`, `tick_step`, `sci`, `shift_ticks`, `pad`) are still read from `kwargs` but trigger `DeprecationWarning` naming the replacement. Unknown keywords raise `TypeError` listing them.
- Produces: `draw_significance(ax, x, y, mask, projection, *, kind="hatch", hatch_type="///") -> mappable` extracted from `plot_map` / `plot_array`.

- [ ] **Step 1: Tests:** `test_get_projection_unknown_lists_valid`, `test_legacy_colorbar_kwarg_warns` (`pytest.warns(DeprecationWarning, match="tick_step")`), `test_unknown_kwarg_raises_typeerror` (`plot_map(da, not_an_option=1)`), `test_colorbar_spec_equivalent_to_legacy` (same ticks and label either way), `test_significance_hatch_draws` (`plot_map(da, significance_mask=mask)` adds a hatched collection).
- [ ] **Step 2: Run;** FAIL where noted. Implement the signatures above; the kwargs that remain legitimate pass-throughs (matplotlib artist options for `contour`, e.g. `ls`, `clabel`) are listed in one module-level tuple and documented.
- [ ] **Step 3: Run** full suite and `tutorials/plotting_tutorial.ipynb` calls via a script that executes its code cells under Agg, checking no `TypeError`; fix any legacy keyword the tutorials use by adding it to the deprecation table.
- [ ] **Step 4: Commit** `refactor(plotting): explicit map arguments, ColorbarSpec, deprecations`.

### Task 10: `prepare_axis` and cleanup

**Files:** Modify `geoutils/plotting/axes.py`, `geoutils/plotting/dendrogram.py`; Test: `tests/test_plotting.py`.

**Interfaces:**
- Produces: `AxisSpec` frozen dataclass in `axes.py` holding the options `prepare_axis` reads today (limits, labels, ticks, spines, scales, rotation, invert flags); `prepare_axis(ax, spec: AxisSpec | None = None, **legacy) -> tuple[Axes, dict]` where `legacy` keys are mapped onto `AxisSpec` with `DeprecationWarning` only for renamed ones (`flip_x`/`invert_x`, `sci`).

- [ ] **Step 1: Tests:** `test_prepare_axis_spec_sets_limits_and_labels`, `test_prepare_axis_legacy_kwargs_equal_spec`, `test_flip_x_deprecated_alias`. Run; FAIL.
- [ ] **Step 2:** Implement; keep the returned leftover-kwargs dict contract that `plot_2d` relies on.
- [ ] **Step 3: Dendrogram:** grep for callers of `get_link_colors`, `get_id_to_coord`, `plot_path`, `plot_dendrogram`. If they are only used as methods bound onto another class elsewhere, leave them and report to the user; if unused, report them as dead code and ask before removing.
- [ ] **Step 4:** `ruff check` and `ruff format --diff` on `geoutils/plotting/`; list (do not fix) pre-existing issues outside lines you changed. Run full suite. **Commit** `refactor(plotting): AxisSpec for prepare_axis`.

---

## Phase 4: New look

### Task 11: Explicit style, no import-time side effects

**Files:** Modify `geoutils/plotting/style.py`, `geoutils/plotting/__init__.py`, and each public plotting function that creates a figure; Test: `tests/test_style.py` (create).

**Interfaces:**
- Produces: `STYLE: dict` of rcParams (white background, light grid `grid.alpha=0.3`, outward ticks, no top/right spines, `savefig.dpi=300`, `savefig.bbox="tight"`, font sizes from the existing constants, `axes.prop_cycle` = Okabe-Ito `["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]`); `set_style() -> None` applies `STYLE` globally; `style_context()` returns `plt.rc_context(STYLE)`. Public figure-creating functions wrap drawing in `style_context()`. `change_fsize`/`reset_fsize` share one `_apply_sizes(fsize)` helper with consistent title sizing; `colors` list removed from `style.py` (palette comes from `prop_cycle`).

- [ ] **Step 1: Tests:** `test_import_does_not_change_rcparams` (run `python -I -c` in a subprocess: record `matplotlib.rcParams` copy, `import geoutils.plotting`, assert equal), `test_plot_does_not_leak_rcparams` (snapshot before/after `plot_2d`), `test_set_style_applies_cycle` (`rcParams["axes.prop_cycle"].by_key()["color"][0] == "#E69F00"`), `test_plot_2d_uses_palette_for_many_series`. Run; the first two FAIL.
- [ ] **Step 2:** Remove import-time `plt.rcdefaults()`, `plt.style.use('bmh')`, the global `warnings.filterwarnings`, and the duplicated `preferred_font` assignment; implement the interfaces; remove `pst.colors` indexing in `xy.py` (rely on the cycle). Keep the font fallback inside `STYLE` construction.
- [ ] **Step 3:** Update `tutorials/plotting_tutorial.ipynb` first code cell to call `set_style()` only if it relied on import-time styling (check; mention to the user, do not rewrite notebook outputs).
- [ ] **Step 4: Run** full suite; PASS. **Commit** `feat(plotting): explicit geoutils style, Okabe-Ito palette`.

### Task 12: Robust, symmetric-for-diverging colour bounds

**Files:** Modify `geoutils/plotting/colors.py` (`resolve_color_scale`), `geoutils/plotting/maps/fields.py`; Test: `tests/test_colors.py`.

**Interfaces:**
- Produces: `color_bounds` default becomes `"robust"` (2nd-98th percentile via `np.nanquantile`); `"quantile"` (10-90) stays available; `"auto"` is not added (YAGNI). When `cmap` is a diverging map (name in a module constant `DIVERGING = {"RdBu", "RdBu_r", "RdYlBu", "RdYlBu_r", "coolwarm", "bwr", "seismic", "BrBG", "PiYG", "PRGn", "PuOr", "RdGy", "Spectral"}` plus their `_r` variants) and neither limit is set explicitly, the bounds are symmetric (`absmax` of the robust bounds).

- [ ] **Step 1: Tests:** `test_default_is_robust` (fixture with one huge outlier: `vmax < data.max()` and `extend in ("max", "both")`), `test_diverging_cmap_symmetric_by_default` (`vmin == -vmax`), `test_sequential_cmap_not_symmetric` (all-positive data, `viridis`: `vmin >= 0`), `test_explicit_limits_win` (`vmin=-1, vmax=3` unchanged even for `RdBu`), `test_quantile_still_available`.
- [ ] **Step 2: Run;** FAIL where noted. Implement. Update the `plot_array` docstring note about clipping.
- [ ] **Step 3: Run** full suite; PASS. **Commit** `feat(plotting): robust symmetric colour bounds`.

### Task 13: Layout-driven colorbars and titles

**Files:** Modify `geoutils/plotting/colors.py` (`make_colorbar`, `add_colorbar`), `geoutils/plotting/figure.py` (`create_multi_plot`, `create_plot`), `geoutils/plotting/maps/fields.py`, `geoutils/plotting/maps/basemap.py`; Test: `tests/test_plotting.py`.

**Interfaces:**
- Produces: `make_colorbar(ax, im, fig=None, spec: ColorbarSpec | None = None, **legacy)` places the bar with `fig.colorbar(im, ax=ax, orientation=..., pad=..., shrink=..., location=...)` rather than `inset_axes` with `borderpad`; `ColorbarSpec.pad` becomes a fraction (default `0.05`) and `shrink` (default `0.9`). `create_plot` and `create_multi_plot` use `layout="constrained"`; `y_title` offset hacks are removed in favour of layout.

- [ ] **Step 1: Tests:** `test_colorbar_does_not_overlap_map` (after `fig.canvas.draw()`, colorbar axes bbox and map axes bbox do not intersect, horizontal and vertical), `test_multi_plot_shared_colorbar_inside_figure` (`add_colorbar(..., multi_plots=True)` bbox within `fig.bbox`), `test_title_not_clipped` (title window extent inside figure extent for a map and an xy axis), `test_save_fig_tight_roundtrip` (saved PNG exists and non-empty).
- [ ] **Step 2: Run;** expect some FAIL (overlap/clipping) against the old placement.
- [ ] **Step 3:** Implement; render the tutorials' main plot types (`colormesh`, `contourf`, `discrete`, multi-panel) to PNGs in the scratchpad and look at them (Read the image files) to confirm no overlap or clipped labels before committing.
- [ ] **Step 4: Run** full suite and `ruff` on changed files. **Commit** `feat(plotting): constrained layout colorbars and titles`.

### Task 14: Wrap-up

**Files:** Modify `README.md` (plotting usage snippet only, if present); `geoutils/plotting/__init__.py` docstring (mention `set_style()`).

- [ ] **Step 1:** Full run: `python -m pytest -q` and `ruff check geoutils/plotting tests` (changed files only for format). Report line counts of `geoutils/plotting/**/*.py` versus the 3,838 baseline, any file over ~600 lines, and the list of pre-existing dead code and issues found but not touched.
- [ ] **Step 2:** Update the docs mentions above. **Commit** `docs(plotting): document set_style and layout`.
- [ ] **Step 3:** Hand off to the user with the output format from `CLAUDE.md` (CHANGES / NOT TOUCHED / VALIDATION / CONCERNS); do not merge or push.
