# Plotting presets and bundled font: design

Date: 2026-10-14. Scope: `geoutils/plotting/` (style, figure, axes, maps) and
packaging. Sub-project 1 of 3 on making plots look professional; colorbar
styles (2) and xy-plot polish (3) get their own specs.

## Goal

Plots that look professional out of the box for both papers and talks: one
shared look, a `paper` and a `talk` preset that scale text, line weights and
figure sizes, and a bundled font so output is identical on every machine.

Success criteria:
- `paper` is the default look; `talk` is available with one call.
- Text, line widths and default figure sizes follow the active preset
  everywhere in `geoutils.plotting`; no plotting code uses a hard-coded size.
- The bundled font is used when installed from a wheel and when run from source.
- Importing `geoutils.plotting` still does not modify global matplotlib state.
- Existing tests pass, apart from ones asserting the old default sizes.

## Decisions (agreed with the user)

- Presets: `paper` (default) and `talk`. A visible change: default text shrinks
  from 10-13 pt to about 7-9 pt and default figures shrink to journal widths.
- Font: bundle one OFL font (Inter) inside the package.
- No `preset=` argument on individual plot functions; presets are global or
  scoped with a context manager.
- Out of scope: colorbar styles, xy-plot changes, map-furniture tweaks beyond
  what the line-width preset implies.

## Design

### Preset mechanism (`style.py`)

A frozen `Preset` dataclass holds what differs between looks: tick, label,
title and figure-title font sizes; line width; axes and tick line width;
marker size; panel width and aspect ratio. `STYLE` (the rc dict) is built from
a preset plus the shared parts (palette, spines, font chain, grid, dpi).

| Preset | Text (tick / label / title) | Panel width | Line width |
|---|---|---|---|
| `paper` | ~7 / 8 / 9 pt | 3.4 in, grids capped at 7.1 in | thin |
| `talk` | ~14 / 16 / 18 pt | ~6 in | heavy (~2.5) |

Exact values are fixed during implementation by rendering the examples; the
table gives the intended scale.

API:
- `set_style(preset="paper")` sets the active preset and applies it globally.
- `style_context(preset=None)` applies a preset only inside a `with` block;
  `None` means the active preset. `@styled` uses it, so plot functions follow
  the active preset with no new arguments.
- `change_fsize` / `reset_fsize` keep working on the active preset.

The `*_SIZE` constants stay importable for compatibility, but plotting code no
longer uses them. About 12 call sites (`axes.py`, `figure.py`,
`maps/basemap.py`, `maps/fields.py`) switch to reading `rcParams` at call time:
signature defaults become `None` and are resolved inside the function.

### Bundled font

- Inter (SIL OFL): Regular, Italic, SemiBold, Bold `.ttf` (~1.2 MB) plus the
  OFL license text, in `geoutils/plotting/fonts/`. Listed under
  `[tool.setuptools.package-data]` in `pyproject.toml`.
- `style.py` registers the files with `font_manager.fontManager.addfont` once,
  on first use of `style_context()` or `set_style()`, not at import.
- Font chain: `["Inter", "Lato", "Arial", "Liberation Sans", "DejaVu Sans"]`,
  so a failed load still falls back as today.
- No tabular-figure workaround: matplotlib cannot select OpenType features,
  and Inter's default figures are close to tabular. True minus signs already
  come from matplotlib's `axes.unicode_minus` default.
- The font files are fetched from the official Inter release at
  implementation time.

### Figure sizing (`figure.py`, `maps/basemap.py`)

- `create_plot`: figure width is `min(ncols * panel_width, double_column)`;
  per-panel height is the actual per-panel width times the preset aspect
  (about 0.75). An explicit `figsize` always wins.
- `create_map` and `create_multi_plot` use the same width rule. Map aspect
  still follows the projection; only width comes from the preset.
- `enumerate_subplots` default size and offset come from the preset.
- `save_fig` is unchanged (300 dpi PNG; PDF/SVG already supported).

## Testing

Unit tests (pytest):
- Each preset yields the expected `rcParams` (font sizes, line widths).
- `style_context("talk")` restores previous settings on exit; `set_style` does
  not leak into a later context.
- The bundled font is found and `font.family` resolves to Inter.
- `create_plot` default size follows the preset; explicit `figsize` overrides.
- Font registration is lazy: importing the package registers nothing.

Visual check: re-render the eight example scripts under `paper` and `talk` and
inspect the PNGs next to the current ones.

Regression: existing plotting tests pass unchanged except those asserting old
default sizes, which are updated deliberately.

## Not doing

- Per-function `preset=` arguments.
- Colorbar or xy changes (sub-projects 2 and 3).
- A tabular-figure hack.

## Risks

- Default look changes for all existing users (smaller text, narrower figures);
  mitigated by `set_style("talk")` and an explicit `figsize`.
- Tutorials may pass stale or size-dependent keywords; unknown keywords only
  emit a FutureWarning today, so they keep running.
- Text that is already tight at small sizes (long map tick labels, colorbar
  labels) must be checked in the visual pass.
