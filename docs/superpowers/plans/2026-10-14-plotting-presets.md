# Plotting presets and bundled font Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `paper` (default) and `talk` presets that scale text, line weights and default figure sizes everywhere in `geoutils.plotting`, plus a bundled Inter font.

**Architecture:** A frozen `Preset` dataclass in `style.py` is the single source of every size. The active preset is module state; `set_style` applies it globally and `style_context` scopes it. Plot code stops using the `*_SIZE` constants and reads the active preset at call time. Figure sizes come from one helper in `figure.py`.

**Tech Stack:** matplotlib rcParams / `font_manager`, setuptools package-data, pytest. Tests run with `.venv/bin/python -m pytest`; lint with `uvx ruff` on changed files only.

**Spec:** `docs/superpowers/specs/2026-10-14-plotting-presets-design.md`

## Global Constraints

- Importing `geoutils.plotting` must not change `matplotlib.rcParams` or register fonts (`tests/test_style.py::test_import_does_not_change_rcparams` stays green).
- Plot functions must not leak rcParams (`test_plot_does_not_leak_rcparams` stays green).
- An explicit `figsize` always wins over a preset default.
- No `preset=` argument on individual plot functions.
- `*_SIZE` constants and `style.colors` stay importable.
- No colorbar, xy-feature or map-furniture changes beyond what sizes imply.
- Never use shell `rm`; ask the user to delete files.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Unknown preset name (`set_style("poster")`): `ValueError` that lists `paper, talk`; nothing is applied. Pinned in Task 1.
- `style_context()` with no argument after `set_style("talk")` uses talk, not paper; nested contexts restore correctly. Pinned in Task 1.
- Text helpers called directly outside `@styled` (`plot_text`, `text_box`, `set_title`) follow the active preset, not matplotlib's global default. Pinned in Task 2.
- Wide grids (`ncols=6`) stay within the double-column width instead of growing without bound. Pinned in Task 3.
- Missing or unreadable font files: no exception, the chain falls back. Pinned in Task 4.
- PDF/SVG export embeds text as TrueType, not Type 3 (journals reject Type 3). Pinned in Task 4. This is one rc line beyond the spec's list.

---

## File Structure

- Modify `geoutils/plotting/style.py`: `Preset`, `PRESETS`, `get_preset`, `build_style`, preset-aware `set_style` / `style_context` / `change_fsize` / `reset_fsize`, font registration.
- Modify `geoutils/plotting/figure.py`: `default_figsize`, used by `create_plot` and `create_multi_plot`; `enumerate_subplots` size from preset.
- Modify `geoutils/plotting/axes.py`, `maps/basemap.py`, `maps/fields.py`, `maps/overlays.py`, `xy.py`: replace constants and hard-coded sizes.
- Create `geoutils/plotting/fonts/` (Inter `.ttf` files + `OFL.txt`); modify `pyproject.toml`.
- Create `tests/test_style_presets.py`; modify `tests/conftest.py` and `tests/test_style.py`.
- Modify `geoutils/plotting/__init__.py` docstring (presets), `examples/` figures and README (Task 5).

---

### Task 1: Preset mechanism in `style.py`

**Files:**
- Modify: `geoutils/plotting/style.py`, `tests/conftest.py`, `tests/test_style.py`
- Create: `tests/test_style_presets.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Preset` with fields `tick, label, title, figure_title, line_width, axes_width, tick_length, marker_size, panel_width, max_width, aspect` (all `float`).
  - `PRESETS: dict[str, Preset]` with keys `"paper"`, `"talk"`.
  - `get_preset(name: str | None = None) -> Preset`: `None` returns the active preset; unknown name raises `ValueError` naming the valid keys.
  - `build_style(preset: Preset) -> dict`: the rc dict.
  - `set_style(preset: str | None = None) -> None`: `None` keeps the active preset; sets it active and updates `rcParams`.
  - `style_context(preset: str | None = None)`: context manager; a named preset applies only inside the block and does not change the active preset for code that calls `get_preset()` outside it. Inside the block `get_preset()` returns the scoped preset.
  - `STYLE = build_style(PRESETS["paper"])` (compat).
- Starting values (tuned in Task 5): paper `tick=7, label=8, title=9, figure_title=10, line_width=1.2, axes_width=0.6, tick_length=3, marker_size=3, panel_width=3.4, max_width=7.1, aspect=0.75`; talk `tick=14, label=16, title=18, figure_title=20, line_width=2.5, axes_width=1.2, tick_length=6, marker_size=7, panel_width=6.0, max_width=13.0, aspect=0.75`.

- [ ] **Step 1: Add an autouse fixture to `tests/conftest.py`** that does `monkeypatch.setattr(style, "_active", "paper")` (import `geoutils.plotting.style as style`), so tests that call `set_style("talk")` cannot leak the active preset.
- [ ] **Step 2: Write failing tests in `tests/test_style_presets.py`:**
  - `test_paper_is_default`: `get_preset() is PRESETS["paper"]`.
  - `test_build_style_uses_preset_sizes`: for each preset, `build_style(p)["xtick.labelsize"] == p.tick`, `["axes.labelsize"] == p.label`, `["axes.titlesize"] == p.title`, `["figure.titlesize"] == p.figure_title`, `["lines.linewidth"] == p.line_width`, `["axes.linewidth"] == p.axes_width`, `["xtick.major.size"] == p.tick_length`, `["lines.markersize"] == p.marker_size`.
  - `test_style_context_scopes_preset`: inside `style_context("talk")`, `get_preset().tick == 14` and `rcParams["xtick.labelsize"] == 14`; after the block `get_preset()` is paper and rcParams are restored.
  - `test_context_without_name_uses_active_preset`: after `set_style("talk")` (restore rcParams in `finally`), `style_context()` yields talk sizes.
  - `test_nested_contexts_restore`: talk inside paper inside talk returns to the outer preset at each exit.
  - `test_unknown_preset_raises`: `pytest.raises(ValueError, match="paper")` for `set_style("poster")` and `get_preset("poster")`; `get_preset()` is unchanged afterwards.
  - `test_fsize_helpers_follow_active_preset`: after `set_style("talk")`, `reset_fsize()` restores `rcParams["axes.titlesize"] == 18`.
- [ ] **Step 3: Update three assertions in `tests/test_style.py` deliberately:** `test_type_scale_is_compact` expects `(7, 8, 9)` from `style.STYLE`; `test_plot_2d_default_line_width_is_readable` expects `PRESETS["paper"].line_width` (rename to `..._follows_the_preset`); `test_map_annotations_use_the_label_size` expects `style.get_preset().label`. `test_font_chain...` waits for Task 4.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/test_style_presets.py -q`. Expected: FAIL (`PRESETS` undefined).
- [ ] **Step 5: Implement in `style.py`.** Keep the existing constants, `OKABE_ITO`, `FONT_CHAIN`, `colors`, helper functions. Replace `_sizes` / `_BASE_SIZES` / `STYLE` with `build_style`. Module global `_active = "paper"`. `style_context(preset)` resolves the name, temporarily sets `_active` for its duration (try/finally), and wraps `plt.rc_context(build_style(...))`. `change_fsize` / `reset_fsize` keep their behaviour on the active preset (`reset_fsize` uses `build_style(get_preset())`'s size keys).
- [ ] **Step 6: Run** `.venv/bin/python -m pytest tests/test_style_presets.py tests/test_style.py -q`. Expected: the new tests pass; `test_plot_2d_default_line_width...` still fails until Task 2 (expected, note it).
- [ ] **Step 7: Commit** `feat(plotting): paper and talk presets in style.py`.

---

### Task 2: Plotting code reads sizes from the active preset

**Files:**
- Modify: `geoutils/plotting/axes.py` (lines 35, 85, 95, 97, 223, 387, 434, 474-476), `geoutils/plotting/figure.py:67`, `geoutils/plotting/maps/basemap.py:151-153`, `geoutils/plotting/maps/fields.py:403`, `geoutils/plotting/xy.py:312`
- Test: `tests/test_style_presets.py`

**Interfaces:**
- Consumes: `style.get_preset() -> Preset` (Task 1).
- Produces: no new public names. Mapping from constants: `TICK_SIZE`/`MINI_SIZE` → `.tick`; `LABEL_SIZE` → `.label`; `TITLE_SIZE`/`BIGGER_SIZE` → `.title`; `FIGURE_TITLE_SIZE` → `.figure_title`; `SMALL_SIZE / 2` tick length → `.tick_length`, tick width `1` → `.axes_width`.

- [ ] **Step 1: Write failing tests** (talk scale makes the difference visible):
  - `test_text_helpers_follow_the_preset_outside_styled`: after `set_style("talk")`, `plot_text(ax, "t")` has `get_fontsize() == 16`; `text_box(ax, "t")` text size is 16 (this also covers the existing bug where `text_box` ignored its `fsize` argument; also assert `text_box(ax, "t", fsize=9)` gives 9).
  - `test_titles_follow_the_preset`: `plot_2d(y=..., title="t")` under `style_context("talk")` has axes title size 18; `create_multi_plot(1, 2)` panel letters have size 18 (talk) and 9 (paper).
  - `test_plot_2d_default_line_width_follows_preset`: `plot_2d(y=np.arange(4.0))` line width is 1.2 (paper) and 2.5 under `style_context("talk")`; an explicit `lw=3` gives 3.
  - `test_tick_length_follows_preset`: `prepare_axis` ticks `get_tick_params()`-based length is 3 under paper, 6 under talk.
  - `test_map_gridline_labels_follow_the_preset`: `create_map(plot_grid=True)` x-label style size is 7 under paper.
- [ ] **Step 2: Run** them; expected FAIL.
- [ ] **Step 3: Replace each site.** Signature defaults that were constants become `None` and are resolved inside (`fontsize = pst.get_preset().title if fontsize is None else fontsize`). In `text_box`, remove the shadowing `kwargs.pop('fsize', ...)` line so the parameter works. `xy.py:312` default `lw_arr` becomes `[None]` so matplotlib uses `lines.linewidth`; check `scatter` and `errorbar` calls still accept `lw=None`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests -q -x`. Expected: all pass, including `test_plot_2d_default_line_width_follows_the_preset` from Task 1.
- [ ] **Step 5: Run** `grep -rnE "pst\.[A-Z_]+_SIZE" geoutils`. Expected: no output.
- [ ] **Step 6: Commit** `refactor(plotting): read sizes from the active preset`.

---

### Task 3: Preset-driven default figure sizes

**Files:**
- Modify: `geoutils/plotting/figure.py` (lines 44-46, 111-113), `maps/basemap.py:335`, `maps/fields.py:91, 522`, `maps/overlays.py:109`, `xy.py:357, 454`
- Test: `tests/test_style_presets.py`

**Interfaces:**
- Produces in `style.py`: `default_figsize(nrows: int = 1, ncols: int = 1, aspect: float | None = None) -> tuple[float, float]`. Width is `min(ncols * panel_width, max_width)`; height is `nrows * (width / ncols) * aspect`, with `aspect` defaulting to `get_preset().aspect`. It lives in `style.py` rather than `figure.py` (the spec's wording) because `figure.py` imports `xy.py` and `maps/*`, so putting it there would be circular; `figure.py` re-exports it.

- [ ] **Step 1: Write failing tests:**
  - `test_default_figsize_single_panel`: paper → `(3.4, 3.4 * 0.75)`; talk → `(6.0, 4.5)`.
  - `test_default_figsize_caps_wide_grids`: `default_figsize(1, 6)[0] == 7.1`; `default_figsize(2, 2)[0] == 6.8`; per-panel height uses the per-panel width.
  - `test_create_plot_uses_preset_size`: `create_plot()["fig"].get_size_inches()` equals paper default; under `style_context("talk")` equals talk default.
  - `test_explicit_figsize_wins` for `create_plot`, `create_multi_plot`, `create_map`, `plot_2d`: `figsize=(9, 5)` gives `(9, 5)`.
  - `test_every_default_figure_fits_double_column`: with paper, `plot_2d`, `plot_map`, `create_map`, `create_multi_plot(2, 3, ...)`, `plot_hist` figures have width ≤ 7.1.
- [ ] **Step 2: Run** them; expected FAIL.
- [ ] **Step 3: Implement** `default_figsize` in `style.py` and replace every default listed above (`create_plot` `(6*ncols, 4*nrows)`, `create_multi_plot` `(4*ncols, 3*nrows)`, `create_map` `(6, 4)`, `plot_map` `(9, 6)`, the two `(7, 5)` sites, `create_axis` `(8, 5)`, `plot_null_model` `(8, 8)` with `aspect=1.0`). Resolve the default only when the caller gave none. `enumerate_subplots` default `fontsize=None` → `get_preset().title` (done in Task 2); verify here.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests -q`. Expected: pass; update `test_plotting.py` assertions only if one fails on an old default size (none found by grep, so none expected).
- [ ] **Step 5: Commit** `feat(plotting): default figure sizes follow the preset`.

---

### Task 4: Bundled Inter font

**Files:**
- Create: `geoutils/plotting/fonts/` (`Inter-Regular.ttf`, `Inter-Italic.ttf`, `Inter-SemiBold.ttf`, `Inter-Bold.ttf`, `OFL.txt`)
- Modify: `geoutils/plotting/style.py`, `pyproject.toml`, `tests/test_style.py`
- Test: `tests/test_style_presets.py`, `tests/test_packaging.py`

**Interfaces:**
- Produces in `style.py`: `register_fonts() -> bool`: registers the bundled files once (idempotent), returns whether Inter is available, never raises. Called by `set_style` and `style_context`.
- `FONT_CHAIN = ["Inter", "Lato", "Arial", "Liberation Sans", "DejaVu Sans"]`.
- `build_style` gains `"pdf.fonttype": 42` and `"ps.fonttype": 42`; SVG stays at matplotlib's default (text as paths) so it renders everywhere.

- [ ] **Step 1: Fetch the files.** From the official Inter release (github.com/rsms/inter, latest 4.x zip) into a new empty directory in the scratchpad, extract, copy the four static TTFs and `LICENSE.txt` as `OFL.txt` into `geoutils/plotting/fonts/`. If the network is unavailable, stop and ask the user for the files.
- [ ] **Step 2: Write failing tests:**
  - `test_register_fonts_finds_inter`: `register_fonts()` is `True` and `"Inter" in {f.name for f in font_manager.fontManager.ttflist}`; `font_manager.findfont("Inter", fallback_to_default=False)` succeeds.
  - `test_register_fonts_is_idempotent`: two calls add each file once (count of Inter entries unchanged).
  - `test_register_fonts_survives_missing_files`: monkeypatch the font directory to an empty tmp path and reset the registered flag; returns `False`, no exception, and a plot still renders (`plot_2d` works).
  - `test_plot_text_renders_in_inter`: `plot_2d(y=..., title="t")` title font family resolves to Inter (`findfont(prop).endswith(".ttf")` and `"Inter" in` its name via `FontProperties`).
  - `test_pdf_embeds_truetype`: `build_style(...)["pdf.fonttype"] == 42`; saving a figure to PDF in `tmp_path` yields bytes containing `/TrueType`.
  - Update `test_font_chain_prefers_designed_fonts_and_ends_in_dejavu` to expect `chain[:4] == ["Inter", "Lato", "Arial", "Liberation Sans"]`.
  - Import test still passes: no font registration at import (existing `test_import_does_not_change_rcparams` plus a subprocess check that `"Inter"` is absent from `fontManager.ttflist` after `import geoutils.plotting`).
  - `tests/test_packaging.py`: the font files exist under the package dir and are listed in `pyproject.toml` package-data (follow that file's existing pattern).
- [ ] **Step 3: Run** them; expected FAIL.
- [ ] **Step 4: Implement** `register_fonts` (module flag + `pathlib.Path(__file__).parent / "fonts"`, `font_manager.fontManager.addfont` per `*.ttf`, wrapped in `try/except Exception` returning `False`), call it at the top of `set_style` and `style_context`, update `FONT_CHAIN`, add the two rc keys, and add `"geoutils.plotting" = ["fonts/*"]` to `[tool.setuptools.package-data]`.
- [ ] **Step 5: Run** `.venv/bin/python -m pytest tests -q`. Expected: pass. Also `.venv/bin/python -m build --wheel` is not required; verify packaging via the `test_packaging.py` assertions.
- [ ] **Step 6: Commit** `feat(plotting): bundle Inter and register it lazily`.

---

### Task 5: Visual pass, tune values, docs

**Files:**
- Modify: `geoutils/plotting/style.py` (preset numbers), `geoutils/plotting/__init__.py` (docstring), `examples/common.py` and `examples/README.md` if they set sizes, `examples/figures/*.png`
- Scratch: render comparison PNGs in the session scratchpad, not the repo.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Re-render all eight example scripts** with the default (`paper`) look and with `set_style("talk")`, writing the talk variants to the scratchpad. Run each script with `.venv/bin/python examples/NN_*.py`.
- [ ] **Step 2: Read the PNGs** (paper, talk, and the committed old versions) and check: tick and colorbar labels not clipped or overlapping, titles and panel letters aligned, line weights readable at the final size, map tick labels fit. Adjust the `Preset` numbers in `style.py` (and the plan's starting table values only in the code) until the figures read well; keep `paper` text between 6 and 9 pt.
- [ ] **Step 3: Update the package docstring** in `geoutils/plotting/__init__.py` with two lines on `set_style("paper"|"talk")` and `style_context("talk")`, and the one-line example in `examples/README.md` if it documents the look.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests -q` and `uvx ruff check` on changed files. Expected: all green, no new ruff findings.
- [ ] **Step 5: Commit** the tuned values, docs and regenerated example figures as `docs(plotting): preset look for example figures`.
- [ ] **Step 6: Report** to the user with the CHANGES MADE / THINGS I DIDN'T TOUCH / VALIDATION / POTENTIAL CONCERNS summary, and list the stray root PNGs and `.superpowers/` scratch (rm is blocked; the user deletes).
