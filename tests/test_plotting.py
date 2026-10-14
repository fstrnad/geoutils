"""Smoke + characterization tests for the map plotting entry points.

These cover each plot_type branch of plot_array so the dispatch refactor can be
verified to preserve behavior.
"""
import inspect

import numpy as np
import pytest

import geoutils.plotting.maps.fields as mp


def test_plot_map_returns_expected_artifacts(latlon_da):
    res = mp.plot_map(latlon_da)
    # PlotResult keeps dict-style access for back-compat.
    for key in ("ax", "fig", "im"):
        assert key in res
        assert res[key] is not None


@pytest.mark.parametrize("plot_type", ["colormesh", "contourf", "contour"])
def test_plot_map_plot_types(latlon_da, plot_type):
    res = mp.plot_map(latlon_da, plot_type=plot_type)
    assert res["im"] is not None


# --- plot_array directly: one test per draw branch -----------------------

@pytest.mark.parametrize("plot_type", ["colormesh", "contourf", "discrete"])
def test_plot_array_field_types(latlon_da, plot_type):
    res = mp.plot_array(z=latlon_da, plot_type=plot_type)
    assert res["im"] is not None


def test_plot_array_scatter():
    x = np.array([0.0, 1, 2, 3])
    y = np.array([0.0, 1, 0, 1])
    z = np.array([1.0, 2, 3, 4])
    res = mp.plot_array(x=x, y=y, z=z, plot_type="scatter")
    assert res["im"] is not None


def test_plot_array_points():
    res = mp.plot_array(x=np.array([0.0, 1, 2]), y=np.array([0.0, 1, 2]),
                        plot_type="points")
    assert res["ax"] is not None


def test_plot_array_hatch(latlon_da):
    res = mp.plot_array(z=(latlon_da > 20).astype(float), plot_type="hatch")
    assert res["im"] is not None


def test_color_bounds_is_explicit_parameter():
    sig = inspect.signature(mp.plot_array)
    assert "color_bounds" in sig.parameters
    assert sig.parameters["color_bounds"].default == "robust"
    assert inspect.signature(mp.plot_map).parameters["color_bounds"].default == "robust"


# --- significance overlays ---------------------------------------------

def _hatched(ax):
    return [c for c in ax.collections if c.get_hatch()]


def test_significance_mask_hatches_map(latlon_da):
    mask = latlon_da > latlon_da.median()
    res = mp.plot_map(latlon_da, significance_mask=mask)
    assert len(_hatched(res["ax"])) == 1


def test_significance_contour_draws_lines(latlon_da):
    mask = latlon_da > latlon_da.median()
    plain = mp.plot_map(latlon_da)
    res = mp.plot_map(latlon_da, significance_mask=mask, sig_plot_type="contour")
    assert len(res["ax"].collections) > len(plain["ax"].collections)


def test_significance_mask_false_rejected(latlon_da):
    with pytest.raises(ValueError, match="not False"):
        mp.plot_map(latlon_da, significance_mask=False)


def test_significance_mask_shape_mismatch_rejected(latlon_da):
    with pytest.raises(ValueError, match="same dimension"):
        mp.plot_map(latlon_da, significance_mask=latlon_da[:2])


# --- explicit options and unknown-keyword warning ----------------------

def test_plot_array_returns_colorbar_when_labelled(latlon_da):
    res = mp.plot_map(latlon_da, label="K")
    assert res["colorbar"] is not None
    assert res["colorbar"].ax.get_xlabel() == "K"


def test_plot_array_no_colorbar_without_label(latlon_da):
    assert mp.plot_map(latlon_da)["colorbar"] is None


def test_unknown_keyword_warns_and_names_it(latlon_da):
    with pytest.warns(FutureWarning, match="not_an_option"):
        mp.plot_map(latlon_da, not_an_option=1)


def test_documented_keywords_do_not_warn(latlon_da, recwarn):
    mp.plot_map(latlon_da, label="K", orientation="vertical", tick_step=1,
                levels=6, extend="both", centercolor="white", round_dec=1,
                title="t", set_map=True, land_ocean=True, hatch_type="//",
                y_title=1.1, fontsize=10)
    assert not [w for w in recwarn if w.category is FutureWarning]


def test_axis_options_documented_for_plain_arrays(recwarn):
    mp.plot_array(np.arange(12.0).reshape(3, 4), xlabel="x", ylabel="y",
                  xlim=(0, 3), levels=4, plot_type="contourf")
    assert not [w for w in recwarn if w.category is FutureWarning]


@pytest.mark.parametrize("module, names", [
    ("axes", ("TITLE_OPTIONS",)),
    ("maps.basemap", ("BASEMAP_OPTIONS",)),
])
def test_option_registries_match_source(module, names):
    import importlib
    import inspect
    mod = importlib.import_module(f"geoutils.plotting.{module}")
    src = inspect.getsource(mod)
    for name in names:
        for key in getattr(mod, name):
            assert f"'{key}'" in src or f'"{key}"' in src, (name, key)


# --- layout: colorbars and titles ---------------------------------------

def _extent(artist, fig):
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer())


def _cbar_axes(fig):
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_colorbar_does_not_overlap_map(latlon_da, orientation):
    res = mp.plot_map(latlon_da, label="K", orientation=orientation)
    (cax,) = _cbar_axes(res["fig"])
    cbar_box = _extent(cax, res["fig"])
    map_box = _extent(res["ax"], res["fig"])
    assert not cbar_box.overlaps(map_box)


def test_colorbar_stays_inside_figure(latlon_da):
    res = mp.plot_map(latlon_da, label="Temperature [K]")
    fig = res["fig"]
    (cax,) = _cbar_axes(fig)
    tight = cax.get_tightbbox(fig.canvas.get_renderer())
    fig_box = fig.bbox
    assert fig_box.x0 <= tight.x0 and tight.x1 <= fig_box.x1
    assert fig_box.y0 <= tight.y0 and tight.y1 <= fig_box.y1


def test_legacy_negative_pad_is_ignored_with_warning(latlon_da):
    with pytest.warns(DeprecationWarning, match="pad"):
        res = mp.plot_map(latlon_da, label="K", pad=-2.5)
    assert len(_cbar_axes(res["fig"])) == 1


@pytest.mark.parametrize("make", [
    lambda da: mp.plot_map(da, title="A title"),
    lambda da: cplt_xy(da),
], ids=["map", "xy"])
def test_title_not_clipped(latlon_da, make):
    res = make(latlon_da)
    fig = res["fig"]
    ax = res["ax"]
    title_box = _extent(ax.title, fig)
    assert fig.bbox.y0 <= title_box.y0 and title_box.y1 <= fig.bbox.y1
    assert fig.bbox.x0 <= title_box.x0 and title_box.x1 <= fig.bbox.x1


def cplt_xy(_):
    import geoutils.plotting as cplt
    return cplt.plot_2d(y=np.arange(5.0), title="A title")


def test_titles_do_not_overlap_map_labels(latlon_da):
    res = mp.plot_map(latlon_da, title="A title")
    fig = res["fig"]
    title_box = _extent(res["ax"].title, fig)
    map_box = _extent(res["ax"], fig)
    assert title_box.y0 >= map_box.y1 - 1


@pytest.mark.parametrize("make", [
    lambda da: mp.plot_map(da),
    lambda da: __import__("geoutils.plotting", fromlist=["x"]).plot_2d(y=np.arange(4.0)),
    lambda da: __import__("geoutils.plotting", fromlist=["x"]).create_plot(),
    lambda da: __import__("geoutils.plotting", fromlist=["x"]).create_plot(2, 2),
    lambda da: __import__("geoutils.plotting", fromlist=["x"]).create_multi_plot(
        2, 2, projection="PlateCarree"),
], ids=["map", "xy", "plot1", "plot4", "multi"])
def test_figures_use_constrained_layout(latlon_da, make):
    from matplotlib.layout_engine import ConstrainedLayoutEngine
    res = make(latlon_da)
    assert isinstance(res["fig"].get_layout_engine(), ConstrainedLayoutEngine)


def test_multi_plot_shared_colorbar_inside_figure(latlon_da):
    import geoutils.plotting as cplt
    res = cplt.create_multi_plot(2, 2, projection="PlateCarree")
    for ax in res["ax"]:
        im = mp.plot_map(latlon_da, ax=ax)
    cplt.add_colorbar(im, res["fig"], "K", multi_plots=True)
    fig = res["fig"]
    (cax,) = _cbar_axes(fig)
    box = _extent(cax, fig)
    assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1
    assert fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1
    for ax in res["ax"]:
        assert not box.overlaps(_extent(ax, fig))


def test_multi_plot_spacing_arguments_are_applied():
    import geoutils.plotting as cplt
    res = cplt.create_multi_plot(1, 2, hspace=0.1, wspace=0.3)
    engine = res["fig"].get_layout_engine()
    assert engine.get()["wspace"] == 0.3 and engine.get()["hspace"] == 0.1


def test_save_fig_tight_roundtrip(latlon_da, tmp_path):
    import geoutils.plotting as cplt
    res = mp.plot_map(latlon_da, label="K", title="t")
    path = cplt.save_fig(tmp_path / "map", fig=res["fig"])
    assert path.suffix == ".png" and path.stat().st_size > 1000


def _label_size(ax):
    from cartopy.mpl.gridliner import Gridliner
    (gl,) = [c for c in ax.get_children() if isinstance(c, Gridliner)]
    return gl.xlabel_style["size"]


def test_gridline_labels_have_one_size_on_every_map():
    import geoutils.plotting as cplt
    wide = cplt.create_map(figsize=(9, 5))["ax"]
    narrow = cplt.create_multi_plot(2, 3, projection="PlateCarree", plot_grid=True,
                                    figsize=(9, 5))["ax"][0]
    import geoutils.plotting.style as style
    assert _label_size(narrow) == _label_size(wide) == style.get_preset().tick


def test_vertical_colorbar_is_shorter_than_horizontal_is_wide(latlon_da):
    def ratio(res, dim):
        fig = res["fig"]
        fig.canvas.draw()
        (cax,) = _cbar_axes(fig)
        box = fig.transFigure.transform_bbox(res["ax"].get_position(original=True))
        return getattr(_extent(cax, fig), dim) / getattr(box, dim)
    horizontal = mp.plot_map(latlon_da, label="K")
    vertical = mp.plot_map(latlon_da, label="K", orientation="vertical")
    assert ratio(vertical, "height") < ratio(horizontal, "width")


# --- map furniture and colorbar style -----------------------------------

def _gridliner(ax):
    from cartopy.mpl.gridliner import Gridliner
    (gl,) = [c for c in ax.get_children() if isinstance(c, Gridliner)]
    return gl


def test_titles_are_left_aligned(latlon_da):
    res = mp.plot_map(latlon_da, title="A title")
    assert res["ax"].get_title(loc="left") == "A title"
    assert res["ax"].get_title(loc="center") == ""


def test_title_loc_can_be_overridden(latlon_da):
    res = mp.plot_map(latlon_da, title="A title", title_loc="center")
    assert res["ax"].get_title(loc="center") == "A title"


def test_gridlines_are_thin_dotted_and_translucent():
    import geoutils.plotting as cplt
    gl = _gridliner(cplt.create_map()["ax"])
    kw = gl.collection_kwargs
    assert kw["linestyle"] == ":" and kw["linewidth"] <= 0.5 and kw["alpha"] < 1


def test_coastlines_are_thin_and_dark_grey():
    from cartopy.mpl.feature_artist import FeatureArtist
    import geoutils.plotting as cplt
    ax = cplt.create_map()["ax"]
    (coast,) = [a for a in ax.get_children() if isinstance(a, FeatureArtist)]
    assert coast._kwargs["lw"] == 0.6 and coast._kwargs["edgecolor"] == "#333333"


def test_explicit_coast_options_still_win():
    from cartopy.mpl.feature_artist import FeatureArtist
    import geoutils.plotting as cplt
    ax = cplt.create_map(coast_color="red", lw_borders=2)["ax"]
    (coast,) = [a for a in ax.get_children() if isinstance(a, FeatureArtist)]
    assert coast._kwargs["lw"] == 2 and coast._kwargs["edgecolor"] == "red"


def test_map_frame_is_thin_grey():
    import geoutils.plotting as cplt
    import geoutils.plotting.style as style
    spine = cplt.create_map()["ax"].spines["geo"]
    assert spine.get_linewidth() == style.get_preset().axes_width and spine.get_edgecolor()[0] < 0.5


def test_colorbar_is_slim_with_fine_outline(latlon_da):
    res = mp.plot_map(latlon_da, label="K")
    box = _extent(_cbar_axes(res["fig"])[0], res["fig"])
    assert box.height < 0.04 * res["fig"].bbox.height
    import geoutils.plotting.style as style
    assert res["colorbar"].outline.get_linewidth() == style.get_preset().axes_width


@pytest.mark.parametrize("levels", [4, 8, 12, 20])
def test_colorbar_shows_at_most_six_ticks_by_default(latlon_da, levels):
    res = mp.plot_map(latlon_da, label="K", plot_type="contourf", levels=levels)
    assert len(res["colorbar"].get_ticks()) <= 6


def test_explicit_tick_step_is_kept(latlon_da):
    res = mp.plot_map(latlon_da, label="K", plot_type="contourf", levels=8,
                      tick_step=1)
    assert len(res["colorbar"].get_ticks()) == 9


def test_panel_letters_use_the_small_title_size():
    import geoutils.plotting as cplt
    res = cplt.create_multi_plot(1, 2)
    letters = [t for ax in res["ax"] for t in ax.texts]
    assert {t.get_text() for t in letters} == {"a.", "b."}
    import geoutils.plotting.style as style
    assert all(t.get_fontsize() == style.get_preset().title for t in letters)
