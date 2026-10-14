"""Presets scale text, line weights and sizes from one source of truth."""
import matplotlib as mpl
import pytest

import geoutils.plotting.style as style
from geoutils.plotting.style import (
    PRESETS,
    build_style,
    get_preset,
    set_style,
    style_context,
)


@pytest.fixture
def restore_rc():
    before = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(before)


def test_paper_is_default():
    assert get_preset() is PRESETS["paper"]


@pytest.mark.parametrize("name", ["paper", "talk"])
def test_build_style_uses_preset_sizes(name):
    p, s = PRESETS[name], build_style(PRESETS[name])
    assert s["xtick.labelsize"] == p.tick
    assert s["axes.labelsize"] == p.label
    assert s["axes.titlesize"] == p.title
    assert s["figure.titlesize"] == p.figure_title
    assert s["lines.linewidth"] == p.line_width
    assert s["axes.linewidth"] == p.axes_width
    assert s["xtick.major.size"] == p.tick_length
    assert s["lines.markersize"] == p.marker_size


def test_style_context_scopes_preset():
    before = mpl.rcParams["xtick.labelsize"]
    with style_context("talk"):
        assert get_preset().tick == 14
        assert mpl.rcParams["xtick.labelsize"] == 14
    assert get_preset() is PRESETS["paper"]
    assert mpl.rcParams["xtick.labelsize"] == before


def test_context_without_name_uses_active_preset(restore_rc):
    set_style("talk")
    with style_context():
        assert mpl.rcParams["xtick.labelsize"] == 14


def test_nested_contexts_restore():
    with style_context("talk"):
        with style_context("paper"):
            assert get_preset() is PRESETS["paper"]
        assert get_preset() is PRESETS["talk"]
    assert get_preset() is PRESETS["paper"]


def test_unknown_preset_raises(restore_rc):
    with pytest.raises(ValueError, match="paper"):
        set_style("poster")
    with pytest.raises(ValueError, match="paper"):
        get_preset("poster")
    assert get_preset() is PRESETS["paper"]


def test_fsize_helpers_follow_active_preset(restore_rc):
    set_style("talk")
    style.change_fsize(10)
    style.reset_fsize()
    assert mpl.rcParams["axes.titlesize"] == 18


# --- plotting code reads sizes from the active preset -----------------------

import matplotlib.pyplot as plt
import numpy as np

import geoutils.plotting as cplt


def test_text_helpers_follow_the_preset_outside_styled(restore_rc):
    set_style("talk")
    _, ax = plt.subplots()
    cplt.plot_text(ax, "t", check_axis=False)
    assert ax.texts[0].get_fontsize() == 16


def test_text_box_uses_preset_and_honours_fsize(restore_rc):
    set_style("talk")
    _, ax = plt.subplots()
    cplt.text_box(ax, "default")
    cplt.text_box(ax, "small", fsize=9)
    sizes = [a.txt._text.get_fontsize() for a in ax.artists]
    assert sizes == [16, 9]


@pytest.mark.parametrize("name, title, letter", [("paper", 9, 9), ("talk", 18, 18)])
def test_titles_and_panel_letters_follow_the_preset(name, title, letter):
    with style_context(name):
        res = cplt.plot_2d(y=np.arange(4.0), title="t")
        assert res["ax"].title.get_fontsize() == title
        panels = cplt.create_multi_plot(1, 2)["ax"]
        assert panels[0].texts[0].get_fontsize() == letter


def test_plot_2d_default_line_width_follows_preset_and_explicit_lw_wins():
    assert cplt.plot_2d(y=np.arange(4.0))["ax"].lines[0].get_linewidth() == 1.2
    with style_context("talk"):
        assert cplt.plot_2d(y=np.arange(4.0))["ax"].lines[0].get_linewidth() == 2.5
        assert cplt.plot_2d(y=np.arange(4.0), lw=3)["ax"].lines[0].get_linewidth() == 3


@pytest.mark.parametrize("name, length", [("paper", 3), ("talk", 6)])
def test_tick_length_follows_the_preset(name, length):
    with style_context(name):
        ax = cplt.plot_2d(y=np.arange(4.0))["ax"]
        assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == length


def test_map_gridline_labels_follow_the_preset():
    (grid,) = cplt.create_map(plot_grid=True)["ax"].artists
    assert grid.xlabel_style["size"] == 7 and grid.ylabel_style["size"] == 7


# --- default figure sizes ----------------------------------------------------

from geoutils.plotting.style import default_figsize


def test_default_figsize_single_panel():
    assert default_figsize() == pytest.approx((3.4, 3.4 * 0.75))
    with style_context("talk"):
        assert default_figsize() == pytest.approx((6.0, 4.5))


def test_default_figsize_caps_wide_grids():
    assert default_figsize(1, 6)[0] == pytest.approx(7.1)
    assert default_figsize(2, 2)[0] == pytest.approx(6.8)
    assert default_figsize(2, 2)[1] == pytest.approx(2 * 3.4 * 0.75)
    w, h = default_figsize(1, 6)
    assert h == pytest.approx(w / 6 * 0.75)
    assert default_figsize(aspect=1.0) == pytest.approx((3.4, 3.4))


def test_create_plot_uses_preset_size():
    assert tuple(cplt.create_plot()["fig"].get_size_inches()) == pytest.approx(default_figsize())
    with style_context("talk"):
        fig = cplt.create_plot()["fig"]
    assert tuple(fig.get_size_inches()) == pytest.approx((6.0, 4.5))


@pytest.mark.parametrize("make", [
    lambda **k: cplt.create_plot(**k)["fig"],
    lambda **k: cplt.create_multi_plot(1, 2, **k)["fig"],
    lambda **k: cplt.create_map(**k)["fig"],
    lambda **k: cplt.plot_2d(y=np.arange(4.0), **k)["fig"],
], ids=["create_plot", "create_multi_plot", "create_map", "plot_2d"])
def test_explicit_figsize_wins(make):
    assert tuple(make(figsize=(9, 5)).get_size_inches()) == pytest.approx((9, 5))


def test_every_default_figure_fits_a_double_column(latlon_da):
    figs = [
        cplt.plot_2d(y=np.arange(4.0))["fig"],
        cplt.plot_map(latlon_da, label="K")["fig"],
        cplt.create_map()["fig"],
        cplt.create_multi_plot(2, 3, projection="PlateCarree")["fig"],
        cplt.plot_hist(np.random.default_rng(0).normal(size=50))["fig"],
    ]
    assert all(f.get_size_inches()[0] <= 7.1 + 1e-9 for f in figs)
