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
