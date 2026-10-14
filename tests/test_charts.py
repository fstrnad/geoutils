"""Tests for the chart primitives: bars, box plots, hexbin, heatmaps, histograms."""
import numpy as np
import pytest

import geoutils.plotting as gp


def test_plot_bar_single_series():
    res = gp.plot_bar([1.0, 2.0, 3.0], labels=["a", "b", "c"])
    assert len(res["ax"].patches) == 3
    assert [t.get_text() for t in res["ax"].get_xticklabels()] == ["a", "b", "c"]


def test_plot_bar_grouped_splits_the_slot():
    res = gp.plot_bar([[1, 2], [3, 4]], labels=["x", "y"],
                      series_labels=["m1", "m2"])
    widths = {round(p.get_width(), 6) for p in res["ax"].patches}
    assert widths == {0.4}
    assert [t.get_text() for t in res["ax"].get_legend().get_texts()] == ["m1", "m2"]


def test_plot_bar_horizontal_with_value_labels():
    res = gp.plot_bar([1.0, 2.0], labels=["a", "b"], horizontal=True,
                      value_fmt="%.1f")
    assert res["ax"].patches[0].get_width() == 1.0
    assert [t.get_text() for t in res["ax"].texts] == ["1.0", "2.0"]


def test_plot_bar_numeric_positions_keep_width():
    res = gp.plot_bar([1.0, 2.0], x=[10.0, 12.0], width=1.5)
    assert [p.get_x() + p.get_width() / 2 for p in res["ax"].patches] == [10.0, 12.0]


def test_plot_bar_per_bar_colors_and_hatch():
    res = gp.plot_bar([1.0, 2.0], colors=["red", "blue"], hatch=["//"])
    assert res["ax"].patches[0].get_hatch() == "//"
    assert res["ax"].patches[0].get_facecolor()[:3] == (1.0, 0.0, 0.0)


def test_plot_box_draws_one_box_per_group():
    rng = np.random.default_rng(0)
    res = gp.plot_box([rng.normal(size=50) for _ in range(3)],
                      positions=[1, 2, 3])
    assert len(res["ax"].lines) > 0
    assert len(res["im"]["boxes"]) == 3


def test_plot_hexbin_adds_colorbar_when_labelled():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=500), rng.normal(size=500)
    res = gp.plot_hexbin(x, y, gridsize=10, log=True, label="count")
    assert res["colorbar"] is not None
    assert res["im"].get_array().size > 0


def test_plot_heatmap_annotations_and_ticks():
    res = gp.plot_heatmap(np.array([[0.0, 1.0], [0.5, 0.25]]),
                          xticklabels=["a", "b"], yticklabels=["c", "d"],
                          annotate=[["w", "x"], ["y", "z"]], label="share")
    assert sorted(t.get_text() for t in res["ax"].texts) == ["w", "x", "y", "z"]
    assert res["colorbar"] is not None


def test_plot_histogram_step_density_integrates_to_one():
    data = np.random.default_rng(0).uniform(size=1000)
    res = gp.plot_histogram([data], bins=np.linspace(0, 1, 11), density=True,
                            histtype="step", labels=["u"])
    heights = res["ax"].patches[0].get_xy()[:, 1]
    assert heights.max() == pytest.approx(1.0, abs=0.3)
    assert res["ax"].get_legend() is not None


def test_plot_histogram_bar_counts_sum_to_n():
    data = np.arange(100.0)
    res = gp.plot_histogram(data, bins=10)
    assert sum(p.get_height() for p in res["ax"].patches) == 100
