"""Smoke test for the x-y plotting entry point, plot_2d."""
import numpy as np
import pytest

import geoutils.plotting.xy as xyp


def test_plot_2d_returns_plotresult():
    res = xyp.plot_2d(y=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    for key in ("ax", "fig"):
        assert key in res
        assert res[key] is not None


def test_pick_falls_back_to_last():
    assert xyp._pick([1, 2], 0) == 1
    assert xyp._pick([1, 2], 5) == 2


def test_plot_2d_per_series_styles():
    ys = [np.arange(5.0) + i for i in range(3)]
    res = xyp.plot_2d(y=ys, lw_arr=[1, 2])
    assert [line.get_linewidth() for line in res["ax"].lines] == [1, 2, 2]


def test_plot_2d_error_band_from_y_err():
    y = np.arange(5.0)
    res = xyp.plot_2d(y=y, y_err=np.ones(5))
    assert len(res["ax"].collections) == 1


def test_plot_2d_error_band_from_bounds():
    y = np.arange(5.0)
    res = xyp.plot_2d(y=[y, y + 1], y_lb=[y - 1, y], y_ub=[y + 1, y + 2])
    assert len(res["ax"].collections) == 2


def test_plot_2d_labels_become_legend_entries():
    ys = [np.arange(5.0), np.arange(5.0) + 1]
    res = xyp.plot_2d(y=ys, label=["a", "b"])
    assert [t.get_text() for t in res["ax"].get_legend().get_texts()] == ["a", "b"]


def test_plot_2d_standardize_centers_series():
    res = xyp.plot_2d(y=np.arange(10.0), standardize=True)
    assert abs(np.mean(res["ax"].lines[0].get_ydata())) < 1e-9


def test_plot_2d_linearize_xaxis():
    res = xyp.plot_2d(y=np.arange(4.0), x=np.array([10.0, 20.0, 30.0, 40.0]),
                      linearize_xaxis=True)
    assert list(res["ax"].lines[0].get_xdata()) == [0, 1, 2, 3]


def test_plot_2d_bar_runs():
    res = xyp.plot_2d(y=[np.array([1.0, 2.0, 3.0])], x=[np.array([0, 1, 2])],
                      plot_type="bar")
    assert res["ax"].patches


def test_plot_2d_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        xyp.plot_2d(y=np.arange(4.0), x=np.arange(3.0))
