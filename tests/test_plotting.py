"""Smoke test for the central plotting entry point, plot_map."""
import geoutils.plotting.map_plots as mp


def test_plot_map_returns_expected_artifacts(latlon_da):
    res = mp.plot_map(latlon_da)
    assert isinstance(res, dict)
    for key in ("ax", "fig", "im"):
        assert key in res
