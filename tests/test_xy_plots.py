"""Smoke test for the x-y plotting entry point, plot_2d."""
import numpy as np

import geoutils.plotting.xy as xyp


def test_plot_2d_returns_plotresult():
    res = xyp.plot_2d(y=np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    for key in ("ax", "fig"):
        assert key in res
        assert res[key] is not None
