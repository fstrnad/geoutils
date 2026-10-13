"""The public plotting API and the deprecated ``plots`` shim keep resolving."""
import importlib

import pytest

import geoutils.plotting as plotting


def test_all_public_names_resolve():
    missing = [n for n in plotting.__all__ if not hasattr(plotting, n)]
    assert missing == []


def test_plots_shim_exposes_public_api():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        import geoutils.plotting.plots as cplt
        importlib.reload(cplt)
    missing = [n for n in plotting.__all__ if not hasattr(cplt, n)]
    assert missing == []
