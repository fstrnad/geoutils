"""The geoutils look is explicit: importing or plotting never leaks rc changes."""
import json
import subprocess
import sys

import matplotlib as mpl
import numpy as np
import pytest

import geoutils.plotting as cplt
import geoutils.plotting.style as style


def test_import_does_not_change_rcparams():
    code = (
        "import json, matplotlib\n"
        "import matplotlib.pyplot\n"
        "before = {k: repr(v) for k, v in matplotlib.rcParams.items()}\n"
        "import geoutils.plotting\n"
        "after = {k: repr(v) for k, v in matplotlib.rcParams.items()}\n"
        "print(json.dumps([k for k in before if before[k] != after[k]]))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-W", "ignore", "-c", code],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.strip().splitlines()[-1]) == []


def _snapshot():
    return {k: repr(v) for k, v in mpl.rcParams.items()}


@pytest.mark.parametrize("call", [
    lambda da: cplt.plot_2d(y=np.arange(5.0)),
    lambda da: cplt.plot_map(da, label="K"),
    lambda da: cplt.create_plot(2, 2),
    lambda da: cplt.create_multi_plot(1, 2, projection="PlateCarree"),
], ids=["plot_2d", "plot_map", "create_plot", "create_multi_plot"])
def test_plot_does_not_leak_rcparams(latlon_da, call):
    before = _snapshot()
    call(latlon_da)
    assert _snapshot() == before


def test_set_style_applies_palette_and_sizes():
    before = mpl.rcParams.copy()
    try:
        style.set_style()
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0] == "#E69F00"
        assert mpl.rcParams["savefig.dpi"] == 300
    finally:
        mpl.rcParams.update(before)


def test_plot_2d_cycles_palette_for_many_series():
    n = len(style.OKABE_ITO) + 2
    res = cplt.plot_2d(y=[np.arange(4.0) + i for i in range(n)])
    colors = [mpl.colors.to_hex(line.get_color()).upper() for line in res["ax"].lines]
    assert colors[:8] == [c.upper() for c in style.OKABE_ITO]
    assert colors[8] == colors[0]


def test_plot_into_user_axis_leaves_user_figure_alone():
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2)
    cplt.plot_2d(y=np.arange(4.0), ax=a)
    assert len(b.lines) == 0 and len(a.lines) == 1 and len(fig.axes) == 2


def test_fsize_helpers_are_consistent():
    before = mpl.rcParams.copy()
    try:
        style.change_fsize(10)
        assert mpl.rcParams["font.size"] == 10
        style.reset_fsize()
        assert mpl.rcParams["axes.titlesize"] == style.MEDIUM_SIZE
        assert mpl.rcParams["figure.titlesize"] == style.BIGGER_SIZE
    finally:
        mpl.rcParams.update(before)
