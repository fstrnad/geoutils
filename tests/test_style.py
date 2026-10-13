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
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0] == style.OKABE_ITO[0]
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
        assert mpl.rcParams["axes.titlesize"] == style.STYLE["axes.titlesize"]
        assert mpl.rcParams["figure.titlesize"] == style.STYLE["figure.titlesize"]
    finally:
        mpl.rcParams.update(before)


def test_palette_leads_with_high_contrast_colors():
    assert style.OKABE_ITO[:3] == ["#0072B2", "#D55E00", "#009E73"]
    assert style.OKABE_ITO[-2:] == ["#F0E442", "#000000"]
    assert sorted(style.OKABE_ITO) == sorted(
        ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
         "#0072B2", "#D55E00", "#CC79A7", "#000000"])


def test_plot_2d_default_line_width_is_readable():
    res = cplt.plot_2d(y=np.arange(4.0))
    assert res["ax"].lines[0].get_linewidth() == 1.5


def test_legend_avoids_data_by_default():
    res = cplt.plot_2d(y=[np.arange(4.0), np.arange(4.0) + 1], label=["a", "b"])
    assert res["ax"].get_legend()._loc == 0  # matplotlib's 'best'


def test_font_chain_prefers_designed_fonts_and_ends_in_dejavu():
    chain = style.STYLE["font.sans-serif"]
    assert style.STYLE["font.family"] == "sans-serif"
    assert chain[:3] == ["Lato", "Arial", "Liberation Sans"] and chain[-1] == "DejaVu Sans"


def test_type_scale_is_compact():
    s = style.STYLE
    assert (s["xtick.labelsize"], s["axes.labelsize"], s["axes.titlesize"]) == (10, 11, 13)
    assert s["axes.titleweight"] == "bold" and s["legend.fontsize"] == 10


def test_plot_text_uses_dark_grey_not_black():
    res = cplt.plot_2d(y=np.arange(4.0), title="t", xlabel="x")
    assert mpl.colors.to_hex(res["ax"].xaxis.label.get_color()) == "#222222"


def test_map_annotations_use_the_label_size():
    ax = cplt.create_map()["ax"]
    cplt.plot_rectangle(ax, lon_range=(0, 10), lat_range=(0, 10), text="roi")
    assert [t.get_fontsize() for t in ax.texts] == [style.LABEL_SIZE]
