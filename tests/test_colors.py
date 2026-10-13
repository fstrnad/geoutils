"""Colour-scale resolution: bounds, levels, rounding and colorbar extension."""
import numpy as np
import pytest

from geoutils.plotting.colors import ColorScale, resolve_color_scale
from geoutils.plotting.maps.fields import plot_array

Z = np.arange(100.0).reshape(10, 10)


# --- characterization of plot_array (values captured before the refactor) ---

@pytest.mark.parametrize("kwargs, z, ticks, extend", [
    (dict(color_bounds="quantile"), Z,
     [9.9, 19.8, 29.7, 39.6, 49.5, 59.4, 69.3, 79.2, 89.1], "both"),
    (dict(color_bounds="minmax"), Z,
     [0.0, 12.4, 24.8, 37.1, 49.5, 61.9, 74.2, 86.6, 99.0], "neither"),
    (dict(color_bounds="absmax"), Z - 50,
     [-50.0, -38.0, -25.0, -12.0, 0.0, 12.0, 25.0, 38.0, 50.0], "neither"),
    (dict(color_bounds="absquantile"), Z - 50,
     [-45.0, -33.8, -22.5, -11.2, 0.0, 11.2, 22.5, 33.8, 45.0], "both"),
    (dict(vmin=20, vmax=60), Z,
     [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0], "both"),
    (dict(vmin=20, vmax=60, extend="neither"), Z,
     [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0], "neither"),
    (dict(vmin=20, vmax=200, extend="min"), Z,
     [20.0, 42.0, 65.0, 88.0, 110.0, 132.0, 155.0, 178.0, 200.0], "min"),
    (dict(), np.full((10, 10), 3.0),
     [2.9, 2.92, 2.95, 2.98, 3.0, 3.02, 3.05, 3.08, 3.1], "neither"),
    (dict(color_bounds="minmax", levels=4), Z,
     [0.0, 24.8, 49.5, 74.2, 99.0], "neither"),
], ids=["quantile", "minmax", "absmax", "absquantile", "explicit-both",
        "explicit-neither", "explicit-min", "constant", "levels-4"])
def test_plot_array_color_scale_unchanged(kwargs, z, ticks, extend):
    res = plot_array(z, plot_type="contourf", **kwargs)
    np.testing.assert_allclose(res["ticks"], ticks, atol=1e-6)
    assert res["extend"] == extend


# --- the resolver itself ------------------------------------------------

def test_quantile_bounds():
    s = resolve_color_scale(Z, color_bounds="quantile")
    assert isinstance(s, ColorScale)
    assert (s.vmin, s.vmax) == pytest.approx((9.9, 89.1))


def test_minmax_bounds():
    s = resolve_color_scale(Z, color_bounds="minmax")
    assert (s.vmin, s.vmax) == (0.0, 99.0)


def test_absmax_symmetric():
    s = resolve_color_scale(Z - 50, color_bounds="absmax")
    assert (s.vmin, s.vmax) == (-50.0, 50.0)


def test_extend_both_when_outliers():
    assert resolve_color_scale(Z, color_bounds="quantile").extend == "both"


def test_extend_neither_when_forced():
    assert resolve_color_scale(Z, vmin=20, vmax=60, extend="neither").extend == "neither"


def test_extend_user_value_kept():
    assert resolve_color_scale(Z, vmin=20, vmax=200, extend="min").extend == "min"


def test_levels_have_no_duplicates():
    s = resolve_color_scale(Z * 1e-6, color_bounds="minmax", levels=8)
    assert len(set(s.levels)) == len(s.levels) == 9


def test_constant_array_widened():
    s = resolve_color_scale(np.full((4, 4), 3.0))
    assert s.vmin < 3.0 < s.vmax


def test_sci_sign_follows_magnitude():
    small = resolve_color_scale(Z * 1e-4, color_bounds="minmax", levels=8)
    large = resolve_color_scale(Z * 1e4, color_bounds="minmax", levels=8)
    assert small.sci < 0 < large.sci


def test_no_data_gives_empty_scale():
    s = resolve_color_scale(None)
    assert (s.vmin, s.vmax, s.levels, s.extend) == (None, None, None, "neither")


@pytest.mark.parametrize("bad", ["0", [1], None.__class__])
def test_non_numeric_limits_rejected(bad):
    with pytest.raises(ValueError, match="vmin"):
        resolve_color_scale(Z, vmin=bad)


# --- ColorbarSpec -------------------------------------------------------

def _mappable():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(16.0).reshape(4, 4))
    return fig, ax, im, mpl


def test_colorbar_spec_matches_flat_keywords():
    from geoutils.plotting.colors import ColorbarSpec, make_colorbar
    fig, ax, im, _ = _mappable()
    a = make_colorbar(ax, im, fig=fig, spec=ColorbarSpec(
        label="K", ticks=np.arange(0.0, 16.0, 2.0), tick_step=2, set_cax=True))
    fig, ax, im, _ = _mappable()
    b = make_colorbar(ax, im, fig=fig, label="K", ticks=np.arange(0.0, 16.0, 2.0),
                      tick_step=2, set_cax=True)
    assert a.ax.get_xlabel() == b.ax.get_xlabel() == "K"
    np.testing.assert_allclose(a.get_ticks(), b.get_ticks())
    np.testing.assert_allclose(a.get_ticks(), [0, 4, 8, 12])


def test_colorbar_spec_from_kwargs_only_pops_known_keys():
    from geoutils.plotting.colors import ColorbarSpec
    kwargs = dict(label="K", orientation="vertical", color="red")
    spec = ColorbarSpec.from_kwargs(kwargs)
    assert (spec.label, spec.orientation) == ("K", "vertical")
    assert kwargs == {"color": "red"}


def test_colorbar_unknown_orientation_raises():
    from geoutils.plotting.colors import ColorbarSpec, make_colorbar
    fig, ax, im, _ = _mappable()
    with pytest.raises(ValueError, match="orientation"):
        make_colorbar(ax, im, fig=fig,
                      spec=ColorbarSpec(orientation="diagonal", set_cax=True))


# --- robust default and diverging colormaps -----------------------------

def test_default_is_robust_and_extends_over_outliers():
    z = np.append(np.linspace(0, 1, 999), 1000.0)
    s = resolve_color_scale(z)
    assert s.vmax < 10 and s.extend in ("max", "both")
    assert (s.vmin, s.vmax) == pytest.approx(
        (np.quantile(z, 0.02), np.quantile(z, 0.98)))


def test_quantile_still_available():
    s = resolve_color_scale(Z, color_bounds="quantile")
    assert (s.vmin, s.vmax) == pytest.approx((9.9, 89.1))


def test_diverging_cmap_symmetric_by_default():
    z = np.append(np.linspace(-1, 4, 500), np.linspace(-1, 1, 500))
    s = resolve_color_scale(z, diverging=True)
    assert s.vmin == -s.vmax and s.vmax > 0


def test_sequential_cmap_not_symmetric():
    z = np.linspace(1, 5, 200)
    s = resolve_color_scale(z, diverging=False)
    assert s.vmin >= 1


def test_explicit_limits_win_over_symmetry():
    s = resolve_color_scale(Z, vmin=-1, vmax=3, diverging=True)
    assert (s.vmin, s.vmax) == (-1, 3)


def test_one_explicit_limit_is_kept():
    s = resolve_color_scale(Z - 50, vmin=-10, diverging=True)
    assert s.vmin == -10


@pytest.mark.parametrize("name, expected", [
    ("RdBu_r", True), ("coolwarm", True), ("viridis", False), ("RdYlBu", True),
])
def test_is_diverging_by_name(name, expected):
    import matplotlib as mpl
    from geoutils.plotting.colors import is_diverging
    assert is_diverging(name) is expected
    assert is_diverging(mpl.colormaps[name]) is expected


def test_is_diverging_handles_missing_cmap():
    from geoutils.plotting.colors import is_diverging
    assert is_diverging(None) is False


def test_plot_array_diverging_cmap_gets_symmetric_ticks():
    z = np.outer(np.linspace(-1, 5, 30), np.ones(30))
    res = plot_array(z, plot_type="contourf", cmap="RdBu_r")
    assert res["ticks"][0] == pytest.approx(-res["ticks"][-1])


def test_plot_array_default_clips_at_robust_percentiles():
    res = plot_array(Z, plot_type="contourf")
    assert res["ticks"][0] == pytest.approx(1.98, abs=0.1)
    assert res["ticks"][-1] == pytest.approx(97.02, abs=0.1)
