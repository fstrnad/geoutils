"""Locks the dict-compatibility bridge that keeps legacy callers working."""
from geoutils.plotting.plot_result import PlotResult


def test_attribute_and_item_access_agree():
    r = PlotResult(ax="A", fig="F", im="I")
    assert r.ax == "A" == r["ax"]
    assert r.mappable == "I" == r["im"]


def test_membership_reflects_set_fields():
    r = PlotResult(ax="A", fig="F")
    assert "ax" in r and "fig" in r
    assert "im" not in r            # unset -> not a member
    assert r["im"] is None          # ...but attribute access still returns None
    assert set(r.keys()) == {"ax", "fig"}


def test_get_and_unknown_key():
    r = PlotResult(ax="A")
    assert r.get("ax") == "A"
    assert r.get("im", 7) == 7
    try:
        r["does_not_exist"]
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown key")
