"""Typed container for the artefacts a geoutils plot returns.

Historically the plotting functions returned plain dicts such as
``{"ax": ax, "fig": fig, "im": im}``. ``PlotResult`` keeps that exact access
pattern working (``res["ax"]``, ``"im" in res``) so existing code and the
tutorials are unaffected, while adding typed attributes that editors can
autocomplete and document.
"""
from dataclasses import dataclass, fields


@dataclass
class PlotResult:
    """Result of a geoutils plotting call.

    Attributes
    ----------
    fig : matplotlib Figure
    ax : matplotlib Axes (or an array of Axes for multi-panel figures)
    im : the mappable / artist returned by the underlying draw call
    projection : cartopy CRS the data was drawn in, if any
    ticks : colorbar tick levels, if they were computed
    extend : colorbar ``extend`` setting, if it was computed
    colorbar : the Colorbar, if one was drawn
    """

    fig: object = None
    ax: object = None
    im: object = None
    projection: object = None
    ticks: object = None
    extend: object = None
    colorbar: object = None

    @property
    def mappable(self):
        """Readable alias for :attr:`im`."""
        return self.im

    # -- dict compatibility: legacy callers use res["ax"], "im" in res, etc. --
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key, value):
        if key not in self._field_names():
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self._field_names() and getattr(self, key) is not None

    def get(self, key, default=None):
        return getattr(self, key) if key in self else default

    def keys(self):
        return [name for name in self._field_names()
                if getattr(self, name) is not None]

    @staticmethod
    def _field_names():
        return tuple(f.name for f in fields(PlotResult))
