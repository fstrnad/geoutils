"""The geoutils look: font sizes, palette and one explicit rc configuration.

Importing this module changes nothing. Plotting functions draw inside
``style_context()``, so the look is applied without touching the user's global
matplotlib state; call ``set_style()`` to adopt it for your own plots as well.
"""
import contextlib
import functools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

TINY_SIZE = 8
MINI_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
MAXIMUM_SIZE = 22
MAX_ZORDER = 100

# The compact type scale of the geoutils look.
TICK_SIZE = 10
LABEL_SIZE = 11
TITLE_SIZE = 13
FIGURE_TITLE_SIZE = 14
INK = "#222222"  # text colour
EDGE = "#444444"  # axes frame colour

# Inter ships with the package (``fonts/``) and is registered on first use.
# Uses the first installed font, so the look degrades gracefully if the files
# cannot be loaded. Liberation Sans is the metric-compatible Linux twin of
# Arial; Helvetica Neue ranks below them because matplotlib only registers its
# regular weight (no real bold).
FONT_CHAIN = ["Inter", "Lato", "Arial", "Liberation Sans", "Helvetica Neue",
              "DejaVu Sans"]
_FONT_DIR = Path(__file__).parent / "fonts"
_fonts_registered = None  # None until the first attempt

# Okabe & Ito colour-blind safe categorical palette.
# Ordered so that the first few series are the easiest to tell apart on white.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
             "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
colors = list(OKABE_ITO)  # kept for code that read ``style.colors``


def _sizes(font, title, label, tick, figure_title):
    return {
        "font.size": font,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": tick,
        "figure.titlesize": figure_title,
    }


@dataclass(frozen=True)
class Preset:
    """Everything that scales between a paper figure and a slide."""
    tick: float
    label: float
    title: float
    figure_title: float
    line_width: float
    axes_width: float
    tick_length: float
    marker_size: float
    panel_width: float  # inches
    max_width: float  # inches; wide grids are capped here
    aspect: float  # panel height / width


PRESETS = {
    "paper": Preset(tick=7, label=8, title=9, figure_title=10,
                    line_width=1.2, axes_width=0.6, tick_length=3,
                    marker_size=3, panel_width=3.4, max_width=7.1,
                    aspect=0.75),
    "talk": Preset(tick=14, label=16, title=18, figure_title=20,
                   line_width=2.5, axes_width=1.2, tick_length=6,
                   marker_size=7, panel_width=6.0, max_width=13.0,
                   aspect=0.75),
}

_active = "paper"


def register_fonts():
    """Register the bundled Inter files once; False if they are unavailable."""
    global _fonts_registered
    if _fonts_registered is None:
        try:
            files = sorted(_FONT_DIR.glob("*.ttf"))
            for path in files:
                font_manager.fontManager.addfont(str(path))
            _fonts_registered = bool(files)
        except Exception:
            _fonts_registered = False
    return _fonts_registered


def _resolve(name):
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}.")
    return name


def get_preset(name=None):
    """The named preset, or the active one when ``name`` is None."""
    return PRESETS[_active if name is None else _resolve(name)]


def _font_family():
    """The first available font of the chain, pinned by name.

    A text artist keeps only the generic ``sans-serif`` and resolves it at
    draw time, usually after our style context has exited, which would give
    DejaVu. Naming the font keeps it for figures that are saved later.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    first = next((n for n in FONT_CHAIN if n in available), "DejaVu Sans")
    return list(dict.fromkeys([first, "DejaVu Sans"]))


def _preset_sizes(preset):
    return _sizes(preset.tick, preset.title, preset.label, preset.tick,
                  preset.figure_title)


def build_style(preset):
    """The rc dict of the geoutils look at the scale of ``preset``."""
    return {
        "font.family": _font_family(),
        "font.sans-serif": FONT_CHAIN,
        **_preset_sizes(preset),
        "axes.titleweight": "bold",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": EDGE,
        "axes.linewidth": preset.axes_width,
        "lines.linewidth": preset.line_width,
        "lines.markersize": preset.marker_size,
        "xtick.major.size": preset.tick_length,
        "ytick.major.size": preset.tick_length,
        "xtick.major.width": preset.axes_width,
        "ytick.major.width": preset.axes_width,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "grid.alpha": 0.3,
        "grid.color": "0.5",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pcolor.shading": "nearest",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # journals reject Type 3 fonts
        "ps.fonttype": 42,
    }


STYLE = build_style(PRESETS["paper"])  # kept for code that read ``style.STYLE``


def set_style(preset=None):
    """Apply the geoutils look to all subsequent matplotlib figures.

    ``preset`` is ``"paper"`` or ``"talk"``; None keeps the active one.
    """
    global _active
    _active = _active if preset is None else _resolve(preset)
    register_fonts()
    plt.rcParams.update(build_style(PRESETS[_active]))


def default_figsize(nrows=1, ncols=1, aspect=None):
    """Figure size (inches) for a grid of panels at the active preset.

    Wide grids are capped at the preset's ``max_width``; ``aspect`` is the
    height/width of one panel and defaults to the preset's.
    """
    p = get_preset()
    width = min(ncols * p.panel_width, p.max_width)
    return width, nrows * width / ncols * (p.aspect if aspect is None else aspect)


@contextlib.contextmanager
def style_context(preset=None):
    """Apply the geoutils look only inside a ``with`` block.

    ``preset`` is ``"paper"`` or ``"talk"``; None uses the active one.
    """
    global _active
    name = _active if preset is None else _resolve(preset)
    previous, _active = _active, name
    register_fonts()
    try:
        with plt.rc_context(build_style(PRESETS[name])):
            yield
    finally:
        _active = previous


def styled(func):
    """Run a plotting function inside ``style_context()``."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with style_context():
            return func(*args, **kwargs)
    return wrapper


def list_available_fonts():
    import matplotlib.font_manager
    all_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    print(all_fonts)
    return all_fonts


def list_available_styles():
    print(plt.style.available)
    return plt.style.available


def change_fsize(fsize):
    plt.rcParams.update(_sizes(fsize, fsize + 4, fsize, fsize, fsize))


def reset_fsize():
    plt.rcParams.update(_preset_sizes(get_preset()))
