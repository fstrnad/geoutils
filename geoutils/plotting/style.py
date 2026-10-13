"""The geoutils look: font sizes, palette and one explicit rc configuration.

Importing this module changes nothing. Plotting functions draw inside
``style_context()``, so the look is applied without touching the user's global
matplotlib state; call ``set_style()`` to adopt it for your own plots as well.
"""
import functools

import matplotlib.pyplot as plt
from cycler import cycler

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

# Uses the first installed font, so it also works where Lato is missing (e.g.
# on a cluster) without bundling any font files. Liberation Sans is the
# metric-compatible Linux twin of Arial; Helvetica Neue ranks below them
# because matplotlib only registers its regular weight (no real bold).
FONT_CHAIN = ["Lato", "Arial", "Liberation Sans", "Helvetica Neue", "DejaVu Sans"]

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


_BASE_SIZES = _sizes(TICK_SIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
                     FIGURE_TITLE_SIZE)


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_CHAIN,
    **_BASE_SIZES,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": EDGE,
    "axes.linewidth": 0.8,
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
}


def set_style():
    """Apply the geoutils look to all subsequent matplotlib figures."""
    plt.rcParams.update(STYLE)


def style_context():
    """Context manager applying the geoutils look only inside its block."""
    return plt.rc_context(STYLE)


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
    plt.rcParams.update(_BASE_SIZES)
