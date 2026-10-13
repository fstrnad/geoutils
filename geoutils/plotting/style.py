"""The geoutils look: font sizes, palette and one explicit rc configuration.

Importing this module changes nothing. Plotting functions draw inside
``style_context()``, so the look is applied without touching the user's global
matplotlib state; call ``set_style()`` to adopt it for your own plots as well.
"""
import functools

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

# Okabe & Ito colour-blind safe categorical palette.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
colors = list(OKABE_ITO)  # kept for code that read ``style.colors``


def _font_family(preferred="Liberation Sans", fallback="DejaVu Sans"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return preferred if preferred in available else fallback


def _sizes(font, title, label, figure_title):
    return {
        "font.size": font,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": label,
        "ytick.labelsize": label,
        "legend.fontsize": label,
        "figure.titlesize": figure_title,
    }


STYLE = {
    "font.family": _font_family(),
    **_sizes(SMALL_SIZE, MEDIUM_SIZE, MEDIUM_SIZE, BIGGER_SIZE),
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
    plt.rcParams.update(_sizes(fsize, fsize + 4, fsize, fsize))


def reset_fsize():
    plt.rcParams.update(
        _sizes(SMALL_SIZE, MEDIUM_SIZE, MEDIUM_SIZE, BIGGER_SIZE))
