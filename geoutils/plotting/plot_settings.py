from matplotlib import rcParams, font_manager
import matplotlib.pyplot as plt
import scienceplots
import warnings
warnings.filterwarnings("ignore", message="facecolor will have no effect")
TINY_SIZE = 8
MINI_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
MAXIMUM_SIZE = 22
MAX_ZORDER = 100

set_new_font = True
enable_all_cmaps = True

plt.rcdefaults()

default_font = 'Liberation Sans'
preferred_font = 'Lato'
preferred_font = 'Liberation Sans'


def set_font_or_fallback(preferred: str = preferred_font,
                         fallback: str = default_font):
    """Set Matplotlib font to `preferred` if available, else to `fallback`."""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if preferred in available_fonts:
        rcParams["font.family"] = preferred
    elif fallback in available_fonts:
        rcParams["font.family"] = fallback
    else:
        # fallback to default (DejaVu Sans) if neither is found
        rcParams["font.family"] = "DejaVu Sans"


# Choose 'serif', 'sans-serif', or 'monospace'
plt.style.use('bmh')
# plt.style.use(['science'])
if set_new_font:
    set_font_or_fallback()
    plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['axes.facecolor'] = 'white'
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
# fontsize of the x and y labels
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# plt.rcParams['pcolor.shading'] ='nearest' # For pcolormesh
colors = [
    'dimgray',
    'firebrick',
    "steelblue",
    "tab:green",
    "tomato",
    "c",
    "tab:orange",
    "y",
    "tab:blue",
    "darkviolet",
    "tab:purple",
    "m",
    "slategray",
]


def list_available_fonts():
    import matplotlib.font_manager
    all_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    print(all_fonts)
    return all_fonts


def list_available_styles():
    print(plt.style.available)
    return plt.style.available


def change_fsize(fsize):
    plt.rc("font", size=fsize)  # controls default text sizes
    plt.rc("axes", titlesize=fsize+4)  # fontsize of the axes title
    # fontsize of the x and y labels
    plt.rc("axes", labelsize=fsize)
    plt.rc("xtick", labelsize=fsize)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=fsize)  # fontsize of the tick labels
    plt.rc("legend", fontsize=fsize)  # legend fontsize
    plt.rc("figure", titlesize=fsize)  # fontsize of the figure title


def reset_fsize():
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # fontsize of the x and y labels
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
