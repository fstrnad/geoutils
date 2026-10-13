"""Colormaps, norms, colour scales and colorbars."""
import numbers
import warnings
from dataclasses import dataclass, field, fields
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import copy
import numpy as np
import geoutils.utils.general_utils as gut
import geoutils.plotting.style as pst




def get_available_mpl_colormaps():
    """
    Get a list of all available colormaps in matplotlib.

    Returns:
        list: A list of strings containing the names of available colormaps.
    """
    gut.load_package_avail('cmocean')
    gut.load_package_avail('cmweather')
    colormaps = plt.colormaps()
    return np.array(colormaps)


def get_available_palettable_colormaps():
    """
    Get a list of all available colormaps in palettable.

    Returns:
        list: A list of strings containing the names of available colormaps.
    """
    import palettable as pt

    diverging_lst = [cmap for cmap in dir(
        pt.colorbrewer.diverging) if not cmap.startswith("__")]
    qualitative_lst = [cmap for cmap in dir(
        pt.colorbrewer.qualitative) if not cmap.startswith("__")]
    sequential_lst = [cmap for cmap in dir(
        pt.colorbrewer.sequential) if not cmap.startswith("__")]
    cmocean_div = [cmap for cmap in dir(
        pt.cmocean.diverging) if not cmap.startswith("__")]
    cmocean_seq = [cmap for cmap in dir(
        pt.cmocean.sequential) if not cmap.startswith("__")]
    scientific_seq = [cmap for cmap in dir(
        pt.scientific.sequential) if not cmap.startswith("__")]
    scientific_div = [cmap for cmap in dir(
        pt.scientific.diverging) if not cmap.startswith("__")]

    all_maps = sorted(plt.colormaps() + diverging_lst + qualitative_lst +
                      sequential_lst + cmocean_div + cmocean_seq +
                      scientific_seq + scientific_div)

    return (all_maps,
            diverging_lst,
            qualitative_lst,
            sequential_lst,
            cmocean_div,
            cmocean_seq,
            scientific_seq,
            scientific_div)


def get_cmap(cmap, levels=None):
    mpl_cmaps = get_available_mpl_colormaps()
    if cmap in mpl_cmaps:
        colormap = cmap
    else:
        import palettable as pt
        pt_cmaps, d_cmaps, q_cmaps, s_cmaps, cmocean_div, cmocean_seq, scientific_seq, scientific_div = get_available_palettable_colormaps()
        cmap_strs = cmap.split("_")
        reverse = True if 'r' in cmap_strs else False

        if cmap in pt_cmaps:
            if cmap in d_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'diverging',  number=int(cmap_strs[1]),
                    reverse=reverse)
            elif cmap in s_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'sequential',  number=int(cmap_strs[1]), reverse=reverse)
            elif cmap in q_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'qualitative',  number=int(cmap_strs[1]), reverse=reverse)
            elif cmap in cmocean_div:
                colormap = pt.cmocean.diverging.get_map(
                    name=cmap, reverse=reverse)
            elif cmap in cmocean_seq:
                colormap = pt.cmocean.sequential.get_map(
                    cmap, reverse=reverse)
            elif cmap in scientific_seq:
                colormap = pt.scientific.sequential.get_map(
                    cmap, reverse=reverse)
            elif cmap in scientific_div:
                colormap = pt.scientific.diverging.get_map(
                    cmap, reverse=reverse)
            else:
                raise ValueError(f'Colormap {cmap} not found in palettable!')
        else:
            raise ValueError(
                f'Special colormap {cmap} not found. Please choose from {pt_cmaps}')
        colormap = colormap.mpl_colormap

    n_colors = len(levels) if levels is not None else None
    cmap = plt.get_cmap(colormap, n_colors)

    return cmap


def create_cmap(cmap, levels=None, **kwargs):
    # set colormap
    if isinstance(cmap, str):
        cmap = get_cmap(cmap, levels=levels)
    elif not isinstance(cmap, mpl.colors.Colormap):
        raise ValueError(
            f'cmap has to be of type str or mpl.colors.Colormap but is of type {type(cmap)}!')
    if levels is not None:
        n_colors = len(levels)
        # Set center of colormap to specific color
        centercolor = kwargs.pop('centercolor', None)
        leftcolor = kwargs.pop('leftcolor', None)
        if centercolor is not None:
            colors = np.array([mpl.colors.rgb2hex(cmap(i))
                               for i in range(n_colors)])

            centercolor = '#FFFFFF' if centercolor == 'white' else centercolor
            idx = [len(colors) // 2 - 1, len(colors) // 2]
            colors[idx] = centercolor
            cmap = mpl.colors.ListedColormap(colors)
        if leftcolor is not None:
            colors = np.array([mpl.colors.rgb2hex(cmap(i))
                               for i in range(n_colors)])
            leftcolor = '#FFFFFF' if leftcolor == 'white' else leftcolor
            colors[0] = leftcolor
            cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(
            levels, ncolors=cmap.N, clip=True)
    else:
        norm = None

    return cmap, norm


def make_colorbar_discrete(ax, im, fig=None, vmin=None, vmax=None, **kwargs):
    if vmin is None:
        vmin = im.get_clim()[0]
    if vmax is None:
        vmax = im.get_clim()[1]
    m = ScalarMappable(cmap=im.get_cmap())
    m.set_array(im.get_array())
    m.set_clim(im.get_clim())
    step = im.levels[1] - im.levels[0]
    num_levels = len(im.levels)

    noextend = "extend" in kwargs.keys() and kwargs["extend"] == "neither"
    # set the colorbar boundaries
    cliplower = im.zmin < vmin if not noextend else False
    clipupper = im.zmax > vmax if not noextend else False
    if "extend" in kwargs.keys() and kwargs["extend"] == "both":
        cliplower = True
        clipupper = True

    ticks = kwargs.pop('ticks', None)
    if ticks is None:
        boundaries = np.linspace(
            vmin - (cliplower and noextend) * step,
            vmax + (clipupper and noextend) * (step + 1),
            num_levels,
            endpoint=True,
        )
    else:
        boundaries = ticks
    boundaries_map = copy.deepcopy(boundaries)
    if cliplower:
        boundaries_map = np.insert(
            boundaries_map, 0, vmin - abs(vmin / 100), axis=0)
    if clipupper:
        boundaries_map = np.insert(
            boundaries_map, len(boundaries_map), vmax + abs(vmax / 100), axis=0
        )

    kwargs["boundaries"] = boundaries_map

    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not ("extend" in kwargs.keys()) or kwargs["extend"] in ["min", "max"]:
        extend_min = cliplower or (
            "extend" in kwargs.keys() and kwargs["extend"] == "min"
        )
        extend_max = clipupper or (
            "extend" in kwargs.keys() and kwargs["extend"] == "max"
        )
        if extend_min and extend_max:
            kwargs["extend"] = "both"
        elif extend_min:
            kwargs["extend"] = "min"
        elif extend_max:
            kwargs["extend"] = "max"

    cbar = make_colorbar(ax=ax, im=m, fig=fig, vmin=vmin, vmax=vmax,

                         **kwargs)

    return cbar


def cbarfmt(x, pos):
    a, b = f'{x:.2e}'.split('e')
    b = int(b)
    return rf'${a} \times 10^{{{b}}}$'


DIVERGING = frozenset({
    "RdBu", "RdYlBu", "RdYlGn", "coolwarm", "bwr", "seismic", "BrBG", "PiYG",
    "PRGn", "PuOr", "RdGy", "Spectral",
})


def is_diverging(cmap):
    """True for a (name or Colormap of a) known diverging colormap."""
    name = cmap if isinstance(cmap, str) else getattr(cmap, "name", "")
    return name.removesuffix("_r") in DIVERGING


@dataclass(frozen=True)
class ColorScale:
    """Resolved colour limits, contour levels and colorbar extension."""
    vmin: float | None
    vmax: float | None
    levels: np.ndarray | None
    extend: str
    sci: int | None
    round_dec: int | None


def _color_bounds(z, vmin, vmax, color_bounds, diverging=False):
    if color_bounds == 'robust':
        symmetric = diverging and vmin is None and vmax is None
        vmin = np.nanquantile(z, q=0.02) if vmin is None else vmin
        vmax = np.nanquantile(z, q=0.98) if vmax is None else vmax
        if symmetric:
            vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
    elif color_bounds == 'quantile':
        vmin = np.nanquantile(z, q=0.1) if vmin is None else vmin
        vmax = np.nanquantile(z, q=0.9) if vmax is None else vmax
    elif color_bounds == 'minmax':
        vmin = np.nanmin(z) if vmin is None else vmin
        vmax = np.nanmax(z) if vmax is None else vmax
    elif color_bounds == 'absmax':
        vmin = -1 * np.nanmax(abs(z)) if vmin is None else vmin
        vmax = np.nanmax(abs(z)) if vmax is None else vmax
    elif color_bounds == 'absquantile':
        vmax = np.nanquantile(abs(z), q=0.9) if vmax is None else vmax
        vmin = -1 * vmax
    else:
        raise ValueError(f'Unknown color_bounds {color_bounds!r}!')
    return vmin, vmax


def _sci_and_rounding(vmin, vmax, sci, round_dec):
    expo = gut.get_exponent10(np.abs(vmax - vmin))
    if sci is None:
        sci = expo if np.abs(expo) > 1 else None
    if sci is not None:
        if sci < 0:
            round_dec = abs(sci) + 1
        elif sci > 0:
            round_dec = -1 * (sci - 2)
    elif round_dec is None:
        round_dec = 1 if vmax > 1 else 2
    return sci, round_dec


def _round_levels(levels, round_dec):
    """Round levels, adding decimals until no two levels coincide."""
    if levels is None or round_dec is None:
        return levels
    rounded = np.around(levels, round_dec)
    extra = 1
    while gut.has_duplicate(rounded):
        rounded = np.around(levels, round_dec + extra)
        extra += 1
    return rounded


def _auto_extend(z, vmin, vmax, extend):
    if extend is not None:
        return extend
    below = np.nanmin(z) < vmin
    above = np.nanmax(z) > vmax
    if below and above:
        return 'both'
    return 'min' if below else 'max' if above else 'neither'


def resolve_color_scale(z, vmin=None, vmax=None, levels=None,
                        color_bounds='robust', extend=None,
                        sci=None, round_dec=None, log_base=None,
                        is_hatch=False, diverging=False) -> ColorScale:
    """Derive colour limits, levels and colorbar extension for data ``z``.

    Parameters
    ----------
    levels : int, optional
        Number of colour intervals; returned as the array of level edges.
    color_bounds : {'robust', 'quantile', 'minmax', 'absmax', 'absquantile'}
        How vmin/vmax are derived when not given: 'robust' is the 2nd-98th
        percentile, 'quantile' the 10th-90th.
    extend : {'neither', 'min', 'max', 'both'}, optional
        Colorbar extension; derived from the data when None.
    sci, round_dec : int, optional
        Scientific-notation exponent and tick rounding; derived when None.
    log_base : float, optional
        If given, levels are log-spaced between ``base**vmin`` and ``base**vmax``.
    is_hatch : bool
        Hatch overlays only need vmin/vmax: no widening or level rounding.
    diverging : bool
        The colormap is diverging: 'robust' bounds become symmetric about zero
        unless a limit was given.
    """
    for name, value in (('vmin', vmin), ('vmax', vmax)):
        if value is not None and not isinstance(value, numbers.Real):
            raise ValueError(
                f'{name} has to be of type float or int but is of type {type(value)}!')
    if z is None:
        return ColorScale(None, None, None, extend or 'neither', None, None)

    vmin, vmax = _color_bounds(z, vmin, vmax, color_bounds, diverging)
    if vmin == vmax and not is_hatch:
        gut.myprint(f'Vmin and vmax are equal: {vmin}!')
        vmin, vmax = vmin - 0.1, vmax + 0.1

    if log_base is not None:
        levels = np.logspace(vmin, vmax, levels + 1, base=log_base)
    elif levels is not None:
        levels = np.linspace(vmin, vmax, levels + 1, endpoint=True)
    if not is_hatch:
        sci, round_dec = _sci_and_rounding(vmin, vmax, sci, round_dec)
        levels = _round_levels(levels, round_dec)
    return ColorScale(vmin, vmax, levels,
                      _auto_extend(z, vmin, vmax, extend), sci, round_dec)


def add_colorbar(im, fig, label,
                 x_pos=0, y_pos=0,
                 width=None, height=None,
                 set_cax=False,
                 multi_plots=False,
                 **kwargs):
    """Add a colorbar for the result ``im`` of a plot to ``fig``.

    With ``multi_plots`` one colorbar is shared by all panels and placed by the
    layout engine; otherwise it goes into an axes at the given figure
    coordinates (``x_pos``, ``y_pos``, ``width``, ``height``).
    """
    orientation = kwargs.pop('orientation', 'horizontal')
    fsize = kwargs.pop('fontsize', pst.BIGGER_SIZE)
    if multi_plots:
        panels = [a for a in fig.axes if a.get_label() != '<colorbar>']
        return make_colorbar(ax=panels, im=im['im'], fig=fig,
                             orientation=orientation, label=label,
                             set_cax=True, ticks=im['ticks'],
                             fontsize=fsize, **kwargs)
    width = 0.8 if width is None else width
    height = 0.05 if height is None else height
    ax_cbar = fig.add_axes([x_pos, y_pos, width, height])
    return make_colorbar(ax=ax_cbar, im=im['im'], fig=fig,
                         orientation=orientation, label=label,
                         set_cax=set_cax, ticks=im['ticks'],
                         fontsize=fsize, **kwargs)


@dataclass(frozen=True)
class ColorbarSpec:
    """Everything that shapes a colorbar, apart from the mappable and its axes."""
    label: str | None = None
    orientation: str = "horizontal"
    ticks: object = None
    extend: str = "neither"
    tick_step: int = 2
    sci: int | None = None
    round_dec: int | None = None
    set_int: bool = False
    shift_ticks: bool | None = None
    set_rot: bool = False
    set_cax: bool = False
    pad: float | None = None
    shrink: float | None = None  # default: 0.9 horizontal, 0.75 vertical
    fontsize: float | None = None
    norm: str | None = None
    kwargs_bar: dict = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, kwargs):
        """Build a spec from flat keywords, popping the ones it knows."""
        return cls(**{f.name: kwargs.pop(f.name)
                      for f in fields(cls) if f.name in kwargs})


def make_colorbar(ax, im, fig=None, spec=None, **kwargs):
    """Draw a colorbar for ``im``; options come from ``spec`` or flat keywords."""
    if spec is None:
        spec = ColorbarSpec.from_kwargs(kwargs)
    ticks = None if spec.ticks is None else spec.ticks[::int(spec.tick_step)]
    if ticks is not None and spec.round_dec is not None:
        ticks = np.around(ticks, spec.round_dec)
    if ticks is not None and spec.set_int:
        ticks = np.round(ticks).astype(int)
    if fig is None:
        fig = (ax[0] if isinstance(ax, (list, tuple)) else ax).get_figure()

    orientation = spec.orientation
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError(
            f"orientation has to be 'horizontal' or 'vertical', not {orientation!r}!")
    if spec.set_cax:
        # the layout engine makes room next to ``ax``; pad/shrink are fractions
        pad = spec.pad
        if pad is not None and pad < 0:
            warnings.warn(
                "A negative colorbar pad (an offset in older versions) is "
                "ignored; pad is now a fraction of the axes size.",
                DeprecationWarning, stacklevel=4)
            pad = None
        placement = dict(
            ax=ax, pad=0.05 if pad is None else pad,
            shrink=(0.9 if orientation == 'horizontal' else 0.75)
            if spec.shrink is None else spec.shrink,
            location='bottom' if orientation == 'horizontal' else 'right')
    else:
        placement = dict(cax=ax, orientation=orientation)

    kwargs_bar = {**spec.kwargs_bar, 'ticks': ticks}
    kwargs, kwargs_bar = set_kwargs_bar(kwargs, kwargs_bar)
    if spec.sci is not None:
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((spec.sci, spec.sci))
    elif spec.norm == 'log':
        fmt = mpl.ticker.FuncFormatter(cbarfmt)
    else:
        fmt = None
    cbar = fig.colorbar(im, label=spec.label, format=fmt, extend=spec.extend,
                        **placement, **kwargs_bar)
    if spec.sci is not None:
        place_colorbar_offset_text(cbar)

    if spec.shift_ticks is not None and ticks is not None:
        cbar.set_ticks(ticks[:-1]+0.5)
        if spec.set_rot:
            cbar.ax.set_xticklabels(ticks[:-1], rotation=45)
        cbar.set_ticklabels(ticks[:-1])

    if spec.fontsize is not None:
        cbar.ax.tick_params(labelsize=spec.fontsize)
        cbar.set_label(label=spec.label, size=spec.fontsize)

    return cbar


def set_kwargs_bar(kwargs, kwargs_bar):
    if 'vmin' in kwargs.keys():
        kwargs_bar['vmin'] = kwargs.pop('vmin')
    if 'vmax' in kwargs.keys():
        kwargs_bar['vmax'] = kwargs.pop('vmax')
    if 'cmap' in kwargs.keys():
        kwargs_bar['cmap'] = kwargs.pop('cmap')
    return kwargs, kwargs_bar


def place_colorbar_offset_text(cbar, pad_x=0.0, pad_y=0.02):
    """
    Place the ScalarFormatter offset text of a colorbar relative to the colorbar axes.

    Vertical colorbar → above the colorbar (centered horizontally)
    Horizontal colorbar → right of the colorbar (centered vertically)

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar whose offset text will be repositioned.
    pad_x : float, optional
        Horizontal offset in axes coordinates (default=0).
    pad_y : float, optional
        Vertical offset in axes coordinates (default=0.02).

    Returns
    -------
    matplotlib.text.Text
        The offset text object (you can further style it).
    """
    # choose offset text depending on orientation
    if cbar.orientation == "vertical":
        txt = cbar.ax.yaxis.get_offset_text()
        # txt.set_transform(cbar.ax.transAxes)  # use axes coordinates
        txt.set_position((0.5 + pad_x, 1 + pad_y))  # above bar
        txt.set_ha("center")
        txt.set_va("bottom")
    else:
        txt = cbar.ax.xaxis.get_offset_text()
        # txt.set_transform(cbar.ax.transAxes)
        txt.set_position((1 + pad_x, pad_y))  # right of bar
        txt.set_ha("left")
        txt.set_va("center")

    return txt


def discrete_norm_ticks(vmin, vmax, shift_ticks=False, num_ticks=None):
    if vmin is None or vmax is None:
        return None
    if num_ticks is None:
        num_ticks = 10

    if shift_ticks is True:
        # +1.1 to account for start and end
        normticks = np.arange(vmin, vmax + 1.1, dtype=int)
    else:
        normticks = np.linspace(vmin, vmax, num_ticks + 1)

    return normticks
