# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2022-07-28 15:13:47
# @Last Modified: 2022-08-08 18:06:47
# ------------------------------------------------------------------------------ #
# Various tweaks for matplotlib
# ------------------------------------------------------------------------------ #

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider as _Divider
from mpl_toolkits.axes_grid1 import Size as _Size
import logging

logging.basicConfig(format='%(asctime)s | %(name)-12s | %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M')
log = logging.getLogger(__name__)
# log.setLevel("DEBUG")

_cm = 1 / 2.54



def save_all_figures(path, fmt="pdf", save_pickle=False, **kwargs):
    """
    saves all open figures as pdfs and pickle. to load an existing figure:
    ```
    import pickle
    with open('/path/to/fig.pkl','rb') as fid:
        fig = pickle.load(fid)
    ```
    """
    path = os.path.expanduser(path)
    assert os.path.isdir(path)

    try:
        import pickle
    except ImportError:
        if pickle:
            log.info("Failed to import pickle")
            save_pickle = False

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(*args):
            return iter(*args)

    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "transparent" not in kwargs:
        kwargs["transparent"] = True

    for i in tqdm(plt.get_fignums()):
        fig = plt.figure(i)
        fig.savefig(f"{path}/figure_{i}.{fmt}", **kwargs)
        if save_pickle:
            try:
                os.makedirs(f"{path}/pickle/", exist_ok=True)
                with open(f"{path}/pickle/figure_{i}.pkl", "wb") as fid:
                    pickle.dump(fig, fid)
            except Exception as e:
                print(e)


def load_fig_from_pickle(path):
    import pickle

    with open(path, "rb") as fid:
        fig = pickle.load(fid)

    return fig


def set_size(ax, w, h, b=1.5, l=2.0, t=0.3, r=0.3):
    """
    Set the size of an axes element, and the precise whitespace around it.
    Only works for figures with a single axes.
    All sizes in centimeters.

    Recommended to call `fig.tight_layout()` before.

    # Parameters
    w, h : float
        width and height of the axes element
    b, l, t, r : float, optional
        whitespace around the axes (where ticks etc are placed)
        bottom, left, top and right
    """
    # https://newbedev.com/axes-class-set-explicitly-size-width-height-of-axes-in-given-units

    # we provide sizes in cm.
    w *= _cm
    h *= _cm
    b *= _cm
    l *= _cm
    t *= _cm
    r *= _cm

    fig = ax.get_figure()

    if len(fig.axes) > 1:
        log.warning("Setting the size of figures with more than one ax is not supported")

    # x: left, width, right; and same for y
    hori = [_Size.Scaled(l), _Size.Fixed(w), _Size.Scaled(r)]
    vert = [_Size.Scaled(b), _Size.Fixed(h), _Size.Scaled(t)]


    # the width and height of the rectangle is ignored.
    divider = _Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig.set_size_inches(r+w+l, t+h+b)


def detick(axis, keep_labels=False, keep_ticks=False):
    """
    Really, should not be needed. Try ax.tick_params instead:
    ```
    tick_params(
        top='off',
        bottom='off',
        left='off',
        right='off',
        labelleft='off',
        labelbottom='on',
    )
    ```

    ```
    detick(ax.xaxis)
    detick([ax.xaxis, ax.yaxis])
    ```
    """

    log.warning("`detick()` is deprecated. Really, `ax.tick_params` has all you need.")

    # Only keeping this for references
    if not isinstance(axis, list):
        axis = [axis]
    for a in axis:
        if not keep_labels and not keep_ticks:
            a.set_ticks_position("none")
            a.set_ticks([])
        elif not keep_labels and keep_ticks:
            a.set_ticklabels([])
        elif keep_labels and not keep_ticks:
            raise NotImplementedError


def apply_default_legend_style(leg):
    """
    a legend style I use frequently

    # Example
    ```
        fig, ax = plt.subplots()
        leg = ax.legend()
        apply_default_legend_style(leg)

    ```
    """
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("#e4e5e6")
    leg.get_frame().set_alpha(0.9)


def move_legend_into_new_axes(ax, figsize=None):
    """
    Move an existing legend into a new figure, e.g. for customizing in post production

    # Example
    ```
    fig, ax = plt.subplots()
    ax.legend()
    move_legend_into_new_axes(ax)
    ```
    """
    if figsize is None:
        figsize = (6 * _cm, 6 * _cm)

    fig, ax_leg = plt.subplots(figsize=figsize)
    h, l = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    leg = ax_leg.legend(h, l, loc="upper left")

    return leg


def get_shifted_formatter(shift=-60, fmt=".1f"):
    """
    # Example
    ```
    ax.xaxis.set_major_formatter(get_formatter_shifted(shift=-120))
    ```
    """

    def formatter(x, pos):
        return "{{:{}}}".format(fmt).format(x + shift)

    return formatter


def ticklabels_lin_to_log10(x, pos):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10
     0 -> 1
    -1 -> 0.1

    # Example
    ```
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(ticklabels_lin_to_log10_power)
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticklocator_lin_to_log_minor())
    ```
    """
    prec = int(np.ceil(-np.minimum(x, 0)))
    return "{{:.{:1d}f}}".format(prec).format(np.power(10.0, x))


def ticklabels_lin_to_log10_power(x, pos, nicer=True, nice_range=[-1, 0, 1]):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10^1
     0 -> 10^0
    -1 -> 10^-1
    """
    if x.is_integer():
        # use easy to read formatter if exponents are close to zero
        if nicer and x in nice_range:
            return ticklabels_lin_to_log10(x, pos)
        else:
            return r"$10^{{{:d}}}$".format(int(x))
    else:
        # return r"$10^{{{:f}}}$".format(x)
        return ""


def ticklocator_lin_to_log_minor(vmin=-10, vmax=10, nbins=10):
    """
    get minor ticks on when manually converting lin to log
    """
    locs = []
    orders = int(np.ceil(vmax - vmin))
    for o in range(int(np.floor(vmin)), int(np.floor(vmax + 1)), 1):
        locs.extend([o + np.log10(x) for x in range(2, 10)])
    return matplotlib.ticker.FixedLocator(locs, nbins=nbins * orders)


def fix_log_ticks(ax_el, every=1, hide_label_condition=lambda idx: False):
    """
    this can adapt log ticks to only show every second tick, or so.

    # Parameters
    ax_el: usually `ax.yaxis`
    every: 1 or 2
    hide_label_condition : function e.g. `lambda idx: idx % 2 == 0`
    """
    ax_el.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax_el.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(0, 1.05, every / 10), numticks=10
        )
    )
    ax_el.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for idx, lab in enumerate(ax_el.get_ticklabels()):
        # print(idx, lab, hide_label_condition(idx))
        if hide_label_condition(idx):
            lab.set_visible(False)


def tick_format_k(prec):
    """
    format axis labels with the `k` abbreviation, i.e. `10_000` becomes `10 k`.
    tick_format_k(0)(1200, 1000.0) gives "1 k"
    tick_format_k(1)(1200, 1000.0) gives "1.2 k"

    Example
    ```
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(tick_format_k(2))
    )
    ```
    """

    def inner(xval, tickpos):
        if xval == 0:
            return "0"
        else:
            return f"${xval/1_000:.{prec}f}\,$k"

    return inner


def pretty_log_ticks(ax_el, prec=2):
    """
    https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting

    Example
    ```
        pretty_log_ticks(ax.yaxis, prec=2))
    ```

    """
    #

    def myLogFormat(y, pos, prec=prec):
        if y > np.power(10.0, prec) or y < np.power(10.0, -prec):
            return r"$10^{{{:d}}}$".format(int(np.log10(y)))

        else:
            # Find the number of decimal places required
            decimalplaces = int(np.maximum(-np.log10(y), 0))
            # Insert that number into a format string
            formatstring = "{{:.{:1d}f}}".format(decimalplaces)
            # Return the formatted tick label
        return formatstring.format(y)

    ax_el.set_major_formatter(matplotlib.ticker.FuncFormatter(myLogFormat))
