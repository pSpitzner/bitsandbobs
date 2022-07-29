# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-18 19:38:37
# @Last Modified: 2022-07-29 11:48:24
# ------------------------------------------------------------------------------ #
# Helper functions for dealing with colors
# ------------------------------------------------------------------------------ #

from matplotlib.colors import LinearSegmentedColormap as _ls
from matplotlib.colors import to_hex, to_rgb, to_rgba, Normalize
from matplotlib.patches import Rectangle as _Rectangle
from matplotlib.colorbar import ColorbarBase as _ColorbarBase
import matplotlib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s | %(name)-12s | %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M')
log = logging.getLogger(__name__)


def get_default_colors(which="both"):
    """
    Returns a list of my default colors.
    Qualitative, somewhat color-blind friendly, in dark and/or light.

    # Parameters
    which : str, "both", "dark", "light"

    # Example
    ```
    import bitsandbobs.plt.get_default_colors
    import matplotlib

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        "color",
        get_default_colors()
    )
    ```

    """

    assert which in ["both", "dark", "light"]

    colors = []
    if which in ["dark", "both"]:
        colors.extend(["#233954", "#ea5e48", "#1e7d72", "#f49546", "#e8bf58"])
    if which in ["light", "both"]:
        colors.extend(["#5886be", "#f3a093", "#53d8c9", "#f2da9c", "#f9c192"])

    return colors



def alpha_to_solid_on_bg(base, alpha, bg="white"):
    """
    Probide a color to start from `base`, and give it opacity `alpha` on
    the background color `bg`.
    """

    def rgba_to_rgb(c, bg):
        bg = matplotlib.colors.to_rgb(bg)
        alpha = c[-1]

        res = (
            (1 - alpha) * bg[0] + alpha * c[0],
            (1 - alpha) * bg[1] + alpha * c[1],
            (1 - alpha) * bg[2] + alpha * c[2],
        )
        return res

    new_base = list(matplotlib.colors.to_rgba(base))
    new_base[3] = alpha
    return matplotlib.colors.to_hex(rgba_to_rgb(new_base, bg))


def fade(k, n, start=1, stop=0.4, invert=False):
    """
    helper to get stepwise lower alphas at same color.
    n = total steps
    k = current step, going from 0 to n-1
    start = maximum obtained value
    stop = minimum obtained value

    # Example
    ```
    num_steps = 5
    for k in range(num_steps):
        color = alpha_to_solid_on_bg(
            base = "red",
            alpha = fade(k = k, n = num_steps, invert=True),
        )
        print(color)
    ```

    """

    if n <= 1:
        return 1

    if invert:
        frac = (k) / (n - 1)
    else:
        frac = (n - 1 - k) / (n - 1)
    alpha = stop + (start - stop) * frac
    return alpha


# ------------------------------------------------------------------------------ #
# Palettes
# ------------------------------------------------------------------------------ #

_palettes = dict()
# good with edge = False
_palettes["cold"] = [
    (0, "0.92"),  # single float as a string for grayscale
    (0.25, "#BEDA9D"),
    (0.45, "#42B3D5"),
    (0.75, "#24295E"),
    (1, "black"),
]
_palettes["hot"] = [
    (0, "0.95"),
    (0.3, "#FEEB65"),
    (0.65, "#E4521B"),
    (0.85, "#4D342F"),
    (1, "black"),
]
_palettes["pinks"] = [
    (0, "0.95"),
    (0.2, "#E0CB8F"),
    # (0.2, "#FFECB3"),
    (0.45, "#E85285"),
    (0.65, "#6A1B9A"),
    (1, "black"),
]

# good with edge = True
_palettes["volcano"] = [
    (0, "#E8E89C"),
    (0.25, "#D29C65"),
    (0.65, "#922C40"),
    (1, "#06102E"),
]

_palettes["pastel_1"] = [
    (0, "#E7E7B6"),
    (0.25, "#ffad7e"),
    (0.5, "#cd6772"),
    (0.75, "#195571"),
    (1, "#011A39"),
]

_palettes["reds"] = [
    (0, "#d26761"),
    (0.25, "#933a3e"),
    (0.5, "#6b354c"),
    (0.75, "#411c2f"),
    (1, "#050412"),
]
_palettes["blues"] = [
    (0, "#bbc2d2"),
    (0.25, "#2865a6"),
    (0.5, "#11395d"),
    (0.75, "#091d35"),
    (1, "#030200"),
]

# hm not super color blind friendly
_palettes["bambus"] = [
    (0, "#D9DFD3"),
    (0.25, "#8FA96D"),
    (0.5, "#9C794F"),
    (1, "#3F2301"),
]

_palettes["div_red_yellow_blue"] = [
    (0.0, "#f94144"),
    (0.2, "#f3722c"),
    (0.3, "#f8961e"),
    (0.5, "#f9c74f"),
    (0.7, "#90be6d"),
    (0.8, "#43aa8b"),
    (1.0, "#577590"),
]

_palettes["div_red_white_blue"] = [
    (0.0, "#233954"),
    (0.25, "#8d99ae"),
    (0.5, "#edf2f4"),
    (0.75, "#ef233c"),
    (1.0, "#d90429"),
]

_palettes["div_pastel_1"] = [
    (0, "#C31B2B"),
    (0.25, "#ffad7e"),
    (0.5, "#E7E7B6"),
    (0.85, "#195571"),
    (1, "#011A39"),
]

_palettes["div_pastel_2"] = [
    (0.0, "#641002"),
    (0.25, "#D82C0E"),
    (0.5, "#FFD500"),
    (0.75, "#B8E0FF"),
    (1.0, "#F0FBFF"),
]

_palettes["grays"] = [
    (0.0, "#999"),
    (1.0, "#000"),
]


def get_palettes():
    return _palettes.keys()

# enable the colormaps for matplotlib cmaps and getting discrete values, eg
# cmap["pinks"](0.5)
cmaps = dict()
for key in _palettes.keys():
    cmaps[key] = _ls.from_list(key, _palettes[key], N=512)


def create_cmap(start="white", end="black", palette=None, N=512, name="custom_colormap"):
    if palette is None:
        palette = [(0.0, start), (1.0, end)]
    return _ls.from_list(name, palette, N=N)


def cmap_cycle(palette="hot", N=5, edge=True, format="hex"):
    if palette not in _palettes.keys():
        raise KeyError(f"Unrecognized palette '{palette}'")

    assert N >= 1

    if format.lower() == "hex":
        to_format = to_hex
    elif format.lower() == "rgb":
        to_format = to_rgb
    elif format.lower() == "rgba":
        to_format = to_rgba

    res = []
    for idx in range(0, N):
        if N == 1:
            arg = 0.5
        else:
            if edge:
                arg = idx / (N - 1)
            else:
                arg = (idx + 1) / (N + 1)

        this_clr = to_format(cmaps[palette](arg))
        if to_format == to_hex:
            this_clr = this_clr.upper()
        res.append(this_clr)

    return res


# TODO: make this work with any cmap argument. cmap_cycle should take either
# a palette string or a cmap callable.
def demo_cmap(palette="hot", Nmax=7, edge=True):
    """
    Get an overview of a colormap.
    """

    assert palette in _palettes.keys()

    dpi = 72
    cell_width = 120
    cell_height = 28
    swatch_width = 117
    swatch_height = 25
    margin = 12
    topmargin = 40
    cbar_width = 50

    ncols = Nmax
    nrows = Nmax
    width = cell_width * ncols + 2 * margin + cbar_width
    height = cell_height * nrows + margin + topmargin

    fig, axes = plt.subplots(
        ncols=2,
        gridspec_kw={"width_ratios": [width - cbar_width, cbar_width]},
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
    )
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - topmargin) / height,
    )
    ax = axes[0]
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(
        f"palette: {palette}", fontsize=24, fontweight="bold", loc="center", pad=10
    )

    print(f"palette: {palette}")
    for N in range(1, Nmax + 1):
        # columns
        colors = cmap_cycle(palette, N, edge, format="rgba")
        clr_desc = cmap_cycle(palette, N, edge, format="hex")
        print(f"N = {N}: {' '.join(clr_desc)}")
        for n in range(1, N + 1):
            # rows

            description = clr_desc[n - 1]
            swatch_clr = colors[n - 1]
            # get a text color that is gray but brightness invert to shown patch
            r, g, b = to_rgb(swatch_clr)
            gray = 0.2989 * (1 - r) + 0.5870 * (1 - g) + 0.1140 * (1 - b)
            if gray < 0.5:
                gray = "black"
            else:
                gray = "white"

            col = N - 1
            row = Nmax / 2 - n + N / 2

            swatch_start_x = col * cell_width
            swatch_start_y = row * cell_height
            text_pos_x = swatch_start_x + cell_width / 2
            text_pos_y = swatch_start_y + cell_height / 2

            if n == N:
                ax.text(
                    text_pos_x,
                    text_pos_y - cell_height,
                    f"N = {N}",
                    fontsize=14,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            ax.text(
                text_pos_x,
                text_pos_y,
                description,
                fontsize=14,
                color=str(gray),
                horizontalalignment="center",
                verticalalignment="center",
            )

            ax.add_patch(
                _Rectangle(
                    xy=(swatch_start_x, swatch_start_y),
                    width=swatch_width,
                    height=swatch_height,
                    facecolor=swatch_clr,
                )
            )

    # add the full color bar to the right
    cbax = axes[1]
    cbar = _ColorbarBase(
        ax=cbax,
        cmap=cmaps[palette],
        norm=Normalize(vmin=0, vmax=1),
        orientation="vertical",
    )
    cbax.axis("off")
    cbar.outline.set_visible(False)

    fig.tight_layout()

    return fig, axes

