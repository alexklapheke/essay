import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

palette = [
    "#76a254",  # green
    "#727ecf",  # blue
    "#e93834",  # red
    "#c55faf",  # pink
    "#913140",  # purple
    "#d67c00"   # orange
]

def setup(title, xlabel, ylabel, xformat=None, yformat=None, xgrid=False, ygrid=False, ax=None):
    """Define basic, consistent format for all plots"""

    plt.rcParams["font.family"] = "Linux Biolinum O"

    if not ax:
        ax = plt.gca()

    # Set labels
    ax.set_title(title,   fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Set label formats if applicable (e.g., '$%dk' for "thousands of dollars")
    if xformat:
        ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    if yformat:
        ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))

    # Remove tick marks <https://stackoverflow.com/a/29988431>
    ax.tick_params(axis="both", which="both", length=0, colors="gray")

    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    # ax.title.set_color("gray")

    # Remove bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(False)
    # Add dashed gridlines if necessary
    if xgrid:
        ax.grid(True, axis="x", ls=":")
    if ygrid:
        ax.grid(True, axis="y", ls=":")

    return ax


def format_suffix(number):
    # the rounding is to deal with floating point errors
    # that underestimate the order of magnitude
    mag = len(str(number))
    suffixes = "  kMBTQ"
    group = 1
    if len(str(number)) <= 4:
        return "{:,}".format(number)
    while mag > group * 3:
        group += 1
        number /= 1000
    return "{}{}".format(str(number)[:3].rstrip("."), suffixes[group])


# <https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html>
def autolabel(rects, fig, ax, suffix="", widths=None, *args, **kwargs):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        width = widths[i] if widths else rect.get_width()
        ax.annotate(format_suffix(width) + suffix,
                    xy=(fig.get_figwidth(), rect.get_y()+rect.get_height()/2),
                    xytext=(225, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    va="top",
                    *args, **kwargs
                    )
