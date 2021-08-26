"""
Matplotlib settings for preparing consistent and beautiful figures for thesis.
_____________________________________________________________________________
@ Lakshminarayanan Mohana Kumar
updated on: 24th Aug 2021.
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

__version__ = '1.3'

# units
# 1 pt = 1/72 in = 0.353 mm
# DPI   |   pt  |   dots
# 300       1       4.17
# 600       1       8.33
# 1000      1       13.89

# Figure widths
fig_width_sml = 3.543  # in
fig_width_med = 5.512  # in
fig_width_lrg = 7.480  # in

# aspect ratio
eql = 1.
med = 1.5
phi = 1.618  # golden ratio
dbl = 2.


def set_fig_size(width=fig_width_sml, aspect_ratio=phi, rotated=False):
    if not rotated:
        return width, width/aspect_ratio
    else:
        return width/aspect_ratio, width

# fsize = 10
font = "Times New Roman"
# font = "times"

sns.set(style='ticks',
        palette='deep',
        rc={'font.family': font,
            # "font.size": fsize,
            # "axes.titlesize": fsize,
            # "axes.labelsize": fsize
            }
        )

# LINES
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 5

# FONT
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = font
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.weight'] = 'normal'

# AXES
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.titlesize'] = 12.
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['axes.labelweight'] = 'bold'

# TICKS
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['xtick.major.size'] = 5        # major tick size in points
mpl.rcParams['xtick.minor.size'] = 2.5      # minor tick size in points
mpl.rcParams['xtick.major.width'] = 0.8     # major tick width in points
mpl.rcParams['xtick.minor.width'] = 0.8     # minor tick width in points
# mpl.rcParams['xtick.major.pad'] = 3.5     # distance to major tick label in points
# mpl.rcParams['xtick.minor.pad'] = 3.4     # distance to the minor tick label in points
mpl.rcParams['ytick.labelsize'] = 10.
mpl.rcParams['ytick.major.size'] = 5        # major tick size in points
mpl.rcParams['ytick.minor.size'] = 2.5      # minor tick size in points
mpl.rcParams['ytick.major.width'] = 0.8     # major tick width in points
mpl.rcParams['ytick.minor.width'] = 0.5     # minor tick width in points
# mpl.rcParams['ytick.major.pad'] = 3.5     # distance to major tick label in points
# mpl.rcParams['ytick.minor.pad'] = 3.4     # distan    ce to the minor tick label in points

# GRIDS
# mpl.rcParams['grid.alpha'] = 0.5

# LEGEND
mpl.rcParams['legend.fancybox'] = False     # if True, use a rounded box for the legend background
mpl.rcParams['legend.edgecolor'] = '0.2'    # background patch boundary color
mpl.rcParams['legend.markerscale'] = 1.0    # the relative size of legend markers vs. original
mpl.rcParams['legend.fontsize'] = 10.0
mpl.rcParams['patch.linewidth'] = 0.5       # legend frame lw. Note: affects other patches*
# Dimensions as fraction of font size:
mpl.rcParams['legend.borderpad'] = 0.3      # border whitespace
mpl.rcParams['legend.labelspacing'] = 0.5   # the vertical space between the legend entries
mpl.rcParams['legend.handlelength'] = 2.0   # the length of the legend lines
mpl.rcParams['legend.handleheight'] = 0.7   # the height of the legend handle
mpl.rcParams['legend.handletextpad'] = 0.6  # the space between the legend line and legend text
mpl.rcParams['legend.borderaxespad'] = 0.2  # the border between the axes and legend edge
mpl.rcParams['legend.columnspacing'] = 0.5  # column separation

# FIGURE
mpl.rcParams['figure.titlesize'] = 12.
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['figure.figsize'] = set_fig_size()
mpl.rcParams['figure.dpi'] = 600

mpl.rcParams['figure.constrained_layout.use'] = True

# LaTex
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'