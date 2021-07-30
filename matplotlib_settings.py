"""
Settings for preparing consistent and beautiful figures for thesis.
"""
import matplotlib as mpl
import seaborn as sns

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
mpl.rcParams['xtick.minor.size'] = 2.5        # minor tick size in points
mpl.rcParams['xtick.major.width'] = 0.8       # major tick width in points
mpl.rcParams['xtick.minor.width'] = 0.8     # minor tick width in points
# mpl.rcParams['xtick.major.pad'] = 3.5     # distance to major tick label in points
# mpl.rcParams['xtick.minor.pad'] = 3.4     # distan    ce to the minor tick label in points
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
mpl.rcParams['legend.markerscale'] = 1.0
mpl.rcParams['legend.fontsize'] = 10.0
mpl.rcParams['legend.borderpad'] = 0.3
mpl.rcParams['legend.labelspacing'] = 0.5
mpl.rcParams['legend.handlelength'] = 2.0
mpl.rcParams['legend.handletextpad'] = 0.6
mpl.rcParams['legend.columnspacing'] = 0.5

# FIGURE
mpl.rcParams['figure.titlesize'] = 12.
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['figure.figsize'] = set_fig_size()
mpl.rcParams['figure.dpi'] = 600

mpl.rcParams['figure.constrained_layout.use'] = True