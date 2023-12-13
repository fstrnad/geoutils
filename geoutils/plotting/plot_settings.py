import matplotlib.pyplot as plt

TINY_SIZE = 8
MINI_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rcdefaults()

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
# fontsize of the x and y labels
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc("title", titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# plt.rcParams['pcolor.shading'] ='nearest' # For pcolormesh
colors = [
    "steelblue",
    'firebrick',
    "tab:green",
    "m",
    "tomato",
    "c",
    "tab:orange",
    "tab:purple",
    'darkgray',
    "y",
    "slategray",
    "darkviolet",
    "tab:blue",
]
