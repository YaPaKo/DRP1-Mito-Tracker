from matplotlib.colors import LinearSegmentedColormap

# Colors for mito/drp1 representation
cm_red = LinearSegmentedColormap.from_list("Custom red", [(1, 0, 0, 0), (1, 0, 0, 1)], N=100)
cm_green = LinearSegmentedColormap.from_list("Custom green", [(0, 1, 0, 0), (0, 1, 0, 1)], N=100)
cm_magenta = LinearSegmentedColormap.from_list("Custom magenta", [(1, 0, 1, 0), (1, 0, 1, 1)], N=100)
cm_yellow = LinearSegmentedColormap.from_list("Custom yellow", [(1, .8, .1, 0), (1, .8, .1, 1)], N=100)
cm_turquoise = LinearSegmentedColormap.from_list("Custom turquoise", [(.1, 1, .86, 0), (.1, 1, .86, 1)], N=100)
