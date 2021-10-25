import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("Package numpy, matplotlib not installed, use \"python -m pip install numpy matplotlib\" for Installation")


# Color map: index->color, where index in [0, ..., 19]
cmap = ["black"    , "gray"   , "lightcoral"   , "maroon"   , "indigo",
        "chocolate", "gold"   , "lightseagreen", "lawngreen", "olive" ,
        "cyan"     , "skyblue", "navy"         , "blue"     , "red"   ,
        "violet"   , "hotpink", "lightpink"    , "azure"    , "beige" ]

# Open the result file (sys.argv[2] indicates the result file name)
if len(sys.argv) != 2:
    print("Usage: python <data-file.txt>")
    exit(-1)

#  Read data from the result file
with open(sys.argv[1], "r") as fi:
    lines = fi.readlines()

# Convert the data to numpy arrays
data = np.zeros( (len(lines), 2) )
colors = ['black'] * len(lines)
for i, line in enumerate(lines):
    data[i][0], data[i][1], index = line.strip('\n').split(' ') # get data
    colors[i] = cmap[int(index)] # map index to color

# Plot the original distribution
plt.subplot(1, 2, 1)    # Original plot
plt.scatter(data[:,0], data[:,1], c='black', s=1)
plt.title("Original Points")

# Plot the result of K-Means clustering
plt.subplot(1, 2, 2)    # Clustering result
plt.scatter(data[:,0], data[:,1], c=colors, s=1)
plt.title("K-Means Result")

# Show the figure
plt.show()
