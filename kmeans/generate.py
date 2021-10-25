from numpy.random import RandomState
from sklearn.datasets import make_blobs
import sys
import random

# Specify a random seed to generate deterministic data
random.seed(123)

# Open the output file (sys.argv[2] indicates the output file name)
fi = open("data.in", "w")

# Ask for dataset parameters. Empty string triggers default values.
point_num = 1000000
cluster_num = 15


box = 900   # Range of centers 
std = 50    # Standard deviation

if not (100000 <= point_num <= 1000000 and 1 <= cluster_num <= 20):
    raise RuntimeError("Invalid input parameter!")

# Generate `POINT_NUM` points with `CLUSTER_NUM` Gaussian clusters.
# Here, you can set `RANDOM_STATE=CONSTANT` will generate deterministic numbers
data, clist, centers = make_blobs(n_samples=point_num, centers=cluster_num,
                         cluster_std=std, center_box=(-box, box), return_centers=True, random_state=123)

# Write the points data to file in a standardized format.
# The input contains `1 + center_num + point_num` lines.
# First line: <point_num> <center_num>'
# 2 ~ center_num + 1 lines: Coordinate of initial centers
# a single data point.
fi.write("%d %d\n" % (point_num, cluster_num))
cs = [random.choice(data) for _ in range(cluster_num)]
for c in cs:
    fi.write("%.8lf %.8lf\n" % (c[0], c[1]))

for i in range(len(data)):
    fi.write("%.8lf %.8lf\n" % (data[i][0], data[i][1])) # All black.

# Close the data file.
fi.close()
