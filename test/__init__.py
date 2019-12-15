import time
import numpy as np
import matplotlib.pyplot as plt
from test.test_module import test_points, var, split_num, batch_size, iter_num
from test.test_module import iter_function


def draw_bifu(bifu):
    plt.figure(figsize=(16, 6))
    plt.scatter(test_points[0], bifu[0], s=1)
    plt.scatter(test_points[0], bifu[1], s=1)
    plt.title('Hopf Bifuracation')
    plt.show()


var = [np.ascontiguousarray(var[:, i]) for i in range(var.shape[1])]
test_points = [np.ascontiguousarray(test_points[:, i]) for i in range(test_points.shape[1])]

start = time.time()
bifu = iter_function(var, test_points)
end = time.time()
print(end - start)
draw_bifu(bifu)