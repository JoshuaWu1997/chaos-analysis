import numpy as np
import matplotlib.pyplot as plt
import time
from numba import vectorize

up_lmt = 0.5
low_lmt = 0.001
iter_num = 2000
split_num = 1000
batch_size = 100
points = np.array([[r * (up_lmt - low_lmt) / split_num + low_lmt, 0.3] for r in range(split_num)] * batch_size,
                  dtype=np.float32)
p0 = np.array(np.random.rand(batch_size * split_num, 2), dtype=np.float32)


@vectorize(["float32(float32,float32,float32)"])
def compute_p1(a, b, c):
    return a + a * c * (5.25 - 2 * a + 0.25 * b)


@vectorize(["float32(float32,float32,float32)"])
def compute_p2(a, b, c):
    return a + a * c * (4 - 2 * a + 0.5 * b)


def iter_function(x, y):
    p1 = compute_p1(x[0], x[1], y[0])
    p2 = compute_p2(x[1], x[0], y[1])
    return [p1, p2]


def batch_compute(price, test_points):
    price = [np.ascontiguousarray(price[:, i]) for i in range(2)]
    test_points = [np.ascontiguousarray(test_points[:, i]) for i in range(2)]
    for i in range(1, iter_num):
        price = iter_function(price, test_points)
    return price, test_points[0]


start = time.time()
bifu, x_axis = batch_compute(p0, points)
end = time.time()
print(end - start)
plt.scatter(x_axis, bifu[0], label='p1', color='r', s=1)
plt.scatter(x_axis, bifu[1], label='p2', color='b', s=1)
plt.show()
