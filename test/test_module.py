from numba import cuda, jit, float32
import numpy as np
import math

up_lmt = 0.5
low_lmt = 0.001
split_num = 1000
batch_size = 100
iter_num = 3000
test_points = np.array(
    [[r * (up_lmt - low_lmt) / split_num + low_lmt, 0.3] for r in range(split_num)] * batch_size,
    dtype=np.float32)
var = np.array(np.random.rand(batch_size * split_num, 2) * 10, dtype=np.float32)


@cuda.jit
def gpu_iter(A, B, C, D):
    i = cuda.grid(1)
    if i < A.shape[0]:
        for k in range(1, iter_num):
            p = A[i] + A[i] * C[i] * (5.25 - 2 * A[i] + 0.25 * B[i])
            B[i] = B[i] + B[i] * D[i] * (4 - 2 * B[i] + 0.5 * A[i])
            A[i] = p


def cpu_iter(x, y):
    for i in range(x[0].shape[0]):
        for k in range(1, iter_num):
            p = x[0][i] + x[0][i] * y[0][i] * (5.25 - 2 * x[0][i] + 0.25 * x[1][i])
            x[1][i] = x[1][i] + x[1][i] * y[1][i] * (4 - 2 * x[1][i] + 0.5 * x[0][i])
            x[0][i] = p
    return x


def improved_cpu(x, y):
    x = np.array(x)
    y = np.array(y)
    for k in range(iter_num):
        x = [
            x[:, i] + x[:, i] * y[:, i] * (
                    np.array([5.25, 4]).T - np.matmul(np.array([[-2, 0.25], [0.5, -2]]), x[:, i]))
            for i in range(x.shape[1])
        ]
        x = np.array(x).T
    return x


def numpy_cpu(x, y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray([[5.25, 4]] * x.shape[1]).T
    s = np.ascontiguousarray([[-2, 0.25], [0.5, -2]])
    for k in range(iter_num):
        x = np.add(
            np.multiply(
                x,
                np.multiply(
                    y,
                    np.add(
                        z,
                        np.matmul(
                            s,
                            x
                        )
                    )
                )
            ),
            x
        )
    return x


@cuda.jit
def gpu_iter_2(A, B, C):
    sA = cuda.shared.array(shape=(2, 1), dtype=float32)
    sB = cuda.shared.array(shape=(2, 2), dtype=float32)
    sA[0, 0] = 5.24
    sA[1, 0] = 4
    sB[0, 0] = -2
    sB[1, 0] = 0.5
    sB[0, 1] = 0.25
    sB[1, 1] = -2
    x, y = cuda.grid(2)
    if x <= A.shape[0] and y <= A.shape[1]:
        C[x, y] = A[x, y] * B[x, y] * (sA[x, 0] + sB[x, 0] * A[0, y] + sB[x, 1] * A[1, y]) + A[x, y]


def gpu_2D(x, y):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray([[5.25, 4]] * x.shape[1]).T
    threadsperblock = (2, 512)
    blockspergrid_x = math.ceil(x.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = math.ceil(x[0].shape[0] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    for k in range(3000):
        gpu_iter_2[blockspergrid, threadsperblock](x, y, z)
        x = z
    return x


def iter_function(x, y):
    threadsperblock = 1024
    blockspergrid = math.ceil(x[0].size + (threadsperblock - 1)) // threadsperblock
    # gpu_iter[blockspergrid, threadsperblock](x[0], x[1], y[0], y[1])
    x = gpu_2D(x, y)
    return x
