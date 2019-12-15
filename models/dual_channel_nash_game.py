from numba import cuda
import numpy as np

eps = 1e-2
up_lmt = 0.5
low_lmt = 0.001
split_num = 100
batch_size = 5000
test_points = np.array(
    [[r * (up_lmt - low_lmt) / split_num + low_lmt, 0.3] for r in range(split_num)] * batch_size,
    dtype=np.float32)
var = np.array(np.random.rand(batch_size * split_num, 2) * 10, dtype=np.float32)


@cuda.jit
def gpu_iter(A, B, AA, BB, C, D, iter_num):
    i = cuda.grid(1)
    if i < A.shape[0]:
        AA[i] = A[i] + eps
        BB[i] = B[i] + eps
        for k in range(1, iter_num):
            p = A[i] + A[i] * C[i] * (5.25 - 2 * A[i] + 0.25 * B[i])
            pp = AA[i] = AA[i] + AA[i] * C[i] * (5.25 - 2 * AA[i] + 0.25 * BB[i])
            B[i] = B[i] + B[i] * D[i] * (4 - 2 * B[i] + 0.5 * A[i])
            BB[i] = BB[i] + BB[i] * D[i] * (4 - 2 * BB[i] + 0.5 * AA[i])
            A[i] = p
            AA[i] = pp


def iter_function(x, xx, y, iter_num):
    threadsperblock = 512
    blockspergrid = (x[0].size + (threadsperblock - 1)) // threadsperblock
    gpu_iter[blockspergrid, threadsperblock](x[0], x[1], xx[0], xx[1], y[0], y[1], iter_num)
    lya_exp = [np.abs(x[i] - xx[i]) / eps for i in range(2)]
    return x, lya_exp
