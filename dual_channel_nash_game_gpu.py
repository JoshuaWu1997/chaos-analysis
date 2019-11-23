from numba import cuda

up_lmt = 0.5
low_lmt = 0.001


@cuda.jit
def gpu_iter(A, B, C, D, iter_num):
    i = cuda.grid(1)
    if i < A.shape[0]:
        for k in range(1, iter_num):
            p = A[i] + A[i] * C[i] * (5.25 - 2 * A[i] + 0.25 * B[i])
            B[i] = B[i] + B[i] * D[i] * (4 - 2 * B[i] + 0.5 * A[i])
            A[i] = p


def iter_function(x, y, iter_num):
    threadsperblock = 128
    blockspergrid = (x[0].size + (threadsperblock - 1)) // threadsperblock
    gpu_iter[blockspergrid, threadsperblock](x[0], x[1], y[0], y[1], iter_num)
    return x
