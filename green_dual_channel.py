from numba import cuda
import numpy as np

eps = 1e-3
up_lmt = 0.5
low_lmt = 0.001
split_num = 2000
batch_size = 1000
es = 20000
var_name = ['w', 'p', 't']
test_points = np.array(
    [[
        r * (up_lmt - low_lmt) / split_num + low_lmt, 0.1
    ] for r in range(split_num)] * batch_size, dtype=np.float32
)
var = np.array(np.random.rand(batch_size * split_num, 3) * 10, dtype=np.float32)

a = 3
b = 1
c = 1
d = 0.12
k1 = 0.15
k2 = 0.35
k3 = 0.1


@cuda.jit
def gpu_iter(A, B, C, AA, BB, CC, l, m, iter_num):
    i = cuda.grid(1)
    if i < A.shape[0]:
        for k in range(1, iter_num):
            p = A[i] + A[i] * k1 * (a - b * B[i] + (l[i] + d) * C[i])
            pp = AA[i] + AA[i] * k1 * (a - b * BB[i] + (l[i] + d) * CC[i])
            q = B[i] + B[i] * k2 * (a - 2 * b * B[i] + (l[i] + d) * C[i] + b * A[i])
            qq = BB[i] + BB[i] * k2 * (a - 2 * b * BB[i] + (l[i] + d) * CC[i] + b * AA[i])
            C[i] = C[i] + C[i] * k3 * ((l[i] + d) * A[i] + l[i] * B[i] - m[i] * C[i] - c * (2 * l[i] + d))
            CC[i] = CC[i] + CC[i] * k3 * ((l[i] + d) * AA[i] + l[i] * BB[i] - m[i] * CC[i] - c * (2 * l[i] + d))
            A[i] = p
            AA[i] = pp
            B[i] = q
            BB[i] = qq


def iter_function(x, xx, y, iter_num):
    threadsperblock = 256
    blockspergrid = (x[0].size + (threadsperblock - 1)) // threadsperblock
    epoch = batch_size * split_num // es
    for i in range(epoch):
        gpu_iter[blockspergrid, threadsperblock](
            x[0][i * es:i * es + es], x[1][i * es:i * es + es], x[2][i * es:i * es + es], xx[0][i * es:i * es + es],
            xx[1][i * es:i * es + es], xx[2][i * es:i * es + es], y[0][i * es:i * es + es], y[1][i * es:i * es + es],
            iter_num)
    lya_exp = [np.abs(x[i] - xx[i]) / eps for i in range(3)]
    sel = x[0] < x[1]
    return y[0][sel], [x[i][sel] for i in range(3)], lya_exp