from numba import cuda
import numpy as np

eps = 1e-5
up_lmt = 0.08
low_lmt = 0.077
split_num = 1000
batch_size = 100000
es = 200000
var_name = ['w', 'p', 't']
test_points = np.array(
    [[
        r * (up_lmt - low_lmt) / split_num + low_lmt, 0.06
    ] for r in range(split_num)] * batch_size, dtype=np.float64
)
var = np.random.rand(batch_size * split_num, 3) * 10
var[:, :-1] += 5
var[:, 2] = var[:, 2] * 3 + 20
var = np.array(var, dtype=np.float64)

a = 3
b = 1
c = 1
d = 0.12

k1 = 0.15
k2 = 0.35
k3 = 0.1
'''
k1 = 0.2
k2 = 0.2
k3 = 0.1
'''

def get_attractor(d, l, m, iter_num):
    j = True
    while j:
        tra = []
        p0 = np.random.rand(3) * 10
        for i in range(iter_num):
            x = p0[0] + p0[0] * k1 * (a - b * p0[1] + (l + d) * p0[2])
            y = p0[1] + p0[1] * k2 * (a - 2 * b * p0[1] + (l + d) * p0[2] + b * p0[0])
            p0[2] = p0[2] + p0[2] * k3 * ((l + d) * p0[0] + l * p0[1] - m * p0[2] - c * (2 * l + d))
            p0[0] = x
            p0[1] = y
            if i > iter_num - 100:
                if np.isnan(x):
                    j = True
                    break
                else:
                    tra.append(np.array(p0))
                    j = False
    return np.array(tra)


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
    threadsperblock = 1024
    blockspergrid = (x[0].size + (threadsperblock - 1)) // threadsperblock
    epoch = batch_size * split_num // es
    for i in range(epoch):
        gpu_iter[blockspergrid, threadsperblock](
            x[0][i * es:i * es + es], x[1][i * es:i * es + es], x[2][i * es:i * es + es], xx[0][i * es:i * es + es],
            xx[1][i * es:i * es + es], xx[2][i * es:i * es + es], y[0][i * es:i * es + es], y[1][i * es:i * es + es],
            iter_num)
    lya_exp = [np.abs(x[i] - xx[i]) / eps for i in range(3)]
    lya_exp = np.log(lya_exp)
    x = [np.concatenate((x[i], xx[i])) for i in range(3)]
    z = np.concatenate((y[0], y[0]))
    sel = x[0] <= x[1] + eps
    sel = np.logical_and(sel, x[0] > 0)
    sel = np.logical_and(sel, x[1] < 15)
    return z[sel], [x[i][sel] for i in range(3)], lya_exp
