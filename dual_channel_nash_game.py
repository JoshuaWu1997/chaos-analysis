from numba import vectorize

up_lmt = 0.5
low_lmt = 0.001


@vectorize(["float32(float32,float32,float32)"])
def compute_p1(a, b, c):
    return a + a * c * (5.25 - 2 * a + 0.25 * b)


@vectorize(["float32(float32,float32,float32)"])
def compute_p2(a, b, c):
    return a + a * c * (4 - 2 * a + 0.5 * b)


def iter_function(x, y, iter_num):
    for i in range(1, iter_num):
        p = compute_p1(x[0], x[1], y[0])
        x[1] = compute_p2(x[1], x[0], y[1])
        x[0] = p
    return x