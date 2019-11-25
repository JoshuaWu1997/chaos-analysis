import time
from bifurcation import Bifuracation
from green_dual_channel import test_points, var, split_num, batch_size

start = time.time()
bifu = Bifuracation(test_points, var, split_num, batch_size)
bifu.batch_compute()
bifu.show_random_attractor([0.2, 0.2])
end = time.time()
print(end - start)
bifu.show_Hopf_Lyapunov(method='save')
