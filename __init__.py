import time
from bifurcation import Bifuracation

start = time.time()
bifu = Bifuracation()
bifu.batch_compute()
bifu.show_random_attractor([0.2, 0.2])
end = time.time()
print(end - start)
bifu.show_Hopf_Lyapunov()
