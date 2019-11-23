import time
from bifurcation import Bifuracation

start = time.time()
bifu = Bifuracation()
bifu.batch_compute()
end = time.time()
print(end - start)
bifu.show()
