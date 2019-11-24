import numpy as np
import matplotlib.pyplot as plt
from dual_channel_nash_game_gpu import up_lmt, low_lmt, iter_function, iter_attractor


class Bifuracation:
    def __init__(self, split_num=100, batch_size=5000, iter_num=2000):
        self.split_num = split_num
        self.batch_size = batch_size
        self.iter_num = iter_num
        test_points = np.array(
            [[r * (up_lmt - low_lmt) / split_num + low_lmt, 0.3] for r in range(split_num)] * batch_size,
            dtype=np.float32)
        price = np.array(np.random.rand(batch_size * split_num, 2) * 10, dtype=np.float32)
        self.price = [np.ascontiguousarray(price[:, i]) for i in range(2)]
        self.test_points = [np.ascontiguousarray(test_points[:, i]) for i in range(2)]
        self.bifu = None
        self.x_axis = None
        self.lya_exp = None

    def batch_compute(self):
        price = self.price
        test_points = self.test_points
        self.bifu, self.lya_exp = iter_function(price, test_points, self.iter_num)
        self.x_axis = test_points[0]
        self.lya_exp = [np.nanmax(self.lya_exp[i].reshape((self.batch_size, -1)), axis=0) for i in range(2)]

    def show_random_attractor(self, k):
        trajectory = iter_attractor(k, 1000)
        trajectory = np.array(trajectory)
        print(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.show()

    def show_Hopf_Lyapunov(self):
        plt.figure(figsize=(20.48, 10.80))
        plt.subplot(121)
        scatter1 = plt.scatter(self.x_axis, self.bifu[0], label='price1', color='r', s=1)
        scatter2 = plt.scatter(self.x_axis, self.bifu[1], label='price2', color='b', s=1)
        plt.title('Hopf Bifuracation')
        plt.legend((scatter1, scatter2), ('price1', 'price2'), loc='upper left')
        plt.subplot(122)
        line1, = plt.plot(self.x_axis[:self.split_num], self.lya_exp[0], label='price1', color='r')
        line2, = plt.plot(self.x_axis[:self.split_num], self.lya_exp[1], label='price2', color='b')
        plt.title('Lyapunov Exponent')
        plt.legend(handles=[line1, line2], loc='upper left')
        plt.savefig('Hopf_Lyapunov.png')
