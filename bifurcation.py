import numpy as np
import matplotlib.pyplot as plt
from green_dual_channel import iter_function, eps, var_name


class Bifuracation:
    def __init__(self, test_points, var, split_num, batch_size, iter_num=3000):
        self.split_num = split_num
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.var = [np.ascontiguousarray(var[:, i]) for i in range(var.shape[1])]
        self.varr = [self.var[i] + eps for i in range(var.shape[1])]
        self.test_points = [np.ascontiguousarray(test_points[:, i]) for i in range(test_points.shape[1])]
        self.bifu = None
        self.a_axis = None
        self.b_axis = None
        self.lya_exp = None

    def batch_compute(self):
        var = self.var
        varr = self.varr
        test_points = self.test_points
        self.a_axis, self.bifu, self.lya_exp = iter_function(var, varr, test_points, self.iter_num)
        self.b_axis = test_points[0]
        self.lya_exp = [np.nanmax(self.lya_exp[i].reshape((self.batch_size, -1)), axis=0) for i in
                        range(len(var))]

    def show_random_attractor(self, k):
        '''
        trajectory = iter_attractor(k, 1000)
        trajectory = np.array(trajectory)
        print(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.show()
        '''

    def show_Hopf_Lyapunov(self, method='show'):
        plt.figure(figsize=(16, 9))
        plt.subplot(121)
        scatter = []
        line = []
        for i in range(len(var_name)):
            new = plt.scatter(self.a_axis, self.bifu[i], label=var_name[i], s=1)
            scatter.append(new)
        plt.title('Hopf Bifuracation')
        plt.subplot(122)
        for i in range(len(var_name)):
            new, = plt.plot(self.b_axis[:self.split_num], self.lya_exp[i], label=var_name[i])
            line.append(new)
        plt.title('Lyapunov Exponent')
        plt.legend(handles=line, loc='upper left')
        if method == 'show':
            plt.show()
        else:
            plt.savefig('Hopf_Lyapunov.png')
