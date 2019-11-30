import numpy as np
import matplotlib.pyplot as plt
from green_dual_channel import iter_function, eps, var_name, get_attractor
from mpl_toolkits.mplot3d import Axes3D


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

    def show_random_attractor(self):
        points = [0.082, 0.083]
        trajectory = [get_attractor(0.12, points[i], 0.2, 2000) for i in range(2)]
        fig = plt.figure(figsize=(16, 6))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.plot(trajectory[i][:, 0], trajectory[i][:, 1], trajectory[i][:, 2])
            ax.set_title('lambda=' + str(points[i]))
        plt.savefig('attractor.png')

    def save_random_attractor(self):
        points = np.array(range(600, 900)) / 10000
        print(len(points))
        trajectory = [get_attractor(0.12, points[i], 0.2, 2000) for i in range(len(points))]
        for i in range(len(points)):
            print(i)
            fig = plt.figure()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            ax = Axes3D(fig)
            ax.plot(trajectory[i][:, 0], trajectory[i][:, 1], trajectory[i][:, 2])
            plt.savefig('attractor/attractor' + str(points[i]) + '.png')

    def show_Hopf_Lyapunov(self, method='show'):
        plt.figure(figsize=(16, 6))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.subplot(221)
        scatter = []
        line = []
        for i in range(len(var_name[:-1])):
            new = plt.scatter(self.a_axis, self.bifu[i], label=var_name[i], s=1)
            scatter.append(new)
        plt.title('Hopf Bifuracation')
        plt.subplot(223)
        plt.scatter(self.a_axis, self.bifu[2], label=var_name[2], s=1)
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
