from PIL import Image
import imageio
import numpy as np

image = []
points = np.array(range(600, 900)) / 10000
for point in points:
    image.append(imageio.imread('attractor' + str(point) + '.png'))
imageio.mimsave('attractor.gif', image, 'GIF', duration=0.1)
