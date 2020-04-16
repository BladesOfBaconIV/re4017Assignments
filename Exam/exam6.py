import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate


def radon(image, steps):
    "Build the Radon Transform using 'steps' projections of 'image’."
    projections = []        # Accumulate projections in a list. 
    dTheta = -180.0 / steps # Angle increment for rotations.
    for i in range(steps):
        projections.append(rotate(image, i*dTheta).sum(axis=0))
    return np.vstack(projections)


image = np.zeros([101,101])

#D:
#image[25:77,38:63] = 1
#A:
image[30:70,38:43] = 1
image[30:70,58:63] = 1
image[25:32,38:63] = 1
image[70:77,38:63] = 1
#C:
#image = np.transpose(image)
#B:
image = rotate(image,-60)
rimage = radon(image,180)
plt.figure()
plt.imshow(image,cmap='gray')
plt.figure()
plt.imshow(np.transpose(rimage))
plt.show()