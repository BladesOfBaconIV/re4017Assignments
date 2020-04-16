import numpy as np
import matplotlib.pyplot as plt


image = np.array(([0,0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,1,1,1,1,0],
                  [0,1,0,1,1,0,0,0,1,0,0],
                  [0,1,0,1,1,1,1,0,1,1,0],
                  [0,1,1,1,1,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0,0]))

kernel = np.zeros(9)
image2 = np.zeros([image.shape[0],image.shape[1]])
for i in range(1,image.shape[0]-1):
    for j in range(1,image.shape[1]-1):
        kernel[0] = image[i][j]
        kernel[1] = image[i+1][j]
        kernel[2] = image[i-1][j]
        kernel[3] = image[i][j+1]
        kernel[4] = image[i][j-1]
        kernel[5] = image[i+1][j+1]
        kernel[6] = image[i-1][j-1]
        kernel[7] = image[i+1][j-1]
        kernel[8] = image[i-1][j+1]
        Middle = np.sort(kernel)[4]
        image2[i][j] = Middle
plt.figure()
plt.imshow(image2,cmap='gray')
plt.figure()
plt.imshow(image,cmap='gray')
plt.show()