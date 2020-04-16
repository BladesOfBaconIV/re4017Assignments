import numpy as np
import numpy.linalg as li

#                  --------------------> x,c,j
image = np.array(([0,0,0,0,0,0],     # |
                  [0,0,1,1,0,0],     # |
                  [0,1,1,1,0,0],     # |
                  [1,1,1,0,0,0],     # |
                  [1,0,0,0,0,0],     # |
                  [0,0,0,0,0,0]))    # \/ i,r,y
M00 = 0
M10 = 0
M01 = 0
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
            M00 += image[i][j]
            M10 += j*image[i][j]
            M01 += i*image[i][j]

Xave = M10/M00
Yave = M01/M00

u11 = 0
u20 = 0
u02 = 0

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
            u11 += (j - Xave)*(i - Yave)*image[i][j]
            u20 += (j - Xave)**2*image[i][j]
            u02 += (i - Yave)**2*image[i][j]

SCMM = np.array(([u20, u11],
                 [u11, u02]))

eigvalue,eigvector = li.eigh(SCMM)
# eigvalue,eigvector = li.eig(SCMM)
if eigvalue[0] >= eigvalue[1]:
    index = 0
else:
    index = 1
print()
print('Solution is: (%f, %f) or (%f, %f).' %(eigvector[0][index],eigvector[1][index],-eigvector[0][index],-eigvector[1][index]))
print()

