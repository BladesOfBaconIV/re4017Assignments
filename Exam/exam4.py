import numpy as np
import matplotlib.pylab as plt


image = np.array(([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                  [0 , 0 , 0 , 0 , 76, 0 , 0 , 0],
                  [0 , 0 , 0 , 0 , 24, 0 , 0 , 0],
                  [0 , 0 , 0 , 76, 67, 59, 55, 0],
                  [0 , 99, 8 , 0 , 0 , 0 , 51, 0],
                  [0 , 0 ,39 , 0 , 76, 0 , 82, 0],
                  [0 , 0 ,24 , 0 , 0 , 0 , 0 , 0],
                  [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]))

outimage = np.zeros([image.shape[0],image.shape[1]])

high = 0.8*99
low = high/2.5

def neighbor(Image,I,J):
    for u in range(I-1, I+2):
        for v in range(J-1, J+2):
            if Image[u][v] == 1:
                return True
    return False

for i in range(1,image.shape[0]-1):
    for j in range(1,image.shape[1]-1):
        if image[i][j] >= high:
            outimage[i][j] = 1

flag = 1
Pass = 0
while flag == 1:
    flag = 0
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            if (image[i][j] >= low) and (neighbor(outimage,i,j)) and (outimage[i][j] == 0):
                outimage[i][j] = 1
                flag = 1
    Pass += 1

print(Pass)
#plt.figure()
#plt.imshow(image,cmap='gray')
plt.figure()
plt.imshow(outimage,cmap='gray')
plt.show()
