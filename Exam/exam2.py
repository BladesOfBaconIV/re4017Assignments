import numpy as np


bins  = 10
#image = np.array(([9,9,9,9,9,9],
#                  [9,2,2,2,2,9],
#                  [9,2,1,2,2,9],
#                  [9,2,2,1,2,9],
#                  [9,2,2,2,2,9],
#                  [9,9,9,9,9,9]))

image = np.array(([0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,9,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]))

size = image.shape[0]*image.shape[1]


hist = np.zeros(bins)
for v in image.ravel(): hist[v] += 1
print(hist)
AccH = np.zeros(bins)
sum = 0
for i in range(hist.shape[0]):
    sum = hist[i] + sum
    AccH[i] = sum

Eqimage = np.zeros([image.shape[0],image.shape[1]]).astype(int)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        Eqimage[i][j] = int(AccH[image[i][j]]*(bins-1)/size)
print(Eqimage)
print(AccH)

hist2 = np.zeros(bins)
for v in Eqimage.ravel(): hist2[v] += 1
print("\n\nThe solution is:")
print(hist2)