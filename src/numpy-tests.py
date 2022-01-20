
#%%
import numpy as np

M = np.zeros((8, 8))

for i in range(8):
    for j in range(8):
        M[j, i] = 0.1 * (i*8+j)

l = 0.1
div = l / M.shape[1]
L2_loss = div * np.sum(np.square(M))
print(L2_loss)

R = np.ones_like(M)
print(M + 0.1 * R)

