import matplotlib.pyplot as plt
import numpy as np

data = np.load('data.npy')

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('min ratio')
plt.ylabel('loss')
plt.savefig('result.png')
plt.show()
