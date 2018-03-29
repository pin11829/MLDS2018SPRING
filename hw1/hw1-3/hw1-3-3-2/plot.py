import numpy as np
import matplotlib.pyplot as plt

data = np.load('plot.npy')
print(data)

space = [2**(i+5) for i in range(5)]

f = plt.figure(figsize=(8, 6))
ax1 = f.add_subplot(111)
ax1.plot(space, data[0], label='train_loss')
ax1.plot(space, data[2], label='test_loss')
ax1.set_xlabel('batch size')
ax1.set_ylabel('loss')
ax1.set_xscale('log')
ax1.legend(loc="best")

ax2 = ax1.twinx()
ax2.plot(space, data[4], '--', label='sensitivity')
ax2.set_ylabel('sensitivity')
ax2.set_xscale('log')
ax2.legend()
plt.savefig('loss.png')
plt.show()
plt.close()

f = plt.figure(figsize=(8, 6))
ax1 = f.add_subplot(111)
ax1.plot(space, data[1], label='train_accuracy')
ax1.plot(space, data[3], label='test_accuracy')
ax1.set_xlabel('batch size')
ax1.set_ylabel('accuracy')
ax1.set_xscale('log')
ax1.legend(loc="best")

ax2 = ax1.twinx()
ax2.plot(space, data[4], '--', label='sensitivity')
ax2.set_ylabel('sensitivity')
ax2.set_xscale('log')
ax2.legend(loc="lower right")
plt.savefig('accu.png')
plt.show()