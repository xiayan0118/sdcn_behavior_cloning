import matplotlib.pyplot as plt
import pickle
import sys

with open(sys.argv[1], 'rb') as f:
  history = pickle.load(f)

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
