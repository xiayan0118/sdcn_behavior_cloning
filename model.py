from keras.models import Sequential
from keras.layers import Flatten, Dense
from scipy import ndimage

# Use generator and add comments
# Train/validation/test
# Adam
# Shuffle before splitting
# http://alexlenail.me/NN-SVG/LeNet.html
# history_object = model.fit_generator(..., verbose=1)
# print(history_object.history.keys())
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
current_path = None
image = ndimage.imread(current_path)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

