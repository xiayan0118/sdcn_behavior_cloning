import csv
import pickle
from math import ceil
from random import shuffle

import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential
from scipy import ndimage
from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D

# Use generator and add comments
# Train/validation/test
# Adam
# Shuffle before splitting
# http://alexlenail.me/NN-SVG/LeNet.html
# history_object = model.fit_generator(..., verbose=1)
# print(history_object.history.keys())
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# image = ndimage.imread(current_path)

# Hyperparameters
BATCH_SIZE = 32
# Todo: change ROW to 80 after trimming
ROW, COL, CH = 160, 320, 3
NUM_EPOCHS = 5

def load_images():
  samples = []
  with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
      samples.append(line)

  shuffle(samples)
  train_samples, val_samples = train_test_split(samples, test_size=0.2)
  return train_samples, val_samples

def generator(samples, batch_size=32):
  num_samples = len(samples)

  while True: # loop forever so the generator never terminates
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      imgs = []
      angles = []
      for batch_sample in batch_samples:
        center_name = './data/' + batch_sample[0]
        # Todo: add left and right side images
        center_img = ndimage.imread(center_name)
        cener_angle = float(batch_sample[3])
        imgs.append(center_img)
        angles.append(cener_angle)

      X_train = np.array(imgs)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

def build_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(ROW, COL, CH)))
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPool2D())
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPool2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(1))
  return model


if __name__ == "__main__":
  train_samples, val_samples = load_images()
  train_gen = generator(train_samples, BATCH_SIZE)
  val_gen = generator(val_samples, BATCH_SIZE)

  next(val_gen)

  model = build_model()
  model.compile(loss='mse', optimizer='adam')
  history_object= model.fit_generator(generator=train_gen,
                      steps_per_epoch=ceil(len(train_samples)/BATCH_SIZE),
                      validation_data=val_gen,
                      validation_steps=ceil(len(val_samples)/BATCH_SIZE),
                      epochs=NUM_EPOCHS,
                      verbose=1)

  # Save model
  model.save("model.h5")

  # Save training/validation losses
  with open('history.p', 'wb') as f:
    pickle.dump(history_object.history, f)
