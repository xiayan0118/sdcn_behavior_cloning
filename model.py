import csv
import pickle
from math import ceil
from random import shuffle

import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda, Cropping2D
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

# Hyperparameters
BATCH_SIZE = 10
ROW, COL, CH = 160, 320, 3
NUM_EPOCHS = 5
CORRECTIONS = [0.0, 0.2, -0.2]

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
        for i in range(3):
          img_name = './data/' + batch_sample[i].strip()
          img = ndimage.imread(img_name)
          angle = float(batch_sample[3]) + CORRECTIONS[i]
          imgs.append(img)
          angles.append(angle)

          flipped_img = np.fliplr(img)
          flipped_angle = -angle
          imgs.append(flipped_img)
          angles.append(flipped_angle)

      X = np.array(imgs)
      y= np.array(angles)
      yield sklearn.utils.shuffle(X, y)

def build_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(ROW, COL, CH)))
  model.add(Cropping2D(cropping=((70, 25), (0, 0))))
  model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
  # model.add(MaxPool2D())
  model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
  # model.add(MaxPool2D())
  model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
  # model.add(MaxPool2D())
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # model.add(MaxPool2D())
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # model.add(MaxPool2D())
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
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
