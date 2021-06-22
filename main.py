# get train data
# get test data
# create model
# load or init model
# compile the model
# fit data to model
# save weights

'''
  A simple Conv3D example with Keras
'''
import os
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def video_to_array(video_name: str, no_frames=10):
  array_video = []
  clip = cv2.VideoCapture(video_name)
  nframe = clip.get(cv2.CAP_PROP_FRAME_COUNT)
  frames = [x * nframe / no_frames for x in range(no_frames)]

  for i in range(no_frames):
    clip.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
    ret, frame = clip.read()
    frame = cv2.resize(frame, (32, 32))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    array_video.append(frame)

  clip.release()

  return np.array(array_video)

# -- Preparatory code --
# Model configuration
batch_size = 50
no_epochs = 1
learning_rate = 0.001
no_classes = 2
validation_split = 0.2
verbosity = 1

X_train = []
labels_train = []
train_files = os.listdir("dataset_processed/train/")

progress_bar = tqdm(total=len(train_files))

for filename in train_files:
  progress_bar.update(1)
  if filename.endswith("-context.mp4"):
    continue
  file_path = os.path.join("dataset_processed/train/", filename)
  label = 1 if filename.startswith("cheater") else 0

  labels_train.append(label)
  X_train.append(video_to_array(file_path))

progress_bar.close()

X_train = np.array(X_train).transpose((0, 2, 3, 1))
X_train = X_train.reshape((X_train.shape[0], 32, 32, 10, 1))
X_train = X_train.astype("float32")
Y_train = to_categorical(labels_train, 2)

print('X_shape:{}\nY_shape:{}'.format(X_train.shape, Y_train.shape))

X_test = []
labels_test = []
test_files = os.listdir("dataset_processed/test/")

progress_bar = tqdm(total=len(test_files))

for filename in test_files:
  progress_bar.update(1)
  if filename.endswith("-context.mp4"):
    continue
  file_path = os.path.join("dataset_processed/test/", filename)
  label = 1 if filename.startswith("cheater") else 0

  labels_test.append(label)
  X_test.append(video_to_array(file_path))

progress_bar.close()

X_test = np.array(X_test).transpose((0, 2, 3, 1))
X_test = X_test.reshape((X_test.shape[0], 32, 32, 10, 1))
X_test = X_test.astype("float32")
Y_test = to_categorical(labels_test, 2)

print('X_shape:{}\nY_shape:{}'.format(X_test.shape, Y_test.shape))

# Create the model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
  input_shape=(X_train.shape[1:]), padding="same"))
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='softmax', padding="same"))
model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding="same"))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='softmax', padding="same"))
model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True,
            to_file=os.path.join('model.png'))
# Fit data to model
history = model.fit(X_train, Y_train,
            validation_data=(X_test, Y_test),
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            shuffle=True)

# # Generate generalization metrics
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)

# # Plot history: Categorical crossentropy & Accuracy
plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance for 3D MNIST Keras Conv3D example')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()