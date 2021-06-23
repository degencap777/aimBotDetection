'''
  Multi Resolution Conv3D with Keras for aimbot detection
'''
import argparse
import os
import cv2
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def video_to_array(video_name: str, no_frames=10, flip=False):
  array_video = []
  clip = cv2.VideoCapture(video_name)
  nframe = clip.get(cv2.CAP_PROP_FRAME_COUNT)
  frames = [x * nframe / no_frames for x in range(no_frames)]

  for i in range(no_frames):
    clip.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
    ret, frame = clip.read()
    frame = cv2.resize(frame, (32, 32))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if flip:
      frame = cv2.flip(frame, 1)
    array_video.append(frame)

  clip.release()

  return np.array(array_video)

# -- Command line argumests --
parser = argparse.ArgumentParser(description="A simple Conv3D for aimbot detection")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--test", type=bool, default=False)
parser.add_argument("--load", type=str, default=False)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

if args.test is True and not args.load:
  raise ValueError("For a test run you need to provide a file with the weights. Use --load.")

# -- Preparatory code --
# Model configuration
batch_size = 100
no_epochs = args.epochs
learning_rate = 0.001
no_classes = 2
verbosity = 1

X_train = []
labels_train = []

if not args.test:
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
    # labels_train.append(label)
    # X_train.append(video_to_array(file_path, flip=True))

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

input_shape = X_train.shape[1:] if len(X_train) else X_test.shape[1:]

# Create the model
# input context stream
visible1 = Input(shape=input_shape)
conv11 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(visible1)
pool11 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(conv11)
conv12 = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding="same")(pool11)
pool12 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(conv12)
flat1 = Flatten()(pool12)
# input fovea stream
visible2 = Input(shape=input_shape)
conv21 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(visible2)
pool21 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(conv21)
conv22 = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding="same")(pool21)
pool22 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(no_classes, activation='softmax')(hidden2)
model = Model(inputs=[visible1, visible2], outputs=output)

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True,
            to_file=os.path.join('multi-res-cnn-model.png'))

# Load weights if provided
# if args.load is not False:
#   model.load_weights(args.load)
# # Fit data to model
# if not args.test:
#   history = model.fit(X_train, Y_train,
#               validation_data=(X_test, Y_test),
#               batch_size=batch_size,
#               epochs=no_epochs,
#               verbose=verbosity,
#               shuffle=True)

# # # Generate generalization metrics
# loss, acc = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', loss)
# print('Test accuracy:', acc)

# if not os.path.isdir(args.output):
#   os.mkdir(args.output)
# if not args.test:
#   model.save_weights(os.path.join(args.output, "multi-res-cnn-{0}-{1}-acc-{2}.h5".format(batch_size, no_epochs, round(acc, 2))))

#   # # Plot history: Categorical crossentropy & Accuracy
#   plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
#   plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
#   plt.plot(history.history['accuracy'], label='Accuracy (training data)')
#   plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
#   plt.title('Model performance for Conv3D for aimbot detection')
#   plt.ylabel('Loss value')
#   plt.xlabel('No. epoch')
#   plt.legend(loc="upper left")
#   plt.show()