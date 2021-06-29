'''
  Multi Resolution Conv3D with Keras for aimbot detection
'''
import argparse
import os
import random
import math
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
from matplotlib.cbook import flatten
from tqdm import tqdm

def video_to_array(video_name: str, no_frames=10, flip=False):
  array_video = []
  clip = cv2.VideoCapture(video_name)
  nframe = clip.get(cv2.CAP_PROP_FRAME_COUNT)
  frames = [x * nframe / no_frames for x in range(no_frames)]

  for i in range(no_frames):
    clip.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
    ret, frame = clip.read()
    # frame = cv2.resize(frame, (32, 32))
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
batch_size = 80
no_epochs = args.epochs
learning_rate = 0.00001
no_classes = 2
verbosity = 1

X_train_context = []
X_train_fovea = []
labels_train = []

X_val_context = []
X_val_fovea = []
labels_val = []

if not args.test:
  train_files = os.listdir("dataset_processed/train/")
  train_files = list(zip(train_files[::2], train_files[1::2]))
  validation_files = random.sample(train_files, math.floor(len(train_files)*0.2))
  train_files = [file for file in train_files if file not in validation_files]

  progress_bar = tqdm(total=len(train_files)*2)

  for filename1, filename2 in train_files:
    progress_bar.update(1)
    
    file_path1 = os.path.join("dataset_processed/train/", filename1)
    file_path2 = os.path.join("dataset_processed/train/", filename2)
    label = 1 if filename1.startswith("cheater") else 0

    labels_train.append(label)
    
    if filename1.endswith("-context.mp4"):
      X_train_context.append(video_to_array(file_path1))
      X_train_fovea.append(video_to_array(file_path2))
    else:
      X_train_context.append(video_to_array(file_path2))
      X_train_fovea.append(video_to_array(file_path1))
  
  for filename1, filename2 in train_files:
    progress_bar.update(1)
    
    file_path1 = os.path.join("dataset_processed/train/", filename1)
    file_path2 = os.path.join("dataset_processed/train/", filename2)
    label = 1 if filename1.startswith("cheater") else 0

    labels_train.append(label)
    
    if filename1.endswith("-context.mp4"):
      X_train_context.append(video_to_array(file_path1, flip=True))
      X_train_fovea.append(video_to_array(file_path2, flip=True))
    else:
      X_train_context.append(video_to_array(file_path2, flip=True))
      X_train_fovea.append(video_to_array(file_path1, flip=True))

  progress_bar.close()

  X_train_context = np.array(X_train_context).transpose((0, 2, 3, 1))
  X_train_fovea = np.array(X_train_fovea).transpose((0, 2, 3, 1))
  X_train_context = X_train_context.reshape((X_train_context.shape[0], 88, 88, 10, 1))
  X_train_fovea = X_train_fovea.reshape((X_train_fovea.shape[0], 88, 88, 10, 1))
  X_train_context = X_train_context.astype("float32")
  X_train_fovea = X_train_fovea.astype("float32")
  Y_train = to_categorical(labels_train, 2)

  print('X_shape:{}\nY_shape:{}'.format(X_train_context.shape, Y_train.shape))

  progress_bar = tqdm(total=len(validation_files))

  for filename1, filename2 in validation_files:
    progress_bar.update(1)
    
    file_path1 = os.path.join("dataset_processed/train/", filename1)
    file_path2 = os.path.join("dataset_processed/train/", filename2)
    label = 1 if filename1.startswith("cheater") else 0

    labels_val.append(label)
    
    if filename1.endswith("-context.mp4"):
      X_val_context.append(video_to_array(file_path1))
      X_val_fovea.append(video_to_array(file_path2))
    else:
      X_val_context.append(video_to_array(file_path2))
      X_val_fovea.append(video_to_array(file_path1))

  progress_bar.close()

  X_val_context = np.array(X_val_context).transpose((0, 2, 3, 1))
  X_val_fovea = np.array(X_val_fovea).transpose((0, 2, 3, 1))
  X_val_context = X_val_context.reshape((X_val_context.shape[0], 88, 88, 10, 1))
  X_val_fovea = X_val_fovea.reshape((X_val_fovea.shape[0], 88, 88, 10, 1))
  X_val_context = X_val_context.astype("float32")
  X_val_fovea = X_val_fovea.astype("float32")
  Y_val = to_categorical(labels_val, 2)

  print('X_shape:{}\nY_shape:{}'.format(X_val_context.shape, Y_val.shape))

X_test_context = []
X_test_fovea = []
labels_test = []
test_files = os.listdir("dataset_processed/test/")

progress_bar = tqdm(total=len(test_files))

for filename in test_files:
  progress_bar.update(1)

  file_path = os.path.join("dataset_processed/test/", filename)
  label = 1 if filename.startswith("cheater") else 0

  labels_test.append(label)
  if filename.endswith("-context.mp4"):
    X_test_context.append(video_to_array(file_path))
  else:
    X_test_fovea.append(video_to_array(file_path))

progress_bar.close()

X_test_context = np.array(X_test_context).transpose((0, 2, 3, 1))
X_test_fovea = np.array(X_test_fovea).transpose((0, 2, 3, 1))
X_test_context = X_test_context.reshape((X_test_context.shape[0], 88, 88, 10, 1))
X_test_fovea = X_test_fovea.reshape((X_test_fovea.shape[0], 88, 88, 10, 1))
X_test_context = X_test_context.astype("float32")
X_test_fovea = X_test_fovea.astype("float32")
Y_test = to_categorical(labels_test[:len(labels_test)//2], 2)

print('X_shape:{}\nY_shape:{}'.format(X_test_context.shape, Y_test.shape))

input_shape = X_train_context.shape[1:] if len(X_train_context) else X_test_context.shape[1:]

# Create the model
# input context stream
visible1 = Input(shape=input_shape)
conv11 = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
  padding="same")(visible1)
pool12 = MaxPooling3D(pool_size=(3, 3, 3))(conv11)
drop1 = Dropout(0.25)(pool12)
# relu
conv12 = Conv3D(32, kernel_size=(3, 3, 3), strides=(3, 3, 3), activation='softmax', padding="same")(drop1)
# softmax
flat1 = Flatten()(conv12)
# input fovea stream
visible2 = Input(shape=input_shape)
conv21 = Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
  padding="same")(visible2)
pool22 = MaxPooling3D(pool_size=(3, 3, 3))(conv21)
drop2 = Dropout(0.25)(pool22)
conv22 = Conv3D(32, kernel_size=(3, 3, 3), strides=(3, 3, 3), activation='softmax', padding="same")(drop2)
flat2 = Flatten()(conv22)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='sigmoid')(merge)
# hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(no_classes, activation='softmax')(hidden1)
model = Model(inputs=[visible1, visible2], outputs=output)

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True,
            to_file=os.path.join('multi-res-cnn-model.png'))

# Load weights if provided
if args.load is not False:
  model.load_weights(args.load)
# Fit data to model
if not args.test:
  history = model.fit([X_train_context, X_train_fovea], Y_train,
              validation_data=([X_val_context, X_val_fovea], Y_val),
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity,
              shuffle=True)

# # Generate generalization metrics
loss, acc = model.evaluate([X_test_context, X_test_fovea], Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)

if not os.path.isdir(args.output):
  os.mkdir(args.output)
if not args.test:
  model.save_weights(os.path.join(args.output, "multi-res-cnn-{0}-{1}-acc-{2}.h5".format(batch_size, no_epochs, round(acc, 2))))

  # # Plot history: Categorical crossentropy & Accuracy
  plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
  plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
  plt.plot(history.history['accuracy'], label='Accuracy (training data)')
  plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
  plt.title('Model performance for Conv3D for aimbot detection')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper left")
  plt.show()