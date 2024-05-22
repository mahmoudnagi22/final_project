import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import cv2
from math import floor, ceil, sqrt
from gtts import gTTS
import IPython.display as display
import time  # استيراد مكتبة الوقت


### Useful functions
def Padding(image): # padding to constant sizes (292,548)
    p = 292
    q = 548
    s = image.shape
    m = s[0]
    n = s[1]
    b_top = floor((p - m) / 2)
    b_bot = ceil((p - m) / 2)
    b_left = floor((q - n) / 2)
    b_right = ceil((q - n) / 2)
    image = cv2.copyMakeBorder(image, b_top, b_bot, b_left, b_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image

### The path to the selected images
path = "/content/Computer-vision/image"
### Pre-definitions 
valid_x = [".jpg", ".jpeg", ".png", ".tga", ".bmp"] # valid image extensions
num_x = len(os.listdir(path))
# final sizes of the image
height = int(212 / 5) - 2 # 40
width = int(390 / 5) + 2 # 80
channels = 3
classes_num = 7
x = np.empty((num_x, height, width, 3), dtype='float32')
# theoretical sizes for resizing stage_1
sized_height = 200
sized_width = 430
sized = sized_height * sized_width
n = 0
### Pre-processing
for f in os.listdir(path):
    ext = os.path.splitext(f)[1] # getting the file extension
    if ext.lower() not in valid_x:
        continue # pass if the file's extension not in the valid extensions list
    j = os.path.join(path, f)
    image = cv2.imread(j) # reading images
    s = image.shape
    # rotate 90deg in case of height > width
    if s[0] > s[1]:
        image = np.rot90(image, k=1)
        s = image.shape
    resized = s[0] * s[1]
    scale = resized / sized
    # resizing to scale (stage_1)
    image = cv2.resize(image, (floor(s[1] / sqrt(scale)), floor(s[0] / sqrt(scale))))
    # filtering
    image = cv2.medianBlur(image, 3)
    # padding to biggest sizes 
    image = Padding(image) # padding to constant sizes (292, 548)
   
    # cropping
    image = image[40:-40, 79:-79]
    # resizing to final sizes (stage_2) 
    image = cv2.resize(image, (width, height))
    image = image / 255
  
    x[n, :, :, :] = image
    n += 1

# read test data 
x_test = x

### The MODEL

# (construct the CNN model)
# define a sequential model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Optimizer
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# saving weights of the model after every epoch in a given directory
filepath = "/content/weightsCNN/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)

# using tensorboard to visualize the training and view learning curves to know when to stop and choose which epoch as the best
tbCallBack = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True)

# loading the model weights
model.load_weights('/content/Computer-vision/weights-31-0.99.hdf5')

# Constructing confusion matrix and data used for fusion
labels1 = model.predict(x_test, batch_size=32, verbose=0) # labels predicted from test data
# Definitions
y_test_pred1 = [None] * len(labels1)

for i in range(len(labels1)):
    y_test_pred1[i] = list(labels1[i]).index(max(labels1[i])) # index of each sample's maximum value that represents the predicted class

### Playing_predictions
for mm in range(len(y_test_pred1)):
    if y_test_pred1[mm] == 0:
        text = "one pound - 1 LE."
        sound_file = "/content/Computer-vision/sound/one_sound.mp3"
    elif y_test_pred1[mm] == 1:
        text = "Five pounds - 5 LE"
        sound_file = "/content/Computer-vision/sound/Five_pounds_sound.mp3"
    elif y_test_pred1[mm] == 2:
        text = "Ten pounds - 10 LE"
        sound_file = "/content/Computer-vision/sound/Ten_pounds_sound.mp3"
    elif y_test_pred1[mm] == 3:
        text = "Twenty pounds - 20 LE"
        sound_file = "/content/Computer-vision/sound/Twenty_pounds_sound.mp3"
    elif y_test_pred1[mm] == 4:
        text = "Fifty pounds - 50 LE"
        sound_file = "/content/Computer-vision/sound/Fifty_pounds_sound.mp3"
    elif y_test_pred1[mm] == 5:
        text = "One hundred pounds - 100 LE"
        sound_file = "/content/Computer-vision/sound/One_hundred_pounds_sound.mp3"
    elif y_test_pred1[mm] == 6:
        text = "Two hundred pounds - 200 LE"
        sound_file = "/content/Computer-vision/sound/Two_hundred_pounds_sound.mp3"
    else:
        text = "Unexpected value"
        sound_file = None

    if sound_file and os.path.exists(sound_file):
        print("The currency predicted value is: ", text)
        display.display(display.Audio(sound_file, autoplay=True))
        # الانتظار حتى انتهاء الصوت
        audio_duration = 3 # فرض أن طول الصوت هو 3 ثوانٍ
        time.sleep(audio_duration)
    else:
        print(f"Sound file for {text} not found or unexpected value.")