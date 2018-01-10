from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import csv
import preprocess
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
EPOCHS = 5

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Number of training examples =', len(train_samples))
print('Number of validation examples =', len(validation_samples))

# compile and train the model using the generator function
train_generator = preprocess.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = preprocess.generator(validation_samples, batch_size=BATCH_SIZE)

model = Sequential()

print('Preprocessing...')
# normalize
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
# trim img to only see section with road
model.add(Cropping2D(cropping=((75,25),(0,0))))
# resize img to 200,66,1
model.add(Lambda(preprocess.resize_imgs))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Dropout(0.9))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Dropout(0.9))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Dropout(0.9))
# model.add(Convolution2D(64, kernel_size=(3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), strides=(1,1), activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Dropout(0.9))
model.add(Conv2D(64, (3, 3), strides=(1,1), activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# model.add(Dropout(0.9))
model.add(Flatten())
# model.add(Dense(units=1164, activation='elu', W_regularizer=l2(0.001)))
# model.add(Dropout(0.9))
model.add(Dense(100, activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(Dropout(0.9))
model.add(Dense(50, activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(Dropout(0.9))
model.add(Dense(10, activation='elu', kernel_regularizer=l2(0.000001)))
# model.add(Dropout(0.9))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

print('Training model...')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, \
                    validation_data=validation_generator, validation_steps=len(validation_samples),\
                    epochs=EPOCHS)

print(model.summary())

#save the model architecture and parameters
print('Saving model...')
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save('model.h5')
model.save_weights("model_weights.h5")
print("Model Saved.")

### print the keys contained in the history object
print(history_object.history.keys())
print(history_object.history.values())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()