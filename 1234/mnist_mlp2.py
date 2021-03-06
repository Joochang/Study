'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.models import Model
import time

batch_size = 128
num_classes = 10
epochs = 20
dropout_probability = 0.2

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)

'''
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
'''

input_image = Input(shape=(784,))
dense_layer = Dense(1024, activation='relu')(input_image)
dense_dropout = Dropout(p=dropout_probability)(dense_layer) 
dense_layer2 = Dense(1024, activation='relu')(dense_dropout)
dense_dropout2 = Dropout(p=dropout_probability)(dense_layer2) 

output = Dense(num_classes, activation='softmax')(dense_dropout2)
	
model = Model(input=input_image, output=output)


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

start = time.time()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
end = time.time()
spend_time = end - start
spend_time = round(spend_time,2)
print("training spend time:" + str(spend_time))

start = time.time()
score = model.evaluate(x_test, y_test, verbose=0)
end = time.time()
spend_time = end - start
spend_time = round(spend_time,2)
print("testing spend time:" + str(spend_time))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

file_path = "/media/ailab/songyoungtak/keras-master/examples/saved_models/mpl1.hdf5"
model.save_weights(file_path)

















