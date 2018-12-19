'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, Input, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import RMSprop
import time

lr = 1e-5
batch_size = 32
nb_classes = 10
nb_epoch = 250
num_classes = 10
dropout_probability = 0.2
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



'''
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''

input_image = Input(shape=(32, 32, 3))
conv_layer = Conv2D(32, (3, 3), activation='relu')(input_image)
conv_layer2 = Conv2D(32, (3, 3), activation='relu')(conv_layer)
max_pooling = MaxPooling2D(pool_size=(2, 2))(conv_layer2)
dropout_layer = Dropout(p=dropout_probability)(max_pooling)

conv_layer3 = Conv2D(64, (3, 3), activation='relu')(dropout_layer)
conv_layer4 = Conv2D(64, (3, 3), activation='relu')(conv_layer3)
max_pooling2 = MaxPooling2D(pool_size=(2, 2))(conv_layer4)
dropout_layer2 = Dropout(p=dropout_probability)(max_pooling2)

global_layer = GlobalAveragePooling2D()(dropout_layer2)

#flatten_layer = Flatten()(dropout_layer2)


'''
dense_layer = Dense(256, activation='relu')(flatten_layer)
dense_dropout3 = Dropout(p=dropout_probability)(dense_layer) 
dense_layer2 = Dense(256, activation='relu')(flatten_layer)
dense_dropout4 = Dropout(p=dropout_probability)(dense_layer2) 

dense_layer3 = Dense(256, activation='relu')(dense_dropout3)
dense_dropout5 = Dropout(p=dropout_probability)(dense_layer3)
dense_layer4 = Dense(256, activation='relu')(dense_dropout4)
dense_dropout6 = Dropout(p=dropout_probability)(dense_layer4)

dense_concatenate = keras.layers.concatenate([dense_dropout5, dense_dropout6])
'''


'''
dense_layer = Dense(512, activation='relu')(flatten_layer)
dense_dropout3 = Dropout(p=dropout_probability)(dense_layer) 
dense_layer2 = Dense(512, activation='relu')(dense_dropout3)
dense_dropout4 = Dropout(p=dropout_probability)(dense_layer2)
'''

'''
dense_layer = Dense(128, activation='relu')(flatten_layer)
dense_dropout3 = Dropout(p=dropout_probability)(dense_layer) 
dense_layer2 = Dense(128, activation='relu')(flatten_layer)
dense_dropout4 = Dropout(p=dropout_probability)(dense_layer2) 

dense_layer3 = Dense(128, activation='relu')(flatten_layer)
dense_dropout5 = Dropout(p=dropout_probability)(dense_layer3)
dense_layer4 = Dense(128, activation='relu')(flatten_layer)
dense_dropout6 = Dropout(p=dropout_probability)(dense_layer4)


dense_layer5 = Dense(128, activation='relu')(dense_dropout3)
dense_dropout7 = Dropout(p=dropout_probability)(dense_layer5) 
dense_layer6 = Dense(128, activation='relu')(dense_dropout4)
dense_dropout8 = Dropout(p=dropout_probability)(dense_layer6) 

dense_layer7 = Dense(128, activation='relu')(dense_dropout5)
dense_dropout9 = Dropout(p=dropout_probability)(dense_layer7)
dense_layer8 = Dense(128, activation='relu')(dense_dropout6)
dense_dropout10 = Dropout(p=dropout_probability)(dense_layer8)
'''

'''
dense_layer = Dense(64, activation='relu')(flatten_layer)
dense_dropout3 = Dropout(p=dropout_probability)(dense_layer) 
dense_layer2 = Dense(64, activation='relu')(flatten_layer)
dense_dropout4 = Dropout(p=dropout_probability)(dense_layer2) 
dense_layer3 = Dense(64, activation='relu')(flatten_layer)
dense_dropout5 = Dropout(p=dropout_probability)(dense_layer3)
dense_layer4 = Dense(64, activation='relu')(flatten_layer)
dense_dropout6 = Dropout(p=dropout_probability)(dense_layer4)
dense_layer5 = Dense(64, activation='relu')(flatten_layer)
dense_dropout7 = Dropout(p=dropout_probability)(dense_layer5) 
dense_layer6 = Dense(64, activation='relu')(flatten_layer)
dense_dropout8 = Dropout(p=dropout_probability)(dense_layer6) 
dense_layer7 = Dense(64, activation='relu')(flatten_layer)
dense_dropout9 = Dropout(p=dropout_probability)(dense_layer7)
dense_layer8 = Dense(64, activation='relu')(flatten_layer)
dense_dropout10 = Dropout(p=dropout_probability)(dense_layer8)

dense_layer9 = Dense(64, activation='relu')(dense_dropout3)
dense_dropout11 = Dropout(p=dropout_probability)(dense_layer9) 
dense_layer10 = Dense(64, activation='relu')(dense_dropout4)
dense_dropout12 = Dropout(p=dropout_probability)(dense_layer10) 
dense_layer11 = Dense(64, activation='relu')(dense_dropout5)
dense_dropout13 = Dropout(p=dropout_probability)(dense_layer11)
dense_layer12 = Dense(64, activation='relu')(dense_dropout6)
dense_dropout14 = Dropout(p=dropout_probability)(dense_layer12)
dense_layer13 = Dense(64, activation='relu')(dense_dropout7)
dense_dropout15 = Dropout(p=dropout_probability)(dense_layer13) 
dense_layer14 = Dense(64, activation='relu')(dense_dropout8)
dense_dropout16 = Dropout(p=dropout_probability)(dense_layer14) 
dense_layer15 = Dense(64, activation='relu')(dense_dropout9)
dense_dropout17 = Dropout(p=dropout_probability)(dense_layer15)
dense_layer16 = Dense(64, activation='relu')(dense_dropout10)
dense_dropout18 = Dropout(p=dropout_probability)(dense_layer16)
'''

#dense_concatenate = keras.layers.concatenate([dense_dropout7, dense_dropout8, dense_dropout9, dense_dropout10])

#dense_layer17 = Dense(512, activation='relu')(dense_dropout4)
#dense_dropout19 = Dropout(p=dropout_probability)(dense_layer17) 

output = Dense(num_classes, activation='softmax')(global_layer)
	
model = Model(input=input_image, output=output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=lr),
              metrics=['accuracy'])




X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

start = time.time()
model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
end = time.time()
spend_time = end - start
spend_time = round(spend_time,2)
print("training spend time:" + str(spend_time))

filepath = "/media/ailab/songyoungtak/keras-master/examples/saved_models/cifar10_cnn_Global.hdf5"
model.save_weights(filepath)

start = time.time()
score = model.evaluate([X_test], Y_test, verbose=0)
end = time.time()
spend_time = end - start
spend_time = round(spend_time,2)
print("testing spend time:" + str(spend_time))
print('Test loss:', score[0])
print("score: ", score)


