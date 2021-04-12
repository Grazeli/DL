from __future__ import division
import keras
import tensorflow as tf
import numpy as np
import importDataset
import sklearn
import time
print( 'Using Keras version', keras.__version__)

print('Start')
time_start = time.time()

x_train, y_train, x_validation, y_validation =  importDataset.importDataset()

print('End import dataset')

x_train = np.array(x_train).astype('float32')
x_train = x_train / 255

x_validation = np.array(x_validation).astype('float32')
x_validation = x_validation / 255

y_train = np.array(y_train)
y_validation = np.array(y_validation)

print(x_train.shape)
print(len(y_train))
print(x_validation.shape)
print(len(y_validation))

print('----')

time_end = time.time()

print(f'End loading time: {time_end - time_start}')

#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 29)
y_validation = np_utils.to_categorical(y_validation, 29)

img_rows, img_cols, channels = 256, 256, 3
input_shape = (img_rows, img_cols, channels)
#Reshape for input
#x_train = x_train.reshape(len(y_train), img_rows, img_cols, channels)
#x_validation = x_validation.reshape(len(y_validation), img_rows, img_cols, channels)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
model = Sequential()
model.add(Conv2D(8, 3, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(29, activation=(tf.nn.softmax)))

#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(model, to_file='model.png', show_shapes=true)

#Compile the NN
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#Start training
history = model.fit(x_train,y_train,validation_data=(x_validation, y_validation), batch_size=128,epochs=20)

#Evaluate the model with test set
#score = model.evaluate(x_validation, y_validation, verbose=0)
#print('validation loss:', score[0])
#print('validation accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('test1_fnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('test1_fnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = model.predict(x_validation)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
#print( 'Analysis of results' )
#target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#print(classification_report(np.argmax(y_validation,axis=1), y_pred,target_names=target_names))
#print(confusion_matrix(np.argmax(y_validation,axis=1), y_pred))

#Saving model and weights
#from keras.models import model_from_json
#model_json = model.to_json()
#with open('model.json', 'w') as json_file:
#        json_file.write(model_json)
#weights_file = "weights-test1_"+str(score[1])+".hdf5"
#model.save_weights(weights_file,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)