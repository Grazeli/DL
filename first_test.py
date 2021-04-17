from __future__ import division
import keras
import tensorflow as tf
import numpy as np
import import_dataset
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

# print('Using Keras version', keras.__version__)

print('Start')
time_start = time.time()

# import_dataset.create_dataframes_csv()

train_size = 20300
val_size = 1450
test_size = 15657
batch_size = 128

img_rows, img_cols, channels = 256, 256, 3
input_shape = (img_rows, img_cols, channels)

# x_train, y_train, x_validation, y_validation =  import_dataset.import_dataset()
train_generator, val_generator = import_dataset.data_generators(batch_size)

print('End import dataset')

print('----')

time_end = time.time()

print(f'End loading time: {time_end - time_start}')

# Adapt the labels to the one-hot vector syntax required by the softmax

# y_train = np_utils.to_categorical(y_train, 29)
# y_validation = np_utils.to_categorical(y_validation, 29)

# Reshape for input
# x_train = x_train.reshape(len(y_train), img_rows, img_cols, channels)
# x_validation = x_validation.reshape(len(y_validation), img_rows, img_cols, channels)

# Define the NN architecture

# Two hidden layers
model = Sequential()
model.add(Conv2D(8, 3, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(29, activation='softmax'))

# Model visualization
# We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
# plot_model(model, to_file='model.png', show_shapes=true)

# Compile the NN
opt = tf.keras.optimizers.Adam(
    lr=0.0001,
)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Start training
# history = model.fit(x_train,y_train,validation_data=(x_validation, y_validation), batch_size=128,epochs=20)
train_steps = (train_size // batch_size) + 1
val_steps = (val_size // batch_size) + 1
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=20,
    workers=40
)

# Saving model and weights
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
weights_file = "weights-test1.hdf5"
model.save_weights(weights_file, overwrite=True)

# Loading model and weights
#json_file = open('model.json', 'r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)

# Confusion Matrix

# Compute probabilities
y_pred = model.predict_generator(val_generator)
# Assign most probable label
y_pred = np.argmax(y_pred, axis=1)
# Plot statistics
# print( 'Analysis of results' )
# target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(confusion_matrix(val_generator.classes, y_pred))
print(classification_report(val_generator.classes, y_pred))


# Evaluate the model with test set
# score = model.evaluate(x_validation, y_validation, verbose=0)
# print('validation loss:', score[0])
# print('validation accuracy:', score[1])

##Store Plots
matplotlib.use('Agg')
# Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('test1_cnn_accuracy.pdf')
plt.close()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('test1_cnn_loss.pdf')

