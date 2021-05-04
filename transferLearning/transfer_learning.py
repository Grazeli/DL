from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt
import time

# Plot the training and validation loss + accuracy
def plot_training(history, name):

    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0, 1)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Training and validation accuracy')
    plt.savefig(name + '_accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Training and validation loss')
    plt.savefig(name + '_loss.pdf')
    plt.close()

img_width, img_height = 256, 256
train_data_dir = "../data/train/"
validation_data_dir = '../data/validation/'

nb_train_samples = 20300
nb_validation_samples = 1450
batch_size = 128
target_classes = 29

### Transfer learning ###
name = 'test_transfer_learning'
epochs = 10
opti = optimizers.SGD(lr=0.0001, momentum=0.9)

imported_model = applications.VGG16(weights = 'imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze layers
for layer in imported_model.layers[:5]:
    layer.trainable = False

# Custom layers
model = Sequential()
model.add(imported_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(29, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(
    rescale = 1./255.0
)

val_datagen = ImageDataGenerator(
    rescale = 1./255.0
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    shuffle = False,
    class_mode = 'categorical'
)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

start_time = time.time()

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    callbacks = [early]
)

fitting_time = time.time() - start_time
print('\n-----\n')
print(f'Training time: {fitting_time}')

#Saving model and weights
from keras.models import model_from_json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-" + name + ".hdf5"
model.save_weights(weights_file,overwrite=True)

plot_training(history, name)