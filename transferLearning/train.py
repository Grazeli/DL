import os
from keras import applications
import keras.backend as K
from fne import full_network_embedding
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import time
from sklearn import svm
import pickle

if __name__ == '__main__':
    """
    Load the features already computed, train a model and print results on validation set
    """
    outputs_path = '../outputs'

    fne_train = np.load(os.path.join(outputs_path, "fne_train.npy"))
    fne_val = np.load(os.path.join(outputs_path, "fne_val.npy"))
    stats = np.load(os.path.join(outputs_path, "stats.npy"))
    train_labels = np.load(os.path.join(outputs_path, "train_labels.npy"))
    val_labels = np.load(os.path.join(outputs_path, "val_labels.npy"))

    start_time = time.time()
    # Train SVM with the obtained features.
    clf = svm.LinearSVC()
    clf.fit(X=fne_train, y=train_labels)
    print('Done training SVM on extracted features of training set')
    training_time = time.time() - start_time
    start_time = time.time()
    print(f'Training time: {training_time}', flush=True)
    print('\n-----\n')

    model_filename = "model.pkl"
    with open(os.path.join(outputs_path, model_filename), 'wb') as file:
        pickle.dump(clf, file)
    print('Model saved successfully', flush=True)

    # Test SVM with the validation set.
    predicted_labels = clf.predict(fne_val)
    print('Done testing SVM on extracted features of validation set')
    predict_time = time.time() - start_time
    start_time = time.time()
    print(f'Feature extraction (validation) time: {predict_time}', flush=True)
    print('\n-----\n')
    np.save(os.path.join(outputs_path, 'predictions.npy'), predicted_labels)

    # Print results
    print(classification_report(val_labels, predicted_labels), flush=True)
    print(confusion_matrix(val_labels, predicted_labels), flush=True)
