import os
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import time
from sklearn import svm
import pickle

if __name__ == "__main__":

    outputs_path = '../outputs'
    model_filename = "model.pkl"

    fne_val = np.load(os.path.join(outputs_path, "fne_val.npy"))
    print('Done loading extracted features of validation set')
    train_labels = np.load(os.path.join(outputs_path, "train_labels.npy"))
    val_labels = np.load(os.path.join(outputs_path, "val_labels.npy"))

    with open(os.path.join(outputs_path, model_filename), 'rb') as file:
        pickle_model = pickle.load(file)

    start_time = time.time()
    # Test SVM with the validation set.
    predicted_labels = pickle_model.predict(fne_val)
    print('Done testing SVM on extracted features of validation set')
    predict_time = time.time() - start_time
    start_time = time.time()
    print(f'Feature extraction (validation) time: {predict_time}', flush=True)
    print('\n-----\n')
    np.save(os.path.join(outputs_path, 'predictions.npy'), predicted_labels)

    # Print results
    print(classification_report(val_labels, predicted_labels), flush=True)
    print(confusion_matrix(val_labels, predicted_labels), flush=True)