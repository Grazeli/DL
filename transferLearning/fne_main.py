import os
from keras import applications
import keras.backend as K
from fne import full_network_embedding
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import time

if __name__ == '__main__':
    # This shows an example of calling the full_network_embedding method using
    # the VGG16 architecture pretrained on ILSVRC2012 (aka ImageNet), as
    # provided by the keras package. Using any other pretrained CNN
    # model is straightforward.

    # Load model
    img_width, img_height = 224, 224
    applications.VGG16(weights = "imagenet", include_top=True, input_shape = (img_width, img_height, 3))
    graph_d = K.get_session().graph_def

    # Define input and target tensors, that is where we want
    #to enter data, and which activations we wish to extract
    input_tensor = 'input_1:0'
    target_tensors = ['block1_conv1/Relu:0','block1_conv2/Relu:0','block2_conv1/Relu:0','block2_conv2/Relu:0','block3_conv1/Relu:0','block3_conv2/Relu:0','block3_conv3/Relu:0','block4_conv1/Relu:0','block4_conv2/Relu:0','block4_conv3/Relu:0','block5_conv1/Relu:0','block5_conv2/Relu:0','block5_conv3/Relu:0','fc1/Relu:0','fc2/Relu:0']

    start_time = time.time()

    # Define data splits
    #Train set
    train_path = '../data/train/'
    train_images = []
    train_labels = []
    #Use a subset of classes to speed up the process. -1 uses all classes.
    num_classes = -1
    remaining_classes = num_classes
    for train_dir in os.listdir(train_path):
        train_dir_path = os.path.join(train_path,train_dir)
        for train_img in os.listdir(train_dir_path):
            train_images.append(os.path.join(train_dir_path,train_img))
            train_labels.append(train_dir)
        remaining_classes-=1
        print(train_dir, "loaded. ", remaining_classes, "classes remaining (train)", flush=True)
        if remaining_classes==0:
            break
    #Validation set
    test_path = '../data/validation/'
    test_images = []
    test_labels = []
    remaining_classes = num_classes
    for test_dir in os.listdir(test_path):
        test_dir_path = os.path.join(test_path,test_dir)
        for test_img in os.listdir(test_dir_path):
            test_images.append(os.path.join(test_dir_path,test_img))
            test_labels.append(test_dir)
        remaining_classes-=1
        print(test_dir, "loaded. ", remaining_classes, "classes remaining (validation)", flush=True)
        if remaining_classes==0:
            break
    
    print('Total train images:',len(train_images),' with their corresponding',len(train_labels),'labels')
    print('Total validation images:',len(test_images),' with their corresponding',len(test_labels),'labels')
    loading_time = time.time() - start_time
    start_time = time.time()
    print(f'Data loading time: {loading_time}', flush=True)
    print('\n-----\n')
    outputs_path = '../outputs'
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    np.save(os.path.join(outputs_path, 'train_labels.npy'), train_labels)
    np.save(os.path.join(outputs_path, 'val_labels.npy'), test_labels)


    #Parameters for the extraction procedure
    batch_size = 128
    input_reshape = (img_width, img_height)
    # Call FNE method on the train set
    fne_features_train, fne_stats_train = full_network_embedding(graph_d, train_images, batch_size, input_tensor, target_tensors, input_reshape)
    print('Done extracting features of training set. Embedding size:', fne_features_train.shape)
    extraction_time_train = time.time() - start_time
    start_time = time.time()
    print(f'Feature extraction (train) time: {extraction_time_train}', flush=True)
    print('\n-----\n')

    # Store output
    np.save(os.path.join(outputs_path, 'fne_train.npy'), fne_features_train)
    np.save(os.path.join(outputs_path, 'stats.npy'), fne_stats_train)

    # To load output do:
    # fne = np.load('fne.npy')
    # fne_stats = np.load('stats.npy')

    # Call FNE method on the validation set, using stats from training
    fne_features_val, fne_stats_val = full_network_embedding(graph_d, test_images, batch_size, input_tensor, target_tensors, input_reshape, stats=fne_stats_train)
    print('Done extracting features of validation set')
    extraction_time_val = time.time() - start_time
    start_time = time.time()
    print(f'Feature extraction (validation) time: {extraction_time_val}', flush=True)
    print('\n-----\n')

    # Store output
    np.save(os.path.join(outputs_path, 'fne_val.npy'), fne_features_val)
    np.save(os.path.join(outputs_path, 'stats_val.npy'), fne_stats_val)


    from sklearn import svm
    #Train SVM with the obtained features.
    clf = svm.LinearSVC()
    clf.fit(X=fne_features_train, y=train_labels)
    print('Done training SVM on extracted features of training set')
    training_time = time.time() - start_time
    start_time = time.time()
    print(f'Training time: {training_time}', flush=True)
    print('\n-----\n')

    #Test SVM with the validation set.
    predicted_labels = clf.predict(fne_features_val)
    print('Done testing SVM on extracted features of validation set')
    predict_time = time.time() - start_time
    start_time = time.time()
    print(f'Feature extraction (validation) time: {predict_time}', flush=True)
    print('\n-----\n')
    np.save(os.path.join(outputs_path, 'predictions.npy'), predicted_labels)

    #Print results
    print(classification_report(test_labels, predicted_labels), flush=True)
    print(confusion_matrix(test_labels, predicted_labels), flush=True)

