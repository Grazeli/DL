import os
import numpy as np
import tensorflow as tf
import cv2
from keras import applications
import keras.backend as K
from sklearn.decomposition import PCA

def full_network_embedding(model, image_paths, batch_size, input_tensor, target_tensors, input_reshape, pca=None, stats=np.empty((0, 0))):
    ''' 
    Generates the Full-Network embedding[1] of a list of images using a pre-trained
    model (input parameter model) with its computational graph loaded. Tensors used 
    to compose the FNE are defined by target_tensors input parameter. The input_tensor
    input parameter defines where the input is fed to the model.

    By default, the statistics used to standardize are the ones provided by the same 
    dataset we wish to compute the FNE for. Alternatively these can be passed through
    the stats input parameter.

    This function aims to generate the Full-Network embedding in an illustrative way.
    We are aware that it is possible to integrate everything in a tensorflow operation,
    however this is not our current goal.

    [1] https://arxiv.org/abs/1705.07706
   
    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to generate the FNE for.
        batch_size (int): Number of images to be concurrently computed on the same batch.
        input_tensor (str): Name of tensor from model where the input is fed to
        target_tensors (list(str)): List of tensor names from model to extract features from.
        input_reshape (tuple): A tuple containing the desired shape (height, width) used to resize the image.
        stats (2D ndarray): Array of feature-wise means and stddevs for standardization.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
       2D ndarry: Mean and stddev per feature. Of shape <2,num_feats>
    '''
    #Print available layers
    #print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

    # Prepare output variable
    len_features = 0
    for tensor_name in target_tensors:
        t_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
        len_features += t_tensor.get_shape().as_list()[-1]
    features = np.empty((len(image_paths), len_features))

    # Prepare tensors to capture
    x0 = tf.get_default_graph().get_tensor_by_name(input_tensor)
    tensorOutputs = []
    for tname in target_tensors:
        t = tf.get_default_graph().get_tensor_by_name(tname)
        tensorOutputs.append(t)
    # Extract features
    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        for idx in range(0, len(image_paths), batch_size):
            batch_images_path = image_paths[idx:idx+batch_size]
            img_batch = np.zeros((len(batch_images_path), *input_reshape, 3), dtype=np.float32)
            for i, img_path in enumerate(batch_images_path):
                cv_img = cv2.imread(img_path)
                try:
                    cv_img_resize = cv2.resize(cv_img, input_reshape)
                    img_batch[i] = applications.vgg16.preprocess_input(np.asarray(cv_img_resize, dtype=np.float32))
                except Exception as excep:
                    print(excep)
                    print(img_path)

            feature_vals = sess.run(tensorOutputs, feed_dict={x0: img_batch})
            features_current = np.empty((len(batch_images_path), 0))
            for feat in feature_vals:
                #If its not a conv layer, add without pooling
                if len(feat.shape) != 4:
                    features_current = np.concatenate((features_current, feat), axis=1)
                    continue
                #If its a conv layer, do SPATIAL AVERAGE POOLING
                pooled_vals = np.mean(np.mean(feat, axis=2), axis=1)
                features_current = np.concatenate((features_current, pooled_vals), axis=1)
            # Store in position
            features[idx:idx+len(batch_images_path)] = features_current.copy()
            print(f"{idx}/{len(image_paths)} ({round(idx/len(image_paths),2)})", flush=True)
    # STANDARDIZATION STEP
    # Compute statistics if needed
    if len(stats) == 0:
        stats = np.zeros((2, len_features))
        stats[0, :] = np.mean(features, axis=0)
        stats[1, :] = np.std(features, axis=0)
    # Apply statistics, avoiding nans after division by zero
    features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features), where=stats[1] != 0)
    if len(np.argwhere(np.isnan(features))) != 0:
        raise Exception('There are nan values after standardization!')
    # DISCRETIZATION STEP
    th_pos = 0.15
    th_neg = -0.25
    if not pca:
        pca = PCA(n_components=0.65)
        features = pca.fit_transform(features)
        print(f"Number of PCA components: {pca.n_components_}")
        return features, stats, pca
    features = pca.transform(features)
    #features[features > th_pos] = 1
    #features[features < th_neg] = -1
    #features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

    # Return
    return features, stats
