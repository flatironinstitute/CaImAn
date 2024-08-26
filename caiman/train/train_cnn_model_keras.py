import numpy as np
import os
import keras 
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense 
from keras.models import save_model, load_model 
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.image_preprocessing_keras import ImageDataGenerator

os.environ["KERAS_BACKEND"] = "torch"

def cnn_model_keras(input_shape, num_classes): 
    sequential_model = keras.Sequential([   
        Input(shape=input_shape, dtype="float32"), 
        Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"), 
        Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"), 
        MaxPooling2D(pool_size=(2, 2)), 
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), 
        padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"),  
        MaxPooling2D(pool_size=(2, 2)), 
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=512, activation="relu"),
        Dropout(rate=0.5),
        Dense(units=num_classes, activation="relu"),
    ])
    return sequential_model 

def save_model_keras(model, name: str):
    model_name = os.path.join(caiman_datadir(), 'model', name)
    model_path = model_name + ".keras"
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    return model_path 

def load_model_keras(model_path: str):
    loaded_model = load_model(model_path)
    print('Load trained model at %s ' % model_path)
    return loaded_model  

if __name__ == "__main__": 
    batch_size = 128
    num_classes = 2
    epochs = 5000
    test_fraction = 0.25
    augmentation = True
    img_rows, img_cols = 50, 50 # input image dimensions

    with np.load('/mnt/ceph/data/neuro/caiman/data_minions/ground_truth_components_curated_minions.npz') as ld:
        all_masks_gt = ld['all_masks_gt']
        labels_gt = ld['labels_gt_cur']
    
    x_train, x_test, y_train, y_test = train_test_split(
    all_masks_gt, labels_gt, test_size=test_fraction)

    class_weight = cw.compute_class_weight(class_weight='balanced', 
                                            classes=np.unique(y_train), y=y_train)

    if keras.config.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    cnn_model_cifar = keras_cnn_model_cifar(input_shape, num_classes)
    cnn_model_cifar.summary()

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)
    cnn_model_cifar.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy']) #don't need this 
    
    #Augmentations 

    score = cnn_model_cifar.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])