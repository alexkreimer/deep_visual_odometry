import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pykitti
import functools
import argparse
import os

def _parse_function(image_path1, image_path2, label, target_size):
    images = []
    for image_path in [image_path1, image_path2]:
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_png(image_string)
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = tf.math.divide(image_decoded, tf.constant(255.0))
        images.append(tf.image.resize_images(image_decoded, target_size))
    return tf.concat(images, axis=2), label


class kitti_data:
    def __init__(self, size=None, basedir='/home/akreimer/work/dataset'):
        drive_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.data = {}
        for drive in drive_list:
            kitti_data = pykitti.odometry(basedir, drive)
            _size = size or len(kitti_data.poses)
            labels = []
            for i in range(_size-1):
                T = np.dot(np.linalg.inv(kitti_data.poses[i]), kitti_data.poses[i+1])
                label = np.linalg.norm(T[:3, 3])
                labels.append(label)
            filename1 = kitti_data.cam0_files[:_size-1]
            filename2 = kitti_data.cam0_files[1:_size]
            self.data[drive] = [labels, filename1, filename2, kitti_data.poses]

    def plot_drive(self, drive):
        poses = self.data[drive][3]
        origins = []
        for pose in poses:
            origins.append(np.dot(pose[:3, :], np.array([0, 0, 0, 1]).reshape(-1, 1)))
        origins = np.concatenate(origins, axis=1).T
        fig, ax = plt.subplots(1, 3)
        axis_names = 'xyz'
        for axis, _ax in enumerate(ax):
            _ax.plot(origins[:, axis], label=axis_names[axis])
            _ax.legend()

        fig, ax = plt.subplots(1)
        ax.plot(origins[:, 1], origins[:, 2])
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        plt.axis('equal')
        plt.grid('on')
        ax.legend()

        fig = plt.figure()
        labels = self.data[drive][0]
        plt.hist(labels)
        print('mean: {}, std: {}'.format(np.mean(labels), np.std(labels)))
        import ipdb; ipdb.set_trace()

def create_dataset(target_size, test_split=.1, val_split=.1, size=None,
        basedir='/home/akreimer/work/dataset'):
    drive = '04'
    kitti_data = pykitti.odometry(basedir, drive)
   
    size = size or len(kitti_data.poses)

    labels = []
    for i in range(size-1):
        T = np.dot(np.linalg.inv(kitti_data.poses[i]), kitti_data.poses[i+1])
        label = np.linalg.norm(T[:3, 3])
        labels.append(label)
    filename1 = kitti_data.cam0_files[:size-1]
    filename2 = kitti_data.cam0_files[1:size]

    train_val_size = int((1 - test_split)*size)
    train_size = int((1 - val_split)*train_val_size)

    assert train_val_size > 0
    assert train_size > 0

    res = []
    slices = [slice(0, train_size), slice(train_size, train_val_size), slice(train_val_size, size)]
    for _slice in slices:
        _f1, _f2, _l = filename1[_slice], filename2[_slice], labels[_slice]
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(_f1), tf.constant(_f2),
            tf.constant(_l)))
        dataset = dataset.map(functools.partial(_parse_function, target_size=target_size))
        res.append((dataset, len(_f1), _l))
    return res

def keras_model(target_size):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', name='1_conv2d', input_shape=(target_size[0], target_size[1], 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

#    model.add(Conv2D(64, (3, 3), padding='same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(64, (3, 3)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
#
#    sgd = keras.optimizers.SGD()
#    model.compile(optimizer=sgd, loss="mean_squared_error",metrics=['mae'])
#
#    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
#    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
#
#    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0)
#
#    model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
#            validation_data=valid_generator,
#            validation_steps=STEP_SIZE_VALID,
#	    epochs=100, callbacks=[tensorboard]
#	    )
#    model_json = model.to_json()
#    with open("model_{}.json".format(sha), "w") as json_file:
#	json_file.write(model_json)
#    # serialize weights to HDF5
#    model.save_weights("model_{}.h5".format(sha))
#    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--kuku', action='store_true')
    args = parser.parse_args()

    num_epochs = 500
    batch_size = 10
    target_size = np.array([613, 185])/2
    target_size = target_size.astype(int)
    dataset = create_dataset(target_size=target_size, size=20)

    (train_set, train_set_size, train_labels), (val_set, val_set_size, _), (test_set, test_set_size, test_labels) = dataset

    train_epoch_size = int(train_set_size/batch_size)
    train_set = train_set.batch(batch_size)
    train_set = train_set.repeat(num_epochs*train_epoch_size)

    val_epoch_size = int(val_set_size/batch_size)
    val_epoch_size = 1 if val_epoch_size == 0 else val_epoch_size
    val_set = val_set.batch(batch_size)
    val_set = val_set.repeat(num_epochs*val_epoch_size)

    model = keras_model(target_size)
    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    checkpoint_path = './train_1/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if args.kuku:
        dataset = kitti_data()
        dataset.plot_drive('04')
        os.exit(0)

    if args.test:
        test_epoch_size = test_set_size
        test_set = test_set.batch(1)
        test_set = test_set.repeat(test_epoch_size)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest_checkpoint)
        loss, mae = model.evaluate(test_set.make_one_shot_iterator(), steps=test_epoch_size,
                verbose=1)
        print('Trained model loss: {:.2f}m'.format(mae))
        labels_predicted = model.predict(test_set.make_one_shot_iterator(), steps=test_epoch_size,
                verbose=1)

        delta = labels_predicted.ravel() - np.array(test_labels)
        plt.plot(delta)
        plt.show()
    else:
        tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_grads=True, write_graph=True, write_images=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)
        model.fit(train_set.make_one_shot_iterator(), steps_per_epoch=train_epoch_size, 
                validation_data=val_set.make_one_shot_iterator(), validation_steps=val_epoch_size,
                epochs=num_epochs, verbose=1, callbacks=[checkpoint_callback, tb_callback])
        model.save('_test.hd5')
