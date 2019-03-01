import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pykitti

def _parse_function(image_path1, image_path2, label):
    images = []
    for image_path in [image_path1, image_path2]:
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_png(image_string)
        images.append(tf.image.resize_images(image_decoded, [613, 185]))
    return tf.concat(images, axis=2), label

def create_dataset(basedir='/home/akreimer/work/dataset'):
    drive = '04'
    kitti_data = pykitti.odometry(basedir, drive)
    
    labels = tf.constant([np.linalg.norm(T[:3, 3]) for T in kitti_data.poses[:-1]])
    filename1 = tf.constant(kitti_data.cam0_files[:-1])
    filename2 = tf.constant(kitti_data.cam0_files[1:])

    dataset = tf.data.Dataset.from_tensor_slices((filename1, filename2, labels))
    dataset = dataset.map(_parse_function)
    return dataset, len(kitti_data.poses)-1

def keras_model(target_size):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding='same', input_shape=(target_size[0], target_size[1], 2)))
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

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

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
    num_epochs = 500
    batch_size = 10
    dataset, size = create_dataset()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    target_size = [613, 185]
    model = keras_model(target_size)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['mae'])
    model.fit(dataset.make_one_shot_iterator(), verbose=1, steps_per_epoch=size//batch_size, epochs=num_epochs)
