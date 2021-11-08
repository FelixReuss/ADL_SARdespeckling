import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
import tensorflow as tf
import keras


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())
print(tf.test.gpu_device_name())

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

def squeeze_lastaxes_operator(x4d) :
    shape = tf.shape(x4d) # get dynamic tensor shape
    x3d = tf.reshape(x4d, [shape[0], shape[1], shape[2] * shape[3]])
    return x3d

def squeeze_lastaxes_shape(x4d_shape) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if (None in [in_rows, in_cols]) :
        output_shape = (in_batch, None, in_filters)
    else:
        output_shape = (in_batch, in_rows, in_cols)
    return output_shape

# add Dropout or Batchnormalization
def unet_model(im_sz=288, n_channels=4, n_filters_start=18, growth_factor=2, kernel_size=(5, 5)):
    droprate = 0.25
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = Conv2D(n_filters, kernel_size,  activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4 = Dropout(droprate)(pool4)

    n_filters //= growth_factor
    conv5 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv5)

    n_filters //= growth_factor
    up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv6)

    n_filters //= growth_factor
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv7)

    n_filters //= growth_factor
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv9)
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)
    #output = Lambda(squeeze_lastaxes_operator, output_shape=squeeze_lastaxes_shape)(conv10)

    model = Model(input=inputs, output=output)
    opt = keras.optimizers.Adam(lr=5e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=opt, loss='logcosh', metrics=['logcosh'])
    model.summary()

    return model


