import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['LD_LIBRARY_PATH'] = r'/usr/local/cuda-10.0/lib64/'
from keras.models import *
from keras.layers import *
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib

# Some helper functions to check if GPU is available
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print('Available devices', get_available_devices())
print('GPU device names', tf.test.gpu_device_name())
#print('List all GPUs:', tf.config.list_physical_devices('GPU'))

#Checks if CUDA is correclt built
print('GPU available:', tf.test.is_gpu_available())
print('Tensorflow built with CUDA:', tf.test.is_built_with_cuda())

def unet_model(im_sz=288, n_channels=4, n_filters_start=16, growth_factor=2, kernel_size=(5, 5), droprate = 0.25):
    """
     Defines a keras sequential U-Net model inspired by https://github.com/Skii3/3D-cnn-denoise-Unet-keras. The model of
      4 downsampling and 4 upsampling layers. The output dimensions are identical to the input dimensions.

     Parameters
     __________
     im_sz: str
         size(shape) of the input image/array. Image has to be square shaped. Default is 288
     n_channels: int
         number of input channels (3rd dimension of the input array). Default is 4.
     n_filters_start: int
         number of filters at the start. Deedfault is 16.
     growth_factor: int
         growth factor of the number of filters. Is applied after each downsampling layer. Default is 2
     kernel_size: int
         size of the filfter kernels. Default is 5,5
     droprate: float
         droprate of the Dropout Layer (percentage of Neurons dropped). Float number between 0-1

     Returns
     _______
     model: Instance if keras model class
     """

    #First downsampling block. Each block consists of 2 convolutional layers, one maxpooling layer and one dropout layer
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = Conv2D(n_filters, kernel_size,  activation=LeakyReLU(alpha=0.1), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    # Second downsampling block
    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    # 3rd downsampling block
    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    # 4th downsampling block
    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4 = Dropout(droprate)(pool4)

    # bottom block. Consists only of 2 concolutional layers
    n_filters //= growth_factor
    conv5 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv5)

    # First upsampling block. Each block consists of 2 convolutional layers, one transpose convolutional layer and a concetanation to the convolutional layer of the corresponding downsampling block
    n_filters //= growth_factor
    up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv6)

    # 2nd upsampling block
    n_filters //= growth_factor
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv7)

    # 3rd upsampling block
    n_filters //= growth_factor
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv8)

    # 4th upsampling block
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters, kernel_size, activation=LeakyReLU(alpha=0.01), padding='same', kernel_initializer='he_normal')(conv9)
    output = Conv2D(1, (1, 1), activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    #output = Lambda(squeeze_lastaxes_operator, output_shape=squeeze_lastaxes_shape)(conv10)

    # First downsampling block. Each block consists of 2 convolutional layers, one maxpooling layer and one dropout layer
    model = Model(input=inputs, output=output)
    opt = keras.optimizers.Adam(lr=5e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=opt, loss='logcosh', metrics=['logcosh'])
    model.summary()

    return model


