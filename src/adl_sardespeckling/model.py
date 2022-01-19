import os
import keras
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['LD_LIBRARY_PATH'] = r'/usr/local/cuda-10.0/lib64/'
from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, concatenate

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
    conv1 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    actv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv1)
    actv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(actv1)
    pool1 = Dropout(droprate)(pool1)

    # Second downsampling block.
    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(pool1)
    actv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv2)
    actv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(actv2)
    pool2 = Dropout(droprate)(pool2)

    # Thrd downsampling block.
    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(pool2)
    actv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv3)
    actv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(actv3)
    pool3 = Dropout(droprate)(pool3)

    # Fourth downsampling block.
    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(pool3)
    actv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv4)
    actv4 = LeakyReLU(alpha=0.1)(conv4)
    drop4 = Dropout(droprate)(actv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4 = Dropout(droprate)(pool4)

    # Bottom block
    n_filters //= growth_factor
    conv5 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(pool4)
    actv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv5)
    actv5 = LeakyReLU(alpha=0.1)(conv5)

    # First upsamplnig block. Each upsampling block consists of a residual layer, resizing convolutions and convolutions
    n_filters //= growth_factor
    up6 = UpSampling2D(size=2, interpolation='bilinear')(actv5)
    conv6 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(up6)
    actv6 = LeakyReLU(alpha=0.1)(conv6)
    conc6 = concatenate([actv6, actv4])
    merge6 = concatenate([drop4, conc6], axis=3)
    conv6 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(merge6)
    actv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv6)
    actv6 = LeakyReLU(alpha=0.1)(conv6)

    # Second upsamplnig block.
    n_filters //= growth_factor
    up7 = UpSampling2D(size=2, interpolation='bilinear')(actv6)
    conv7 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(up7)
    actv7 = LeakyReLU(alpha=0.1)(conv7)
    conc7 = concatenate([actv7, actv3])
    merge7 = concatenate([actv3, conc7], axis=3)
    conv7 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(merge7)
    actv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv7)
    actv7 = LeakyReLU(alpha=0.1)(conv7)

    # Third upsamplnig block.
    n_filters //= growth_factor
    up8 = UpSampling2D(size=2, interpolation='bilinear')(actv7)
    conv8 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(up8)
    actv8 = LeakyReLU(alpha=0.1)(conv8)
    conc8 = concatenate([actv8, actv2])
    merge8 = concatenate([actv2, conc8], axis=3)
    conv8 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(merge8)
    actv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv8)
    actv8 = LeakyReLU(alpha=0.1)(conv8)

    # Fourth upsamplnig block.
    up9 = UpSampling2D(size=2, interpolation='bilinear')(actv8)
    conv9 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(up9)
    actv9 = LeakyReLU(alpha=0.1)(conv9)
    conc9 = concatenate([actv9, actv1])
    merge9 = concatenate([actv1, conc9], axis=3)
    conv9 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(merge9)
    actv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(actv9)
    actv9 = LeakyReLU(alpha=0.1)(conv9)

    # Output layer
    output = Conv2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(actv9)

    # Compiling model
    model = Model(input=inputs, output=output)
    opt = keras.optimizers.Adam(lr=8e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=opt, loss='logcosh', metrics=['logcosh'])
    model.summary()

    return model


