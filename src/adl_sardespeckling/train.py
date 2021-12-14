from adl_sardespeckling.model import *
from adl_sardespeckling.utils import *
from random import shuffle
import tifffile
import os
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


os.environ['LD_LIBRARY_PATH']=r'/usr/local/cuda-10.0/lib64/'

def get_model(patch_size, n_channels, **kwargs):
    """
     Helper function creating a U-Net model instance with the given parameters

     Parameters
     __________
     patch_size: int
         shape of the input patches
     n_channels: int
         number of channels of the in input patches
     Keyword Arguments:
         n_filters_start (int): Number of filters at the fist Conv block
         growth_factor (int): Growth factor of the filters after ever downsampling block
         kernel_siz (int): Size of filter kernel
         droprate (float): drop rate of the Dropout layers (percentage of Neurons dropped)

     Returns
     _______
     unet_model: Instance of the Keras model class
     """
    return unet_model(im_sz=patch_size, n_channels=n_channels, **kwargs)

def train_model(image_path, reference_path, steps_per_epoch, batch_size, patch_size, n_channels, epochs, save_model=None):
    """
     Function to train Residual U-Net Depspekling Model

     Parameters
     __________
     image_path: str
         path to the input images
     reference_path: str
         path to the reference images
     steps_per_epoch: int
         number of steps per epoch. steps_per_epoch*batch_size should equal number of training samples
     bacth_size: int
         number of samples for each training step. steps_per_epoch*batch_size should equal number of training samples
     patch_size: int
         shape of the input patches
     n_channels: int
         number of channels of the in input patches
     epochs: int
         number of epochs to train the model on
     """

    #Name of the label image patches and reference image patches, currently hardcoded
    labels = 'train'
    labels_ref = 'reference'

    #Number of image_ids, currently hardcoded to 6000
    image_ids = list(range(0, 6000))

    #Shuffle the image ids to ensure random split
    shuffle(image_ids)

    #Split Image Ids in 0.8 training and 0.2 validation
    trainIds, valIds = train_test_split(image_ids, test_size=0.2, random_state=42)

    #Initialize training and validation generator
    training_generator = DataGenerator(trainIds, labels, labels_ref, image_path, reference_path, batch_size=batch_size, dim=(400, 400))
    validation_generator = DataGenerator(valIds, labels, labels_ref, image_path, reference_path, batch_size=batch_size, dim=(400, 400))

    #Initialize U-Net model
    model = get_model(patch_size, n_channels)

    #Fit the model
    model.fit_generator(training_generator, validation_data=validation_generator,
                                  steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save the model
    if save_model is not None:
        if not os.path.exists():
            os.mkdir(save_model)
        model.save(save_model)

def make_prediction(path2input, path2model, outpath):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     path2input: str
         path to the noisy input image to despeckle
     path2model: str
         path to the keras model. .hd5 or .h5 file expected
     outpath: str
         output path including file name to store the despeckled image
     """

    #Load the input data
    image_array = tiff.imread(path2input)

    #Scale input data to match with training data
    scaler = StandardScaler()
    scaler.fit(image_array)
    X_norm = scaler.transform(image_array)

    #Expand the dimensions to match requirements from model
    X_norm = np.expand_dims(X_norm, axis=2)
    X_norm = np.expand_dims(X_norm, axis=0)

    #Load the Keras U-Net model
    model = load_model(path2model)

    #Make the prediction
    pred_array = model.predict(X_norm)

    #Remove expandes dimensions
    pred_array = pred_array[0, :, :, 0]

    #Save prediction to file
    pred_array_unscaled = scaler.inverse_transform(pred_array)
    tifffile.imsave(r'//home/freuss/ADL/pred_2575_standard_tanh100_leaky_mse.tif', pred_array_unscaled)