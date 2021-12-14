from adl_sardespeckling.model import *
from adl_sardespeckling.utils import *
from random import shuffle
import matplotlib.pyplot as plt
import tifffile
import os
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


os.environ['LD_LIBRARY_PATH']=r'/usr/local/cuda-10.0/lib64/'

def get_model(patch_size, n_channels):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     input_array: str
         path to the input image
     size: int
         size of the filter kernel

     Returns
     _______
     out_array: array
         filtered output array
     """
    return unet_model(patch_size, patch_size, n_channels)

def train_model(image_path, reference_path, steps_per_epoch, save_model=None):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     input_array: str
         path to the input image
     size: int
         size of the filter kernel

     Returns
     _______
     out_array: array
         filtered output array
     """
    image_path = r"///home/freuss/ADL/inputdata/patches/train3"
    reference_path = r"///home/freuss/ADL/inputdata/patches/reference3"

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
    training_generator = DataGenerator(trainIds, labels, labels_ref, image_path, reference_path, dim=(400, 400))
    validation_generator = DataGenerator(valIds, labels, labels_ref, image_path, reference_path, dim=(400, 400))

    #Initialize U-Net model
    model = get_model(patch_size, n_channels)

    #Fit the model
    history = model.fit_generator(training_generator, validation_data=validation_generator,
                                  steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save the model
    if save_model is not None:
        if not os.path.exists():
            os.mkdir(save_model)
        model.save(save_model)

def make_prediction(path2input):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     input_array: str
         path to the input image
     size: int
         size of the filter kernel

     Returns
     _______
     out_array: array
         filtered output array
     """
    pass

if __name__ == '__main__':
    weights_path = 'weights'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    weights_path += '/unet_weights.hdf5'

    n_channels = 1
    # calcualte automatically CLASS_WEIGHTS = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=weight)
    epochs = 100
    steps_per_epoch = 16
    upconv = True
    patch_size = 400  # should divide by 16
    batch_size = 500

    # ------------Traiining-------------#
    image_path = r"///home/freuss/ADL/inputdata/patches/train3"
    mask_path = r"///home/freuss/ADL/inputdata/patches/reference3"
    labels = 'train'
    labels_ref = 'reference'
    image_ids = list(range(0, 6000))
    shuffle(image_ids)
    trainIds, valIds = train_test_split(image_ids, test_size=0.2, random_state=42)
    training_generator = DataGenerator(trainIds, labels, labels_ref, image_path, mask_path, dim=(400, 400))
    validation_generator = DataGenerator(valIds, labels, labels_ref,image_path, mask_path, dim=(400, 400))
    #test = training_generator._generate_X(trainIds)
    model = get_model(patch_size, n_channels)

    history = model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # ------------Evaluation-------------#
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='test loss ')
    plt.legend()
    plt.show()



    # ------------Prediction-------------#
    image_array = tiff.imread(r'//home/freuss/ADL/inputdata/patches/train3/train_2575.tif')
    scaler = StandardScaler()
    scaler.fit(image_array)
    X_norm = scaler.transform(image_array)
    X_norm = np.expand_dims(X_norm, axis=2)
    X_norm = np.expand_dims(X_norm, axis=0)
    pred_array = model.predict(X_norm)
    pred_array = pred_array[0,:, :,0]
    tifffile.imsave(r'//home/freuss/ADL/pred_2575_standard_unscaled_tanh100_leaky_mse.tif', pred_array)
    pred_array_unscaled = scaler.inverse_transform(pred_array)
    tifffile.imsave(r'//home/freuss/ADL/pred_2575_standard_tanh100_leaky_mse.tif', pred_array_unscaled)

    #Save the model
    model.save(r'//home/freuss/ADL/model1.h5')
    #Save the model
    #model.save(filepath)