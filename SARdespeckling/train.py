from model import *
from utils import *
from random import shuffle
import matplotlib.pyplot as plt
import tifffile
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_model(patch_size, n_channels):
    return unet_model(patch_size, n_channels)

def make_prediction(path2input):
    pass

if __name__ == '__main__':
    weights_path = 'weights'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    weights_path += '/unet_weights.hdf5'

    n_channels = 1
    # calcualte automatically CLASS_WEIGHTS = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=weight)
    epochs = 25
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

    history = model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=16, epochs=epochs)

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
    image_array = np.expand_dims(image_array, axis=2)
    image_array = np.expand_dims(image_array, axis=0)
    pred_array = model.predict(image_array)
    pred_array = pred_array[0,:, :,0]
    tifffile.imsave(r'//home/freuss/ADL/pred_2575_unscaled.tif', pred_array)
    pred_array_unscaled = scaler.inverse_transform(pred_array)
    tifffile.imsave(r'//home/freuss/ADL/pred_2575.tif', pred_array_unscaled)

    #Save the model
    #model.save(filepath)