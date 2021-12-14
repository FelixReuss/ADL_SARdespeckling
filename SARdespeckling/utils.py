import os
import numpy as np
import tifffile as tiff
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Generates data for Keras Sequence based data generator. Suitable for building data generator for training and prediction.
    Inherited from the Keras 'Sequence' class.
    """
    def __init__(self, list_IDs, labels, labels_ref, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=1, shuffle=True):
        """
        Constructor of DataGenerator.

        Parameters
        __________
        list_IDs: list
            list of all 'label' ids to use in the generator
        labels: list
            list of image labels (file names)
        labels_ref: list
            list of image labels of reference data(file names)
        image_path: str
            path to images location
        mask_path: str
            path to masks location
        to_fit: boolean
            True to return X and y, False to return X only
        batch_size: int
            batch size at each iteration
        dim: tuple
            tuple indicating image dimension
        n_channels: int
            number of image channels
        n_classes: int
            number of output masks
        shuffle: boolean
            True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.labels_ref = labels_ref
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch

        Returns
        _______
        number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        Parameters
        __________
        index: int
            index of the batch

        Returns
        _______
        Y, X: array
            number of batches per epoch X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """
        Generates data containing batch_size images

        Parameters
        __________
        list_IDs_temp: list
            list of label ids to load

        Returns
        _______
        X: array
            batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_array = self._load_grayscale_image(os.path.join(self.image_path,f"{self.labels}_{ID}.tif"))
            #image_array = self._convert2lin(image_array)
            image_array = self._standardize(image_array)
            image_array = np.expand_dims(image_array, axis=2)
            X[i,] = image_array

        #X = self._standardize(X)
        return X

    def _standardize(self, X):
        """
        Standardizes a batch of data using scikit standarizer function

        Parameters
        __________
        X: array
            one batch of data

        Returns
        _______
        X_stand: array
            batch with standardized data
        """
        scaler = StandardScaler()
        scaler.fit(X)
        X_stand = scaler.transform(X)
        return X_stand

    def _generate_y(self, list_IDs_temp):
        """
        Generates data containing batch_size masks

        Parameters
        __________
        list_IDs_temp: list
            list of label ids to load

        Returns
        _______
        Y: array
            array containing the mask data
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_array = self._load_grayscale_image(os.path.join(self.mask_path,f"{self.labels_ref}_{ID}.tif"))
            y[i,] = self._convert2lin(image_array)
            y[i,] = self._standardize(image_array)
        y = y[..., None]

        return y

    def _convert2lin(self, db_array):
        """
        converts linear units to db

        Parameters
        __________
        db_array: array
            input array in logarithmic units

        Returns
        _______
        linear_array: array in linear units

        """
        linear_array = 10. ** (db_array / 10.)

        return linear_array

    def _convert2db(self, linear_array):
        """
        converts linear units to db

        Parameters
        __________
        linear_array: array
            input array in linear units

        Returns
        _______
        db_array: array
            array with logarithmic units
        """
        db_array = 10. * np.log10(linear_array)

        return db_array

    def _load_grayscale_image(self, image_path):
        """
        Load grayscale image

        Parameters
        __________
        image_path: str
            path to image to load

        Returns
        _______
        img: array
            loaded image
        """
        img = tiff.imread(image_path)
        return img