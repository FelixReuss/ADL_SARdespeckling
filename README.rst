==================
ADL_SARdespeckling
==================

Python workflow to remove speckle nosie from C-Band SAR images

Description
===========

The basic workflow setup includes the following modules and tasks:

- preprocessing: preperation of the training data
	- add_speckle_noise: Function to add simulated speckle noise to SAR data
	- patch_extractor: Extracts image square shaped patches from input images and writes them to file
- model: Function defining the Residual U-Net Model
- train:
	- get_model: Helper function creating a U-Net model instance with the given parameters.
	- train_model: Function to train Residual U-Net Despeckling model using the provided parameters
	- make_prediction: Function to make prediction using the input path for a pretrained Residual U-Net Despeckling model
- postprocessing: postprocessing to evaluate the predictions and reconstruct predicted data
	- reconstruct_image: Function to merge patches to the extent of the original SAR scene and georeference it
	- diff_image: Function to calculte the difference in dB between to given images
- utils: Provides class to load and feed data to the model
	- DataGenerator: Class that generates data for Keras Sequence based data generator

Installation
============

To install *adl_sardespeckling*, you can create a conda environment, if no appropriate environment is in place yet:

.. code-block:: bash

    conda create --name adl_sardespeckling -c conda-forge python=3.6 gdal=3.0.2
    conda activate adl_sardespeckling


**ATTENTION**: It is expected that the existing environment has already a working GDAL installation. GDAL needs more OS
support and has more dependencies then other packages and can therefore not be installed solely via pip.
Please have a look at https://pypi.org/project/GDAL/ what requirements are needed. Thus, for a fresh setup, an
existing environment with `python=3.6` and `gdal=3.0.2` are expected.


The last installation step involves the setting of the required environmental variables for GDAL.

.. code-block:: bash

    export PROJ_LIB="[...]/miniconda/envs/adl_sardespeckling/share/proj"
    export GDAL_DATA="[...]/miniconda/envs/adl_sardespeckling/share/gdal"
	
	
Data
=========
As the input SAR images are very large and exceed the allowed file size, they are not added to the repository.

Model training
=========
Currently the workflow is only designed to be run in a Python IDE. In the final version, a command line interface will be supported.
To train the model on prepared data, use the function train_model of the train module and provided the required parameters. 


Model prediction
==============
Currently the workflow is only designed to be run in a Python IDE. In the final version, a command line interface will be supported.
To make predictions aka despeckle data, use the function make_predictions of the train module and provided the required parameters. 


Note
====
The workplan as well as a log book describing the current status of the proect, open issues and the next steps can be found here: https://github.com/FelixReuss/ADL_SARdespeckling/blob/main/ADL_Exercise2-1431542.pdf


This project has been set up using PyScaffold 3.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

