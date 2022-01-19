import os
import numpy as np
from keras.models import load_model
from veranda.io.geotiff import GeoTiffFile
from adl_sardespeckling.preprocessing import lee_filter
from geopathfinder.naming_conventions.sgrt_naming import SgrtFilename
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from equi7grid.equi7grid import Equi7Grid
from PIL import Image
from sklearn.preprocessing import StandardScaler

def despeckle_sar_image(input_path, output_path, path2model, overlay=0):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     input_path: str
         path to the input image
     output_path: int
         path where the output image will be saved
     path2model: str
        path to the keras model (h5 file type expected)
     overlay: int
        overlay in pixel between the patches
     """

    # Create SGRT Naming object from filename to extract meta data
    sgrt_name = SgrtFilename.from_filename(os.path.basename(input_path))

    # Define parameters for the geo reference
    grid = Equi7Grid(10)
    subgrid = sgrt_name.obj.grid_name
    tile_name = sgrt_name.obj.tile_name
    equi7tile = grid.create_tile(subgrid+'_'+tile_name)
    sref = equi7tile.core.projection.wkt
    gt = equi7tile.geotransform()

    # Read the input image as array
    image = GeoTiffFile(filepath=input_path, mode='r', n_bands=1, sref=sref,
                        geotrans=gt)
    image_array = image.read(band=1)
    rows, cols = image_array.shape

    # Fit scaler
    scaler_array = image_array.flatten()
    scaler_array = scaler_array.reshape(-1, 1)

    # Normalize the patch array
    scaler = StandardScaler()
    scaler.fit(scaler_array)
    del scaler_array

    # Initialize output array
    output_array = np.zeros([rows, cols])

    # Load the pre-trained despeckling model
    model = load_model(path2model)

    # Get the required input patch size
    row_size = model.layers[0].input_shape[1]
    col_size = model.layers[0].input_shape[1]

    # Get the row and column indexes for extracting the patches from the input image
    row_idxs = [i for i in range(0, rows, row_size-2*overlay)]
    col_idxs = [i for i in range(0, cols, col_size-2*overlay)]

    #Check for incomplete row and cols. Those need to be considered when looping over the image patches
    incomplete_rows = []
    incomplete_cols = []

    if row_idxs[-1] + row_size != rows:
        incomplete_rows.append(row_idxs[-1])
    if col_idxs[-1]+col_size != cols:
        incomplete_cols.append(row_idxs[-1])

    #Looping over all rows and cols
    for r, row in enumerate(row_idxs):
        for c, col in enumerate(col_idxs):

            #First extract patch as an input for the model
            #If row and col is not incomplete entire patch size is extracted
            if row not in incomplete_rows and col not in incomplete_cols:
                patch_array = image_array[row:(row + row_size), col:(col + col_size)]

            # If row or col is incomplete, the missing rows or cols have to be zero padded to match the model input shape
            # Later the zero padded rows and/or cols are removed again
            elif row in incomplete_rows and col not in incomplete_cols:
                temp_array = image_array[row:row+(rows - row), col:(col + col_size)]
                patch_array = np.zeros(shape=(row_size, col_size))
                patch_array[:(rows - row), :] = temp_array

            elif row not in incomplete_rows and col in incomplete_cols:
                temp_array = image_array[row:(row + row_size), col:col+(cols - col)]
                patch_array = np.zeros(shape=(row_size, col_size))
                patch_array[:, :(cols - col)] = temp_array

            elif row in incomplete_rows and col in incomplete_cols:
                temp_array = image_array[row:(row + row_size), col:col+(cols - col)]
                patch_array = np.zeros(shape=(row_size, col_size))
                patch_array[:(rows - row), :(cols - col)] = temp_array

            patch_array = patch_array.flatten()
            patch_array = patch_array.reshape(-1, 1)

            # Normalize the patch array
            patch_array = scaler.transform(patch_array)

            # Reshape the patch array
            patch_array = np.reshape(patch_array, (row_size, col_size))
            patch_array = np.expand_dims(patch_array, axis=2)
            patch_array = np.expand_dims(patch_array, axis=0)

            # Predict patch
            y_pred = model.predict(patch_array)
            y_pred = y_pred[0, :, :, 0]
            y_rescaled = scaler.inverse_transform(y_pred)

            y_rescaled = y_rescaled.astype(np.int32)

            # Reshape the predicted patches. Depending on their position, the zero padded rows/cols need to be removed
            # Depending if they are a first row or col, there's no overlap to consider
            # If row and col is not incomplete no rows/cols need to be removed
            if row not in incomplete_rows and col not in incomplete_cols:

                #Slice out array. Depending on the row/col number is a or no overlap to consider
                if row is not 0 and col is not 0:
                    y_rescaled = y_rescaled[overlay:-overlay, overlay:-overlay]
                if row is 0 and col is not 0:
                    y_rescaled = y_rescaled[:-overlay, overlay:-overlay]
                if row is not 0 and col is 0:
                    y_rescaled = y_rescaled[overlay:-overlay, :-overlay]
                if row is 0 and col is 0:
                    y_rescaled = y_rescaled[:-overlay, :-overlay]

            # If row is incomplete, the zero padded rows need to be removed
            elif row in incomplete_rows and col not in incomplete_cols:

                # Slice out array. Depending on the row/col number is a or no overlap to consider
                if row is not 0 and col is not 0:
                    y_rescaled = y_rescaled[overlay-overlay:, overlay:-overlay]
                    y_rescaled = y_rescaled[:(rows - row - overlay), :]
                if row is not 0 and col is 0:
                    y_rescaled = y_rescaled[overlay:-overlay, :-overlay]
                    y_rescaled = y_rescaled[:(rows - row-overlay), :cols - col]

            # If col is incomplete, the zero padded cols need to be removed
            elif row not in incomplete_rows and col in incomplete_cols:

                # Slice out array. Depending on the row/col number is a or no overlap to consider
                if row is not 0 and col is not 0:
                    y_rescaled = y_rescaled[overlay:-overlay, overlay:-overlay]
                    y_rescaled = y_rescaled[:, :(cols - col - overlay)]
                if row is 0 and col is not 0:
                    y_rescaled = y_rescaled[:-overlay, overlay:-overlay]
                    y_rescaled = y_rescaled[:(rows - row), :cols - col - overlay]

            # If row and col are incomplete, the zero padded rows and cols eed to be removed
            elif row in incomplete_rows and col in incomplete_cols:
                # extract only remaing parts and zero pad the array. Later remove the zero padded rows
                y_rescaled = y_rescaled[:(rows - row - overlay), :(cols - col - overlay)]

            print('Writing row {}/{} and col {}/{}'.format(r+1, len(row_idxs), c+1, len(col_idxs)))

            # Insert predicted patch in array
            # If it is the 1st row and col, no overlap is considered on the left and top
            if row is 0 and col is 0:
                output_array[row:(row + row_size-overlay), col:(col + col_size-overlay)] = y_rescaled

            # If it is the 1st row no overlap is considered on the left
            if row is 0 and col is not 0:
                output_array[row:(row + row_size-overlay), col_size-overlay+(c-1)*(col_size-2*overlay):col + col_size-overlay] = y_rescaled

            # If it is the 1st col no considered is added on the top
            if row is not 0 and col is 0:
                output_array[row_size-overlay+(r-1)*(row_size-2*overlay):row + row_size-overlay, col:(col + col_size-overlay)] = y_rescaled

            # If it is not a first row or column no overlap is considered
            if row is not 0 and col is not 0:
                output_array[row_size-overlay+(r-1)*(row_size-2*overlay):row + row_size-overlay, col_size-overlay+(c-1)*(col_size-2*overlay):col + col_size-overlay] = y_rescaled

    # Create filename
    fields = {'var_name': 'SIG0-SPECKLE-TEST3Kernel-Log-',
              'datetime_1': sgrt_name.obj.dtime_1+'T'+sgrt_name.obj.dtime_2,
              'datetime_2': sgrt_name.obj.dtime_1+'T'+sgrt_name.obj.dtime_2,
              'band': 'VV',
              'tile_name': tile_name,
              'grid_name': subgrid,
              'data_version': 'V1M0R01',
              'sensor_field': sgrt_name.obj.mission_id+sgrt_name.obj.spacecraft_id+sgrt_name.obj.mode_id+sgrt_name.obj.product_type+sgrt_name.obj.res_class}
    outname = str(YeodaFilename(fields))

    #Mask NoData
    output_array[output_array <= -3000] == np.nan

    #Write array to file
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    out_raster = GeoTiffFile(filepath=os.path.join(output_path, outname), mode='w', n_bands=1, sref=sref,
                             geotrans=gt)
    out_raster.write(output_array, band=1, nodataval=[-9999])
    out_raster.close()


def lee_filter_sar_image(input_path, output_path, filter_size):
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
    # Create SGRT Naming object from filename to extract meta data
    sgrt_name = SgrtFilename.from_filename(os.path.basename(input_path))

    # Define parameters for the geo reference
    grid = Equi7Grid(10)
    subgrid = sgrt_name.obj.grid_name
    tile_name = sgrt_name.obj.tile_name


    # Define parameters for the georeference
    equi7tile = grid.create_tile(subgrid+'_'+tile_name)
    sref = equi7tile.core.projection.wkt
    gt = equi7tile.geotransform()

    # Read the input image as array
    image = GeoTiffFile(filepath=input_path, mode='r', n_bands=1, sref=sref,
                        geotrans=gt)
    image_array = image.read(band=1)

    lee_filter_array = lee_filter(image_array, filter_size)

    # Create filename
    fields = {'var_name': 'SIG0-SPECKLE-LEE',
              'datetime_1': '20190101T000000',
              'datetime_2': '20201231T235959',
              'band': 'VV',
              'tile_name': tile_name,
              'grid_name': subgrid,
              'data_version': 'V1M0R01',
              'sensor_field': 'S1IWGRDH'}
    outname = str(YeodaFilename(fields))

    lee_filter_array[lee_filter_array <= -3000] == np.nan

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    out_raster = GeoTiffFile(filepath=os.path.join(output_path, outname), mode='w', n_bands=1, sref=sref,
                             geotrans=gt)
    out_raster.write(lee_filter_array, band=1, nodataval=[-9999])
    out_raster.close()


def diff_image(image1_path, image2_path, diffimage_outpath):
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
    # Read the input image
    image1 = Image.open(image1_path)
    image1_array = np.array(image1)
    image2 = Image.open(image2_path)
    image2_array = np.array(image2)

    diffimage_array = image1_array - image2_array
    diffimage_sum = sum(diffimage_array)

    if diffimage_outpath is not None:
        tile = Image.fromarray(diffimage_array)
        tile.save(diffimage_outpath)

    return diffimage_sum