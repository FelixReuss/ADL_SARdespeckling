import os
import numpy as np
from veranda.io.geotiff import GeoTiffFile
from equi7grid.equi7grid import Equi7Grid
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.ndimage import gaussian_filter
from PIL import Image

def lee_filter(input_array, size):
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
    # Calculate array mean, square mean and variance
    img_mean = uniform_filter(input_array, (size, size))
    img_sqr_mean = uniform_filter(input_array**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    # Calculate the overall variance to determine the weights for the smoothing
    overall_variance = variance(input_array)

    # remove speckle by smoothing the input. The higher the deviation from the mean, the higher the smoothing weight
    img_weights = img_variance / (img_variance + overall_variance)
    out_array = img_mean + img_weights * (input_array - img_mean)
    return out_array

def add_speckle_noise(image_path, stdv_path, out_path):
    """
     Generate one batch of data

     Parameters
     __________
     image_path: str
         path to the image image to add the noise to
     stdv_path: str
         path to the corresponding standard deviation image (required to calculate the noise level parameter)
     out_path: str
         path for the output image
     """
    print(f"Processing image {os.path.basename(image_path)}")
    # Define parameters for the georeference
    grid = Equi7Grid(10)
    equi7tile = grid.create_tile(os.path.basename(image_path)[-21:-4])
    sref = equi7tile.core.projection.wkt
    gt = equi7tile.geotransform()

    # Read the input image as array
    image = GeoTiffFile(filepath=image_path, mode='r', n_bands=1, sref=sref,
                                   geotrans=gt)
    image_array = image.read(band=1)
    row,col = image_array.shape

    # Create some random array for the gaussian noise
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)

    # Read the standard deviation array to determine the noise level parameter
    stdv = GeoTiffFile(filepath=stdv_path, mode='r', n_bands=1, sref=sref,
                                   geotrans=gt)
    stdv_array = stdv.read(band=1)

    # Create the gaussian noise array
    noisy_array = image_array + (stdv_array/3200.)*(image_array * gauss)
    noisy_array = noisy_array.astype(np.int32)

    # Apply some smoothing to remove extreme outliers
    noisy_array = gaussian_filter(noisy_array, sigma=0.3)

    # Save the noise array to file
    outname = os.path.basename(image_path).replace('TMENSIG38', 'NOISY')
    tile_outdir = out_path+'/'+os.path.basename(image_path)[-14:-4]
    if not os.path.exists(tile_outdir):
        os.makedirs(tile_outdir)
    out_raster = GeoTiffFile(filepath=os.path.join(tile_outdir, outname), mode='w', n_bands=1, sref=sref,
                           geotrans=gt)
    out_raster.write(noisy_array, band=1, nodataval=[-9999])
    out_raster.close()

def patch_extractor(image_paths, outdir, outstring, patch_size=400):
    """
     Extracts image square shaped patches from input images and writes them to file

     Parameters
     __________
     image_paths: str or list of string
         path(s) to the input images
     outdir: str
         path to the output directory
     outstring: str
         string for the output file name
     tile_size: int
         size of the output patches
     """

    # Create output path if it not exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Counter for all files and tiles
    main_idx = 0
    # Loop over all input paths
    for f, file_path in enumerate(image_paths):
        file_name = os.path.basename(file_path)[:-4]

        # Read the input image
        image = Image.open(file_path)
        image_array = np.array(image)  # "data" is a height x width x 4 numpy array

        image_size = image_array.shape[0]

        # Calculate the number of patches per row an overall
        split_size = image_size / patch_size
        number_of_tiles = split_size ** 2

        print(f"Processing image {file_name}")

        # Counter for tiles per file
        idx = 0
        # Loop over all patches
        for tile_index_i in range(int(split_size)):
            for tile_index_j in range(int(split_size)):

                # Determine next row/col to extract
                row_from = tile_index_i * patch_size
                row_to = tile_index_i * patch_size + patch_size
                col_from = tile_index_j * patch_size
                col_to = tile_index_j * patch_size + patch_size

                # Extract patch array for corresponding rows/cols
                tile_array = image_array[row_from:row_to, col_from:col_to]

                # Write the patch to file
                tile_file_name = f"{outstring}_{main_idx}.tif"
                tile_file_path = os.path.join(outdir, tile_file_name)
                tile = Image.fromarray(tile_array)
                tile.save(tile_file_path)

                idx += 1
                main_idx += 1
