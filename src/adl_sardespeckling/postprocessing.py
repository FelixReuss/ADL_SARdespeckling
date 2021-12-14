from PIL import Image
import numpy as np

def reconstruct_image(patch_dir, out_dir):
    pass

def diff_image(image1_path, image2_path, diffimage_outpath):
    """
     Python Implementation of lee filter function. Removes speckle like noise from an input image using a weighted uniform filter

     Parameters
     __________
     image1_path: str
         path to the first input image
     image2_path: str
         path to the second input image
     diffimage_outpath: str
        path including file name for the output image

     Returns
     _______
     diffimage_sum: total sum of the difference between the two input images
     """
    # Read the input image
    image1 = Image.open(image1_path)
    image1_array = np.array(image1)
    image2 = Image.open(image2_path)
    image2_array = np.array(image2)

    #Calculate the diff array
    diffimage_array = image1_array - image2_array

    #Calcualte the absolute difference between the two images
    diffimage_sum = sum(diffimage_array)

    if diffimage_outpath is not None:
        tile = Image.fromarray(diffimage_array)
        tile.save(diffimage_outpath)

    return diffimage_sum

pass
