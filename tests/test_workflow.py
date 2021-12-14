# Copyright (c) 2020,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
Tests for the ADL_SARdespeckling workflow
"""
import os
import shutil
import unittest
import random
import numpy as np
from PIL import Image
from keras.engine.training import Model
from adl_sardespeckling.preprocessing import add_speckle_noise, patch_extractor
from adl_sardespeckling.postprocessing import diff_image
from adl_sardespeckling.utils import DataGenerator
from adl_sardespeckling.train import get_model
from adl_sardespeckling.train import train_model
import warnings

class TestWorkflow(unittest.TestCase):

    def setUp(self):
        """ Set up input and output path """
        self.data_path = os.path.join(os.path.dirname(__file__), "test_data")
        self.output_path = os.path.join(os.path.dirname(__file__), "test_output")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


    def tearDown(self):
        """ Removes all test data """
        shutil.rmtree(self.output_path)

    def test_extract_patches(self):
        """
        Tests extracting patches from SAR scence by checking if extracted patches can be opened as image
        """
        patch_size = 400
        outstring = 'test'
        patch_extractor(os.path.join(self.data_path, 'TMENSIG38_E065N034T1.tif'), self.output_path, outstring, patch_size=patch_size)
        image_path = random.choice(os.listdir(self.output_path))
        image = Image.open(image_path)

        self.assertIsInstance(image, Image.Image)

    def test_add_speckle_noise(self):
        """
        Tests adding speckle noise to image patch by comparing original image array with noise image array
        """
        image_path = os.path.join(self.data_path, 'TMENSIG38_E065N034T1.tif')
        stdv_path = os.path.join(self.data_path, 'TSDVSIG38_E065N034T1.tif')
        out_path = os.path.join(self.output_path, 'NOISY_E065N034T1.tif')
        add_speckle_noise(image_path=image_path, stdv_path=stdv_path, out_path=self.output_path)

        noise_image = Image.open(out_path)
        noise_array = np.array(noise_image)

        org_image = Image.open(image_path)
        org_array = np.array(org_image)

        self.assertNotEqual(noise_array, org_array)


    def test_unet_model(self):
        patch_size = random.choice([100, 200, 400, 800])
        unet_model = get_model(patch_size=patch_size)

        self.assertIsInstance(unet_model, Model)

    def test_diff_image(self):
        image_path = os.path.join(self.data_path,
                              'TMENSIG38_E065N034T1.tif')
        stdv_path = os.path.join(self.data_path,
                             'TSDVSIG38_E065N034T1.tif')

        diffimage_outpath = os.path.join(self.output_path, 'diff_iamge.tif')
        diff_image(image1_path=image_path, image2_path=stdv_path, diffimage_outpath=diffimage_outpath)
        image = Image.open(diffimage_outpath)

        self.assertIsInstance(image, Image.Image)

if __name__ == '__main__':
    unittest.main()
