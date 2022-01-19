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
from adl_sardespeckling.train import get_model
from adl_sardespeckling.postprocessing import despeckle_sar_image

class TestWorkflow(unittest.TestCase):

    def setUp(self):
        """ Set up input and output path """
        self.data_path = os.path.join(os.path.dirname(__file__), 'test_data')
        self.output_path = os.path.join(os.path.dirname(__file__), 'test_output')
        self.model_path = os.path.abspath(os.path.dirname(os.path.dirname( __file__ )), 'src', 'adl_sardespeckling', 'momdels')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


    def tearDown(self):
        """ Removes all test data """
        shutil.rmtree(self.output_path)

    def test_extract_patches(self):
        """
        Tests extracting patches from SAR scence by checking if extracted patches can be opened as image
        """
        patch_size = random.choice([10, 20, 40, 80])
        outstring = 'test'
        patch_extractor([os.path.join(self.data_path, 'TMENSIG38_E065N034T1.tif')], self.output_path, outstring, patch_size=patch_size)
        random_image = random.choice(os.listdir(self.output_path))
        image_path = os.path.join(self.output_path, random_image)
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

    def test_diff_image(self):
        """
        Tests diff image calculation by comparing two image arrays
        """
        image_path = os.path.join(self.data_path,
                              'TMENSIG38_E065N034T1.tif')
        stdv_path = os.path.join(self.data_path,
                             'TSDVSIG38_E065N034T1.tif')

        diffimage_outpath = os.path.join(self.output_path, 'diff_iamge.tif')
        diff_image(image1_path=image_path, image2_path=stdv_path, diffimage_outpath=diffimage_outpath)
        image = Image.open(diffimage_outpath)

        self.assertIsInstance(image, Image.Image)

    def test_despeckle_Sar_image(self):
        """
        Tests despeckling sar image by comparing original image array with despeckled image array
        """
        image_path = os.path.join(self.data_path,
                              'M20180731_055009--_SIG0-----_S1AIWGRDH1VVD_037_A0104_EU010M_E045N021T1.tif')
        model_path = os.path.join(self.model_path, 'model_5kernel_mse.h5')
        noise_image = Image.open(image_path)
        despeckle_sar_image(image1_path=image_path, output_path=self.output_path, path2model=model_path, overlay=40)
        noise_image_path = os.path.join(self.output_path,
                                        r'SIG0-SPECKLE-TEST3Kernel-Log-_20180731T055009--_20180731T055009--_VV__E045N021T1_EU010M_V1M0R01_S1AIWGRDH.tif')
        noise_free_image = Image.open(noise_image_path)

        self.assertNotEqual(noise_image, noise_free_image)


if __name__ == '__main__':
    unittest.main()
