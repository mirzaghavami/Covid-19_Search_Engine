from tensorflow.keras.preprocessing import image 

from tensorflow.keras.applications.xception import Xception,preprocess_input # Xception
from tensorflow.keras.models import Model
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class FeatureExtractorXception :
    def __init__(self):
        base_model = Xception(weights = "imagenet")
        self.model = Model(inputs= base_model.input, outputs = base_model.get_layer("conv2d_2").output)




    def extract(self,img):

        img = img.resize((299,299)).convert("RGB")
        x= image.img_to_array(img) # To np.array
        x = np.expand_dims(x,axis = 0) # (H, W, C) -> (1, H, W, C)
        x = preprocess_input(x) # Subtract avg pixel value
        feature = self.model.predict(x)[0][2][1] #(1,4096 ) -> (4096, )
        return feature / np.linalg.norm(feature) # Normalize