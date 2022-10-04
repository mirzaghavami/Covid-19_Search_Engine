from xml.etree.ElementInclude import include
import numpy as np
import os
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input # VGG19
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class FeatureExtractorVGG19 :
    def __init__(self):
        base_model = VGG19(weights = "imagenet")
        self.model = Model(inputs= base_model.input, outputs = base_model.get_layer("fc1").output)




    def extract(self,img):

        img = img.resize((224,224)).convert("RGB")
        x= image.img_to_array(img) # To np.array
        x = np.expand_dims(x,axis = 0) # (H, W, C) -> (1, H, W, C)
        x = preprocess_input(x) # Subtract avg pixel value to adequate the image with the format model requires
        feature = self.model.predict(x)[0] #(1,4096 ) -> (4096, )
        return feature / np.linalg.norm(feature) # Normalize