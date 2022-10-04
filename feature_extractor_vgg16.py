# Import libraries

# Tensorflow to use vgg16 pre-trained model
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Numpy to work with matrices of layers of convolutional neural networks
import numpy as np

# Use os for path and operating system related tasks
import os
# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define a class for extracting feature using VGG16

class FeatureExtractorVGG16 :

    # Define constructor
    # Using imagenet weights and use keras pre-trainded model 
    # The output is the first fully connected layer as we need for extracting features
    def __init__(self):
        base_model = VGG16(weights = "imagenet")
        self.model = Model(inputs= base_model.input, outputs = base_model.get_layer("fc1").output)

    # Define extract method with the arguments which are constrcutor and the image it self
    def extract(self,img):
        img = img.resize((224,224)).convert("RGB")
        x = image.img_to_array(img) # To np.array
        x = np.expand_dims(x,axis = 0) # (H, W, C) -> (1, H, W, C)
        x = preprocess_input(x) # Subtract avg pixel value
        feature = self.model.predict(x)[0]#(1,4096 ) -> (4096, )
        print(feature,feature.shape)


        return feature / np.linalg.norm(feature) # Normalize