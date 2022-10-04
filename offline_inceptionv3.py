# Libraries for extracting features
# PIL for opening an image
from PIL import Image

# Pathlib to get path of the os
from  pathlib import Path
# Numpy to manipulate deep feature array 
import numpy as np

# Use FeatureExtractorVGG16 written class in other module
from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3



if __name__ == "__main__":

    # Instantiate the object from the class
    fe = FeatureExtractorInceptionV3()

    # Extracting features of each image in DB(Here db is a local storage not an actual DB)
    for img_path in sorted(Path("./static/img").glob("*.png")):
        # Just for log purpose and print path when extracting features
        print(img_path)

        # Extract a deep feature here
        # Using extract method which is for the object of the class FeatureExtractorVGG16()
        feature = fe.extract(img= Image.open(img_path))
        print(type(feature),feature.shape)
        feature_path = Path('./static/feature/features_inceptionv3') / (img_path.stem + ".npy")
        print(feature_path)
       

        # Save the feature 
        np.save(feature_path,feature)
