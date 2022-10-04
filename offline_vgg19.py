from PIL import Image
from  pathlib import Path
import numpy as np


from feature_extractor_vgg19 import FeatureExtractorVGG19



if __name__ == "__main__":

    fe = FeatureExtractorVGG19()

    for img_path in sorted(Path("./static/img").glob("*.png")):
        print(img_path)

        # Extract a deep feature here

        feature = fe.extract(img= Image.open(img_path))
        print(type(feature),feature.shape)
        feature_path = Path('./static/feature/features_vgg19') / (img_path.stem + ".npy")
        
       

        # Save the feature 
        np.save(feature_path,feature)
