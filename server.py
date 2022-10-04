from tracemalloc import start
from winreg import QueryValueEx
from django.db import router
from django.shortcuts import render
import numpy as np
import pandas as pd
from IPython.display import HTML
import time
from PIL import Image
from feature_extractor_vgg16 import FeatureExtractorVGG16
from feature_extractor_vgg19 import FeatureExtractorVGG19
from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from feature_extractor_xception import FeatureExtractorXception
from datetime import datetime
from flask import Flask, request , render_template
from pathlib import Path


app = Flask(__name__)


# Read img features for VGG16
fe_vgg16 = FeatureExtractorVGG16()
features_vgg16 = []
img_path_vgg16 = []
for feature_vgg16_path in Path("./static/feature/features_vgg16").glob("*.npy"):
    features_vgg16.append(np.load(feature_vgg16_path))
    img_path_vgg16.append(Path("./static/img") / (feature_vgg16_path.stem + ".png"))
features_vgg16 = np.array(features_vgg16)

# Read img features for Xception
fe_xception = FeatureExtractorXception()
features_xception = []
img_path_xception = []
for feature_xception_path in Path("./static/feature/features_xception").glob("*.npy"):
    features_xception.append(np.load(feature_xception_path))
    img_path_xception.append(Path("./static/img") / (feature_xception_path.stem + ".png"))
features_xception = np.array(features_xception)

# Read img features for VGG19
fe_vgg19 = FeatureExtractorVGG19()
features_vgg19 = []
img_path_vgg19 = []
for feature_vgg19_path in Path("./static/feature/features_vgg19").glob("*.npy"):
    features_vgg19.append(np.load(feature_vgg19_path))
    img_path_vgg19.append(Path("./static/img") / (feature_vgg19_path.stem + ".png"))
features_vgg19 = np.array(features_vgg19)


# Read img features for InceptionV3
fe_InceptionV3 = FeatureExtractorInceptionV3()
features_InceptionV3 = []
img_path_InceptionV3 = []
for feature_InceptionV3_path in Path("./static/feature/features_inceptionv3").glob("*.npy"):
    features_InceptionV3.append(np.load(feature_InceptionV3_path))
    img_path_InceptionV3.append(Path("./static/img") / (feature_InceptionV3_path.stem + ".png"))
features_InceptionV3 = np.array(features_InceptionV3)








@app.route("/", methods=["GET", "POST"])
def index():


    if request.method == "POST":
        
        start = time.perf_counter()
        timestamp = datetime.now().isoformat().replace("T"," ")
        file = request.files["query_img"]
        # Save query img
        img = Image.open(file.stream) #PIL image
        uploaded_img_path = "static/uploaded/" +\
            datetime.now().isoformat().replace(":",".") + "_" + file.filename
        img.save(uploaded_img_path)


        if request.form['operation'] == 'VGG16FeaturExtraction':
            # Run search VGG16
            
            query_vgg16 = fe_vgg16.extract(img)
            
            print(uploaded_img_path)
            dists = np.linalg.norm(features_vgg16 - query_vgg16, axis = 1) # L2 distances to features
            ids = np.argsort(dists)[:30] # top 30 results

            scores = [(dists[id], img_path_vgg16[id]) for id in ids]
            end = time.perf_counter()
            computation_time = end - start
            return render_template("index.html", query_path = uploaded_img_path, scores= scores, ct = computation_time, ts = timestamp)


        if request.form['operation'] == 'XceptionFeaturExtraction':
            # Run search Xception
            
            
            
            
            query_xception = fe_xception.extract(img)
            
            print(uploaded_img_path)
            dists = np.linalg.norm(features_xception - query_xception, axis = 1) # L2 distances to features
            ids = np.argsort(dists)[:30] # top 30 results
            
            scores = [(dists[id], img_path_xception[id]) for id in ids]
            end = time.perf_counter()
            
            computation_time = end - start
            return render_template("index.html", query_path = uploaded_img_path, scores= scores, ct = computation_time, ts=timestamp)        
        
        if request.form['operation'] == 'VGG19FeatureExtraction':
            # Run search VGG19
            
            
            
            
            query_vgg19 = fe_vgg19.extract(img)
            
            print(uploaded_img_path)
            dists = np.linalg.norm(features_vgg19 - query_vgg19, axis = 1) # L2 distances to features
            ids = np.argsort(dists)[:30] # top 30 results
            
            scores = [(dists[id], img_path_vgg19[id]) for id in ids]
            end = time.perf_counter()
            
            computation_time = end - start
            return render_template("index.html", query_path = uploaded_img_path, scores= scores, ct = computation_time, ts=timestamp)        
        

        if request.form['operation'] == 'InceptionV3FeatureExtraction':
            # Run search InceptionV3
            
            
            
            
            query_InceptionV3 = fe_InceptionV3.extract(img)
            
            print(uploaded_img_path)
            dists = np.linalg.norm(features_InceptionV3 - query_InceptionV3, axis = 1) # L2 distances to features
            ids = np.argsort(dists)[:30] # top 30 results
            
            scores = [(dists[id], img_path_InceptionV3[id]) for id in ids]
            end = time.perf_counter()
            
            computation_time = end - start
            return render_template("index.html", query_path = uploaded_img_path, scores= scores, ct = computation_time, ts=timestamp)        
        

    else:
        return render_template("index.html")




    



if __name__ == "__main__":
    app.run(debug=True)