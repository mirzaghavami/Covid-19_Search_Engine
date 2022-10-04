from fileinput import filename
from matplotlib import image
from pymongo import MongoClient
from  pathlib import Path
from PIL import Image
import gridfs
import io

def mongo_conn():
    try:
        client = MongoClient("mongodb+srv://CBIR-Covid-19:0mEDz3n0BpO3XxMA@cluster0.4kaupcc.mongodb.net/?retryWrites=true&w=majority")
        print("MongoDB connected ",client)
        return client.features_vgg16_grid_file
    except Exception as ex:
        print("Error in mongoDB connection", ex)

db = mongo_conn()

for feature_path in sorted(Path("./static/feature/features_vgg16").glob("*.npy")):
    name = feature_path.stem + feature_path.suffix
    file_location = feature_path
    file_data = open(file_location,"rb")
    data = file_data.read()
    fs = gridfs.GridFS(db)
    fs.put(data,filename = name)
    print("Upload Completed",name)
    

