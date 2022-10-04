from pymongo import MongoClient
from bson.binary import Binary
from  pathlib import Path
from PIL import Image
import io
import os
import gridfs
import matplotlib.pyplot as plt


def mongo_conn():
    try:
        client = MongoClient("mongodb+srv://CBIR-Covid-19:0mEDz3n0BpO3XxMA@cluster0.4kaupcc.mongodb.net/?retryWrites=true&w=majority")
        print("MongoDB connected ",client)
        return client.images_grid_file
    except Exception as ex:
        print("Error in mongoDB connection", ex)

db = mongo_conn()

for data in db.fs.files.find({}):
    my_id = data['_id']
    fs = gridfs.GridFS(db)
    outputdata = fs.get(my_id).read()
    name = data['filename']
    download_location = "downloads/" + name
    output = open(download_location,'wb')
    output.write(outputdata)
    output.close()
    print("Download completed,",download_location)
