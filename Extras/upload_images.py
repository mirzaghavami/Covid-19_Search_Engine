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
        return client.images_grid_file
    except Exception as ex:
        print("Error in mongoDB connection", ex)

db = mongo_conn()

for image_path in sorted(Path("./static/img").glob("*.png")):
    name = image_path.stem + image_path.suffix
    file_location = image_path
    file_data = open(file_location,"rb")
    data = file_data.read()
    fs = gridfs.GridFS(db)
    fs.put(data,filename = name)
    print("Upload Completed",name)
    


# for image_path in sorted(Path("./static/img").glob("*.png")):
#     name = "Covid (1).png"
#     im = Image.open(image_path)

#     image_bytes = io.BytesIO()
#     im.save(image_bytes, format='png')

#     image = {
#         'data': image_bytes.getvalue()
#     }

#     image_id = images.insert_one(image).inserted_id




# db = mongo_conn()
# name = "Covid (1).png"
# file_location = "static/img/" + name
# file_data = open(file_location, "rb")
# data = file_data.read()
# fs = gridfs.GridFS(db)
# fs.put(data, filename = name)
# print("Upload completed")

# data = db.fs.files.find_one({'filename' : name})
# my_id = data['_id']
# outputdata = fs.get(my_id).read()
# download_location = "downloaded/" + name
# output = open(download_location,'wb')
# output.write(outputdata)
# output.close()
# print("download completed")