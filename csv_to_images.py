import numpy as np
import cv2
import os

os.mkdir("output")

with open("train.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size

for i in range(1,num_of_instances):
    try:
        emotion, img = lines[i].split(",")
        img = img.replace('"', '')
        img = img.replace('\n', '')
        pixels = img.split(" ")

        pixels = np.array(pixels, 'float32')
        image = pixels.reshape(48, 48)

        path_file_name = f"output/{i}_{emotion}.jpg"
        cv2.imwrite(path_file_name, image)

    except Exception as ex:
        print(ex)
