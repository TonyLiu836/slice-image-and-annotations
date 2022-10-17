import os
import cv2
import json
import numpy as np
import json 
import math
import base64

img_path = "IMG_20221012_145230.jpg"
json_path = "IMG_20221012_145230.json"

#fill in with desired save locations for sliced images and adjusted jsons
save_slice_img_path = ""    
save_slice_json_path = ""

stride_H, stride_W = 960, 1280    # change stride to desired sliced image size
gen_new_JSON = True               # change to false if only sliced images are needed

json_file = open(json_path)
data = json.load(json_file)
img = cv2.imread(img_path)
img_name = os.path.basename(img_path)[:-4]

img_H, img_W, img_C = np.shape(img)

def get_slice_boundary():

    x_coor = []
    y_coor = []

    for i in range(math.ceil(img_H/stride_H)):
        if stride_H * i + stride_H >= img_H:
            y_coor.append(img_H - stride_H)
        else:
            y_coor.append(stride_H * i)

    for j in range(math.ceil(img_W/stride_W)):
        if stride_W * j + stride_W >= img_W:
            x_coor.append(img_W - stride_W)
        else:
            x_coor.append(stride_W * j)
    
    return y_coor, x_coor


def slice_img(img, y_coor, x_coor, gen_new_json):
    counter = 0
    for i in range(len(y_coor)):
        for j in range(len(x_coor)):
            y_top, y_bot = y_coor[i], y_coor[i] + stride_H
            x_left, x_right = x_coor[j], x_coor[j] + stride_W

            img_slice = img[y_top:y_bot, x_left:x_right]

            if gen_new_json:
                generate_new_json(img_slice, counter, x_left, y_top)
            
            cv2.imwrite(os.path.join(save_slice_img_path, f"{img_name}_{counter}.jpg"), img_slice)
            counter += 1

def generate_new_json(img_slice, counter, x_left, y_top):
    initial_text = {"version": "5.0.1",
              "flags": {},
              "shapes": [],
              "imagePath": None,
              #"imageData":None,                          #uncomment if need to generate imageData for LabelMe
              "imageHeight": stride_H,
              "imageWidth": stride_W
            }
    
    shapes = []
    for polygon in data["shapes"]:
        poly_length = len(polygon["points"])
        temp = 0
        for x,y in polygon["points"]: 
            if x < x_left or x > x_left + stride_W or y < y_top or y > y_top+stride_H:    #check if annotation point extends beyond boundary of sliced img
                continue                                                                  #don't include the point in counter if outside boundary of sliced img
            else:
                temp += 1


        # use this portion if you want to include polygons that might be partially cutoff
        if temp >= 0.65 * poly_length:
            polygon_copy = polygon.copy()                       #make a copy of polygon to avoid modifying the original
            points_np = np.asarray(polygon_copy["points"])
            points_np = np.subtract(points_np, [x_left, y_top])           #adjust the annotation points to fit inside sliced image
            polygon_copy["points"] = points_np.tolist()
            initial_text["shapes"].append(polygon_copy)
        

        # uncomment this portion if you want to only use complete objects (no masks are cutoff)
        """
        if temp == poly_length:
            
            polygon_copy = polygon.copy()                       #make a copy of polygon to avoid modifying the original
            #print(polygon_copy)
            points_np = np.asarray(polygon_copy["points"])
            #points_np %= [stride_W,stride_H]                   
            polygon_copy["points"] = points_np.tolist()
            initialText["shapes"].append(polygon_copy)
        """

    initial_text["imagePath"] = f'{img_name}_{counter}.jpg'
    #initialText["imageData"] = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()     #uncomment if need to generate imageData for LabelMe
    json_name = f"{img_name}_{counter}"
    json_path_w_ext = os.path.join(save_slice_json_path, json_name + ".json")
    
    with open(json_path_w_ext, 'w') as fp:
        json_string = json.dumps(initial_text)
        fp.write(json_string)
        fp.close()


y_coor,x_coor = get_slice_boundary()
slice_img(img, y_coor, x_coor, gen_new_JSON)
