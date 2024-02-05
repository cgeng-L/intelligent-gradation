import os
import numpy as np
import cv2
from tqdm import tqdm
# from pycococreatortools import pycococreatortools
import json
from tools import *

def get_approx(img, contour, length_p=0.005):
    """Get the approximate polygon
    :param img: Image to process
    :param contour: Contour of the connected component
    :param length_p: Approximation length percentage
    """
    img_adp = img.copy()
    # Approximation length calculation
    epsilon = length_p * cv2.arcLength(contour, True)
    # Get the approximate polygon
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

# train
imgfolder = r"/datasets/640-10" 
outputfolder = r"/samseg/segment-anything-main/train_outputs_01" 
savefolder = r"/datasets/640-10" 
# savepath = os.path.join(savefolder, "draws")
datafolder_images = os.path.join(savefolder, "images")
datafolder_jsons = os.path.join(savefolder, "jsons")


if not os.path.exists(datafolder_images):
    os.mkdir(datafolder_images)
if not os.path.exists(datafolder_jsons):
    os.mkdir(datafolder_jsons)
# if not os.path.exists(savepath):
#     os.mkdir(savepath)

alpha = 0.5 
beta = 1
for ii , basename_ in enumerate(tqdm(os.listdir(outputfolder))):
    imgname = "{}.JPG".format(basename_)
    imgpath = os.path.join(imgfolder, imgname)
    outputpath = os.path.join(outputfolder, basename_)
    im = cv2.imread(imgpath)
    # print("--------------------------------")
    # print(imgpath)
    imageHeight, imageWidth = im.shape[:2]
    im_copy = im.copy()

    cons = []
    annos = {
        "flags":{},
        "version": "4.5.10",
        "imagePath":imgname,
        "shapes":[],
        "imageHeight":imageHeight,
        "imageWidth":imageWidth,
        "imageData": None
    }

    classes = []
    segments = []
    for indx, pathi in enumerate(os.listdir(outputpath)):
        if pathi == "metadata.csv":
            continue
        
        tmp_im_org = cv2.imread(os.path.join(outputpath, pathi))
        src_h, src_w = tmp_im_org.shape[:2]
        # Erosion
        gray = cv2.cvtColor(tmp_im_org,cv2.COLOR_BGR2GRAY)
        ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dst = cv2.erode(binary,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
        dst = cv2.dilate(dst,kernel)
        dst = dst[...,np.newaxis]
        tmp_im_org = np.repeat(dst,3,axis=2)
        binary_mask = tmp_im_org.copy() 

        # Choose color for mask
        tmp_im = tmp_im_org.copy()
        color = np.random.randint(0,255,size=(3))
        tmp_im[:,:,0][tmp_im[:,:,0]==255] = color[0]
        tmp_im[:,:,1][tmp_im[:,:,1]==255] = color[1]
        tmp_im[:,:,2][tmp_im[:,:,2]==255] = color[2]
        draw_im = cv2.addWeighted(tmp_im, alpha, im, beta,0)
        
        # Get contour points for the mask
        thresh = cv2.Canny(tmp_im_org, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        x, y, w, h = cv2.boundingRect(contours[max_id])
       
        polygons = process(binary_mask)
        
        if len(polygons)>=2:
            print("len(polygons):{}".format(len(polygons)))
            n = [len(polygoni) for polygoni in polygons]
            polygoni = np.array(polygons[np.argmax(n)]).reshape(-1,2)
            shape_ = {
            "label":"stone_{}".format(indx),
            "points": polygoni.tolist(),
            "group_id":None,
            "shape_type":"polygon",
            "flags":{},
            }
            
            classes.append(0)
            annos["shapes"].append(shape_)
            
        else:
            polygons = np.array(polygons).reshape(-1,2)
            if len(polygons.tolist())==0:
                    continue
            shape_ = {
                "label":"stone_{}".format(indx),
                "points": polygons.tolist(),
                "group_id":None,
                "shape_type":"polygon",
                "flags":{},
            }
            classes.append(0)
            annos["shapes"].append(shape_)
    
    with open(os.path.join(datafolder_jsons, imgname.replace("JPG","json")), "w") as ff:
        json.dump(annos, ff, indent=2)
    
    # cv2.imwrite(os.path.join(savepath, imgname), draw_im)
    cv2.imwrite(os.path.join(datafolder_images, imgname), im_copy)


