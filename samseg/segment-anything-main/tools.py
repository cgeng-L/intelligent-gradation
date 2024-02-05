import os
import numpy as np
import cv2
import json
import math

def cal_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def cal_ang(p1, p2, p3):
    
    eps = 1e-12
    a = math.sqrt((p2[0]-p3[0])*(p2[0]-p3[0]) + (p2[1]-p3[1])*(p2[1]-p3[1]))
    b = math.sqrt((p1[0]-p3[0])*(p1[0]-p3[0]) + (p1[1]-p3[1])*(p1[1]-p3[1]))
    c = math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))
    ang = math.degrees(math.acos((b**2-a**2-c**2)/(-2*a*c+eps)))
    return ang

def get_binary(img, minConnectedArea=1):
    rows, cols = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    _, labels, stats,  _ = cv2.connectedComponentsWithStats(img_bin, connectivity=4)
    
    # print("stats.shape[0]:{}".format(stats.shape[0]))
    for index in range(1, stats.shape[0]):
        if stats[index][4] < minConnectedArea or stats[index][4]<0.00001 * (stats[index][2] * stats[index][3]):
            labels[labels == index] == 0
    labels[labels != 0] = 1
    img_bin =  np.array(img_bin*labels).astype(np.uint8)
    return img, img_bin, rows, cols

def approx_poly_DIY(contour, min_dist = 10, ang_err = 5):
    cs = [contour[i][0] for i in range(contour.shape[0])]
    i = 0 
    while i <len(cs):
        try:
            j = (i+1) if (i!=len(cs)-1) else 0
            if cal_dist(cs[i], cs[j]) < min_dist:
                last = (i-1) if (i!=0) else (len(cs)-1)
                next = (j-1) if (j!=0) else (len(cs)-1)
                ang_i = cal_ang(cs[last], cs[i], cs[next])
                ang_j = cal_ang(cs[last], cs[j], cs[next])
                if abs(ang_i-ang_j) < ang_err:
                    dist_i = cal_dist(cs[last], cs[i]) + cal_dist(cs[i], cs[next])
                    dist_j = cal_dist(cs[last], cs[j]) + cal_dist(cs[j], cs[next])
                    
                    if dist_j < dist_i:
                        del cs[j]
                    else:
                        del cs[i]
                else:
                    i+=1
            else:
                i+=1
        except:
            i+=1
            
    # i = 0
    while i <len(cs):
        try:
            last = (i-1) if (i!=0) else (len(cs)-1)
            next = (i-1) if (i!=len(cs)-1) else 0
            ang_i = cal_ang(cs[last], cs[i], cs[next])
            if abs(ang_i) > (180-ang_err):
            # if abs(ang_i) > 75:
                del cs[j]
            else:
                i+=1
        except:
            del cs[i]
    
    res = np.array(cs).reshape([-1,1,2])
    return res
    


def get_multiregion(img, img_bin):
    contours, hierarchys = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    
    if len(contours):
        polygons = []
        relas = []
        for idx, (contour, hierarchy) in enumerate(zip(contours, hierarchys[0])):
            area_ = cv2.contourArea(contour)
            # if area_ <500:
            #     continue
            
            epsilon_ = (0.005 * cv2.arcLength(contour, True))
            if not isinstance(epsilon_, float) and not isinstance(epsilon_, int):
                epsilon_ = 0
            contour = cv2.approxPolyDP(contour, epsilon_/10, True)
            out = approx_poly_DIY(contour)
            rela_ = (idx, hierarchy[-1] if hierarchy[-1] else None)
            
            polygon_ = []
            for p in out:
                polygon_.append(p[0])    
            polygons.append(polygon_)
            relas.append(rela_)
            
        return polygons, relas
    
    else:
        return []
            
def find_min_point(i_list, o_list):
    min_dis = 1e7
    idx_i = -1
    idx_o = -1
    for i in range(len(i_list)):
        for o in range(len(o_list)):
            dis = math.sqrt((i_list[i][0]-o_list[o][0])**2 + (i_list[i][1]-o_list[o][1])**2)
            if dis <= min_dis:
                min_dis = dis
                idx_i = i
                idx_o = o
    return idx_i, idx_o


def change_list(polygons, idx):
    if idx == -1:
        return polygons
    s_p = polygons[:idx]
    polygons = polygons[idx:]
    polygons.extend(s_p)
    polygons.append(polygons[0])

    return polygons

    
def reduce_relas(polygons, relas):
    for i in range(len(relas)):
        if relas[i][1] != None:
            for j in range(len(relas)):
                if relas[j][0] == relas[i][1]:
                    if polygons[i] is not None and polygons[j] is not None:
                        min_i, min_o = find_min_point(polygons[i], polygons[j])
                        polygons[i] = change_list(polygons[i], min_i)
                        polygons[j] = change_list(polygons[j], min_o)
                        
                        if min_i != -1 and len(polygons[i])>0:
                            polygons[j].extend(polygons[i])
                        polygons[i] = None
                        
    polygons = list(filter(None, polygons))
    return polygons
      
def check_size_minmax(polygons, img_size):
    h_max, w_max = img_size
    for ps in polygons:
        for j in range(len(ps)):
            x, y = ps[j]
            if x<0:
                x=0
            elif x>w_max:
                x=w_max
            if y<0:
                y=0
            elif y>h_max:
                y = h_max
            
            ps[j] = np.array([x,y])
    return polygons                  

def process(ori_img):
    img, img_bin, rows, cols = get_binary(ori_img)
    polygons, relas = get_multiregion(img, img_bin)
    polygons = reduce_relas(polygons, relas)
    if rows is not None and cols is not None:
        polygons = check_size_minmax(polygons, (rows, cols))
    return polygons

# imgpath = r"/root/project/Modules/segment-anything-main/outputs/[1.8]11/0.png"
# img = cv2.imread(imgpath)
# polygons = process(img)
# polygons = np.array(polygons).reshape(-1,2)
# print(polygons.tolist())





