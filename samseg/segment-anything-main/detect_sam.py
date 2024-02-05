import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

label_folder = r"/runs/detect/exp/labels"
image_folder = r"/project/Datas/minStones/test/images"
save_folder = r"/samseg/segment-anything-main/train_output"

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Initialize model weights
print("Loading model....")
sam_checkpoint = "/segeverything_pretrains/vit_h.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for path_i in tqdm(os.listdir(label_folder)):
    with open(os.path.join(label_folder, path_i), "r") as ff:
        datas = ff.readlines()
        
    im = cv2.imread(os.path.join(image_folder, path_i.replace("txt", "jpg")))
    folder_name = path_i.split(".")[0]
    
    image = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
    # Feed the image
    print("Loading image....")
    predictor.set_image(image)
    
    h_, w_ = im.shape[:2]
    bboxs = []
    for data_i in datas:
        data_i = np.array(data_i[:-1].split(" "), dtype=np.float64)
        class_num = data_i[0]
        cx = data_i[1] * w_
        cy = data_i[2] * h_
        dw = data_i[3] * w_
        dh = data_i[4] * h_
        
        x1, y1, x2, y2 = cx - dw/2, cy - dh/2, cx + dw/2, cy + dh/2
        bboxs.append([x1, y1, x2, y2])
  
    input_boxes = torch.tensor(bboxs, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    # Predict
    print("Loading prediction ...")
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    print(len(masks))
    for index_, mask_ in enumerate(masks):
        save_im = mask_.detach().cpu().numpy()
        h, w = save_im.shape[-2:]
        save_im = save_im.reshape(h, w, 1)
        
        if not os.path.exists(os.path.join(save_folder, folder_name)):
            os.mkdir(os.path.join(save_folder, folder_name))
            
        cv2.imwrite(os.path.join(save_folder, folder_name, "{}.png".format(index_)), save_im * 255)
