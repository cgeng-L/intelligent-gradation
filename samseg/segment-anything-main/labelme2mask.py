import os
import glob
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
from labelme import utils
from PIL import Image

# Added border drawing, mimicking the demo

if __name__ == '__main__':
    json_folder = r"/datasets/640-10/jsons"  # Path to JSON files
    image_folder = r"/datasets/640-10/images"  # Path to image files
    save_folder = r"/datasets/640-10/masks"  # Path to save generated masks
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    border_color = (255, 0, 0)  # Blue border color in RGB format
    border_thickness = 2  # Border thickness

    for path_i in os.listdir(json_folder):
        img_path = os.path.join(image_folder, path_i.replace(".json", ".JPG"))
        basename_ = os.path.basename(img_path)
        label_path = os.path.join(json_folder, path_i)

        img = cv2.imread(img_path)
        data = json.load(open(label_path))  # Load JSON file

        new_shapes = []
        for shape_i in data['shapes']:
            if len(shape_i["points"]) <= 2:
                continue
            new_shapes.append(shape_i)

        lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, new_shapes)
        masks = []
        class_ids = []
        for i in range(1, len(lbl_names)):  # Skip the first class (0 is the background, skip and do not take!)
            masks.append((lbl == i).astype(np.uint8))  # Example: when the pixel value is 1, the first mask is a 0-1 composition (0 is the background, 1 is the object)
            class_ids.append(i)  # Record corresponding mask and class_id

        masks = np.asarray(masks).squeeze().astype(np.uint8)
        w, h = masks.shape[1:]
        new_mask = np.zeros((w, h, 3)).astype(np.uint8)

        for mask_ in masks:
            color = np.random.randint(0, 255, size=(3))
            new_mask[:, :, 0][mask_ == 1] = color[0]
            new_mask[:, :, 1][mask_ == 1] = color[1]
            new_mask[:, :, 2][mask_ == 1] = color[2]

            # Find the boundary of the mask
            contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the boundary
            cv2.drawContours(img, contours, -1, border_color, 6)

        alpha = 0.4
        beta = 0.8
        draw_img = cv2.addWeighted(new_mask, alpha, img, beta, 0)

        # Convert image array to PIL image
        img_pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))

        # Save image with 300 dpi
        img_pil.save(os.path.join(save_folder, basename_), dpi=(300, 300))
