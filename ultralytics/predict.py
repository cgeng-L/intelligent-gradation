from ultralytics import YOLO
from PIL import Image
import cv2

# Load pre-trained YOLOv8n model
model = YOLO('/ultralytics/runs/segment/train9_OptAug_Dual_SKAPPF/weights/best.pt')

# Define the path of the image file
source = '/datasets/test/2307/H03_30.jpg'

# Perform inference on the source
results = model(source, max_det=1000, retina_masks=True, stream=False)  # Results object list

# Display results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs

annotated_frame = results[0].plot(boxes=False)
cv2.imwrite('/ultralytics/runs/segment/H03_30_results.jpg', annotated_frame)  # Save the image
