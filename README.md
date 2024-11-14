# An Enhanced Deep-Learning-Based Approach for Automatic Gradation Analysis of Rockfill Materials

## Research Paper

(Link to be provided later)

## Overview

In the construction of earth-rockfill dams, achieving precise control over rockfill gradation is imperative due to its significant impact on dam settlement and stability. A recent trend in this domain involves employing image segmentation methods for gradation detection of rockfill materials.

## Steps for Rockfill Material Gradation Detection

The process of rockfill material gradation detection based on image segmentation involves two main steps:

1. **Particle Segmentation:**
   - This step focuses on segmenting particles in images.

2. **Establishment of Mapping:**
   - Establishing a mapping between surficial 2D particle size distribution and overall gradation.

## Approaches & Tools
We utilized the following approaches and tools:

- **Particle Annotation with SAM:**
  - SAM (Segment Anything Model) was employed for initial particle annotation. Subsequent adjustments by annotation experts were deemed necessary.

- **Particle Segmentation with DCS-YOLOv8s:**
  - DCS-YOLOv8s, an extension of the YOLOv8s baseline, was used for particle segmentation. Model comparisons were conducted with other models such as Mask R-CNN, SOLOv2, and YOLOv5.

- **Regression Analysis with CatboostRegressor:**
  - CatboostRegressor, a simple and effective regression analysis model, was employed to establish a mapping between surface gradation and overall gradation.

## Citation

If you find our research paper relevant and wish to cite it, please use the following format:

(To be provided later)

## References

The following open-source libraries were used in this research:

1. **Segment Anything Model (SAM)** - Facebook Research. Available at: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
2. **PaddleDetection** - PaddlePaddle. Available at: [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
3. **YOLOv5** - Ultralytics. Available at: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
4. **Ultralytics YOLOv8** - Ultralytics. Available at: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
5. **Catboost** - Catboost. Available at: [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
