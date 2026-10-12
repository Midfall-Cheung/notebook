import cv2
import numpy as np

image_path = 'cv算法/阶段一/opencv-defect-detection/data/patches_1.jpg'  
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read the image.")
    exit()
else:
    print("Image read successfully.")
    print(f"Image path: {image_path}")
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min value: {np.min(img)}")
    print(f"Image max value: {np.max(img)}") 
    print(f"Image channels: {img.ndim}")  # Number of channels (1 for grayscale)