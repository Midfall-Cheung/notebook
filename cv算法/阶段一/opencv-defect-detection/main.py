import cv2
import numpy as np
from pathlib import Path

input_dir = Path('cv算法/阶段一/opencv-defect-detection/data')

output_dir = Path('cv算法/阶段一/opencv-defect-detection/output')

image_paths = sorted(input_dir.glob('*.jpg'))
# Get the first image in the input directory  

for path in image_paths:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not read the image.")
        exit()
    else:
        print("Image read successfully.")
        print(f"Image path: {path}")
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image min value: {np.min(img)}")
        print(f"Image max value: {np.max(img)}") 
        print(f"Image channels: {img.ndim}") 

    img_blur_mean = cv2.blur(img, (5, 5))
    cv2.imwrite(str(output_dir / f"mean_blur_{path.name}"), img_blur_mean)
    img_blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(str(output_dir / f"gaussian_blur_{path.name}"), img_blur_gaussian)
    img_blur_median = cv2.medianBlur(img, 5)
    cv2.imwrite(str(output_dir / f"median_blur_{path.name}"), img_blur_median)

    threshold_value, binary = cv2.threshold(img_blur_mean,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_dir / f"binary_{path.name}"), binary)
    print(f"Threshold value: {threshold_value}")
