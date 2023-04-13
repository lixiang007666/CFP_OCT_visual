import cv2
import numpy as np

# Load original image and predicted segmentation map
original_image = cv2.imread("./oct/0091.png")
predicted_map = cv2.imread("./oct/pred_11.jpg", cv2.IMREAD_GRAYSCALE)

# Create a color map for class labels
color_map = {
    80: (0, 0, 255),   # Red
    160: (0, 255, 0),   # Green
    0: (255, 0, 0)   # Blue
}

# Replace class labels in the predicted array with colors
predicted_image = np.zeros_like(original_image)
for label, color in color_map.items():
    mask = predicted_map == label
    predicted_image[mask] = color

# Blend original image and predicted image
alpha = 0.5
blended_image = cv2.addWeighted(original_image, 1-alpha, predicted_image, alpha, 0)

# Save the blended image
cv2.imwrite("visualized_image.png", blended_image)