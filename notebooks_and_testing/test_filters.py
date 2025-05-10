import matplotlib.pyplot as plt
import numpy as np
import cv2
from filters import ImageAugmentation

# Load the original image
image_path = 'motorcycle.jpg'  
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Apply all filters
filters = {
    "Original": original_image,
    "High Exposure\n(1.8x)": ImageAugmentation.change_brightness(original_image, 1.8),
    "Low Exposure\n(0.4x)": ImageAugmentation.change_brightness(original_image, 0.4),
    "High Contrast\n(2.0x)": ImageAugmentation.change_contrast(original_image, 2.0),
    "Low Contrast\n(0.5x)": ImageAugmentation.change_contrast(original_image, 0.5),
    "High Saturation\n(2.0x)": ImageAugmentation.change_saturation(original_image, 2.0),
    "Grayscale": cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY),
    "Salt & Pepper\n(5%)": ImageAugmentation.salt_and_pepper_noise(original_image, 0.05),
    "Gaussian Blur\n(σ=2)": ImageAugmentation.gaussian_blur(original_image, kernel_size=15, sigma=2),
    "Sharpened": ImageAugmentation.sharpen_image(original_image),
    "Rotated\n(30°)": ImageAugmentation.rotate_image(original_image, 30),
    "Sheared\n(0.2x)": ImageAugmentation.shear_image(original_image, shear_factor_x=0.2),
    "Reflected\n(Y-axis)": ImageAugmentation.reflect_image(original_image, 'y')
}

# Display the images
plt.figure(figsize=(20, 14)) 
plt.suptitle("Image Filter Comparisons", fontsize=20, y=0.95) 
rows, cols = 3, 5
for i, (title, filtered_image) in enumerate(filters.items(), 1):
    ax = plt.subplot(rows, cols, i)
    
    if len(filtered_image.shape) == 2:
        ax.imshow(filtered_image, cmap='gray')
    else:
        ax.imshow(filtered_image)
    
    ax.set_title(title, fontsize=9, pad=6)
    ax.axis('off')

# Adjust layout with more precise spacing
plt.tight_layout(pad=3.0, h_pad=8.0, w_pad=6.0)
plt.subplots_adjust(top=0.85)
plt.show()