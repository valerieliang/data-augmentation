import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageAugmentation:
    """
    A class containing various methods for image data augmentation including
    brightness, contrast, saturation adjustments, and other transformations.
    """
    
    @staticmethod
    def display_comparison(original, modified, title="Modified Image", factor=None):
        """
        Helper method to display original and modified images side by side.
        
        Args:
            original: Original image (numpy array)
            modified: Modified image (numpy array)
            title: Title for the modified image
            factor: Optional factor value to display in the title
        """
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('on')
        
        # Modified image
        plt.subplot(1, 2, 2)
        plt.imshow(modified)
        if factor is not None:
            plt.title(f'{title} (factor: {factor})')
        else:
            plt.title(title)
        plt.axis('on')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def change_brightness(image, factor, display=True):
        """
        Change the brightness of an image by multiplying pixel values by a factor.
        
        Args:
            image: Input image (numpy array)
            factor: Brightness factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)
            display: Whether to display the result (default: True)
            
        Returns:
            Brightened image (numpy array)
        """
        # Increase brightness by multiplying pixel values
        # Clip to ensure values stay within valid range (0-255)
        brightened = np.clip(image.astype(float) * factor, 0, 255).astype(np.uint8)
        
        if display:
            ImageAugmentation.display_comparison(image, brightened, "Brightness", factor)
        
        return brightened
    
    @staticmethod
    def change_contrast(image, factor, display=True):
        """
        Change the contrast of an image by adjusting pixel values around the mean.
        
        Args:
            image: Input image (numpy array)
            factor: Contrast factor (1.0 = no change, <1.0 = lower contrast, >1.0 = higher contrast)
            display: Whether to display the result (default: True)
            
        Returns:
            Contrast adjusted image (numpy array)
        """
        # Calculate the mean pixel value as the reference point
        mean = np.mean(image, axis=(0, 1))
        
        # Adjust the contrast by moving pixel values away from or toward the mean
        # Formula: new_pixel = mean + factor * (old_pixel - mean)
        adjusted = mean + factor * (image.astype(float) - mean)
        
        # Clip to ensure values stay within valid range (0-255)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        if display:
            ImageAugmentation.display_comparison(image, adjusted, "Contrast", factor)
        
        return adjusted
    
    @staticmethod
    def change_saturation(image, factor, display=True):
        """
        Change the saturation of an image by converting to HSV, adjusting saturation, and converting back to RGB.
        
        Args:
            image: Input image (numpy array in RGB format)
            factor: Saturation factor (1.0 = no change, <1.0 = less saturated, >1.0 = more saturated)
            display: Whether to display the result (default: True)
            
        Returns:
            Saturation adjusted image (numpy array)
        """
        # Convert the image from RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adjust the saturation channel
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        if display:
            title = "Black & White" if factor == 0.0 else "Saturation"
            ImageAugmentation.display_comparison(image, adjusted, title, factor)
        
        return adjusted
    