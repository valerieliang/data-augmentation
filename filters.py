import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

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
    def change_brightness(image, factor):
        """
        Change the brightness of an image by multiplying pixel values by a factor.
        
        Args:
            image: Input image (numpy array)
            factor: Brightness factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)
            
        Returns:
            Brightened image (numpy array)
        """
        # Increase brightness by multiplying pixel values
        # Clip to ensure values stay within valid range (0-255)
        brightened = np.clip(image.astype(float) * factor, 0, 255).astype(np.uint8)
        
        return brightened
    
    @staticmethod
    def change_contrast(image, factor):
        """
        Change the contrast of an image by adjusting pixel values around the mean.
        
        Args:
            image: Input image (numpy array)
            factor: Contrast factor (1.0 = no change, <1.0 = lower contrast, >1.0 = higher contrast)
            
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
        
        return adjusted
    
    @staticmethod
    def change_saturation(image, factor):
        """
        Change the saturation of an image by converting to HSV, adjusting saturation, and converting back to RGB.
        
        Args:
            image: Input image (numpy array in RGB format)
            factor: Saturation factor (1.0 = no change, <1.0 = less saturated, >1.0 = more saturated)
            
        Returns:
            Saturation adjusted image (numpy array)
        """
        # Convert the image from RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adjust the saturation channel
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return adjusted
    
    @staticmethod
    def salt_and_pepper_noise(image, density=0.02):
        """
        Add salt and pepper noise to an image.
        
        Args:
            image: Input image (numpy array)
            density: Proportion of pixels to be affected by noise (0.0 to 1.0)
            
        Returns:
            Image with salt and pepper noise (numpy array)
        """
        total_pixels = image.size
        num_salt = int(density * total_pixels/2)
        num_pepper = int(density * total_pixels/2)
        img_noisy = image.copy()

        # randomly select salty and peppery pixels
        for i in range(num_salt):
            x = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            img_noisy[x, y] = 255  
        for i in range(num_pepper):
            x = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            img_noisy[x, y] = 0
        
        return img_noisy
    
    @staticmethod
    def convolution2d(image, kernel, stride=1):
        """
        Helper function to perform a 2D convolution on an image with a given kernel.
        
        Args:
            image: 2D numpy array representing the input image
            kernel: 2D numpy array representing the convolution kernel
            stride: integer, the step size for the convolution (default: 1)
            
        Returns:
            output: 2D numpy array representing the convolved image
        """
        # get dimensions
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # pads to keep output the same size as input
        pad_h = (kernel_height - 1) // 2
        pad_w = (kernel_width - 1) // 2

        # output array
        output_height = (image_height - kernel_height + 2 * pad_h) // stride + 1
        output_width = (image_width - kernel_width + 2 * pad_w) // stride + 1

        output = np.zeros((output_height, output_width))
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # convolve image and filter
        for y in range(output_height):
            for x in range(output_width):
                region = padded_image[y * stride:y * stride + kernel_height,
                                      x * stride:x * stride + kernel_width]
                output[y, x] = np.sum(region * kernel)

        return output
    
    @staticmethod
    def gaussian_blur(image, kernel_size=9, sigma=1.0):
        """
        Apply Gaussian blur to an image using a Gaussian kernel.
        
        Args:
            image: Input image (numpy array)
            kernel_size: size of the Gaussian kernel (must be odd)
            sigma: standard deviation of the Gaussian distribution
            
        Returns:
            Blurred image (numpy array)
        """
        # set up the kernel
        gaussian_blur_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                gaussian_blur_kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * \
                    np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        gaussian_blur_kernel /= np.sum(gaussian_blur_kernel)

        # handle grayscale and RGB images
        if image.ndim == 2:
            blurred = ImageAugmentation.convolution2d(image, gaussian_blur_kernel)
        elif image.ndim == 3:
            blurred_channels = [
                ImageAugmentation.convolution2d(image[:, :, c], gaussian_blur_kernel)
                for c in range(image.shape[2])
            ]
            blurred = np.stack(blurred_channels, axis=-1)
        
        # Convert to proper type for display
        blurred = blurred.astype(np.uint8)
        
        return blurred
    
    @staticmethod
    def sharpen_image(image):
        """
        Sharpen the image using a Laplacian kernel.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Sharpened image (numpy array)
        """
        # define kernel
        laplacian_kernel = np.array([[0, 1, 0],
                                                                [1, -4, 1],
                                                                [0, 1, 0]], dtype=np.float32)

        if image.ndim == 3:
            sharpened = np.stack([
                image[:, :, c] - ImageAugmentation.convolution2d(image[:, :, c], laplacian_kernel)
                for c in range(image.shape[2])
            ], axis=-1)
        else:
            sharpened = image - ImageAugmentation.convolution2d(image, laplacian_kernel)

        # Clip to ensure valid range and convert to uint8
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    @staticmethod
    def rotate_image(image, angle):
        """
        Rotate the image by a given angle.
        
        Args:
            image: Input image (numpy array)
            angle: angle in degrees to rotate the image
            
        Returns:
            Rotated image (numpy array)
        """
        if angle == 0:
            return image

        # Get image dimensions
        height, width = image.shape[:2]
        rotated_image = np.zeros_like(image)

        # Convert angle to radians
        theta = math.radians(angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Rotation center
        cx, cy = width // 2, height // 2

        for y in range(height):
            for x in range(width):
                # shift to origin
                xt = x - cx
                yt = y - cy

                # apply rotation
                xr = cos_theta * xt - sin_theta * yt
                yr = sin_theta * xt + cos_theta * yt

                # shift back
                src_x = int(round(xr + cx))
                src_y = int(round(yr + cy))

                # if within bounds, assign pixel
                if 0 <= src_x < width and 0 <= src_y < height:
                    rotated_image[y, x] = image[src_y, src_x]
            
        return rotated_image
    
    @staticmethod
    def reflect_image(image, axis):
        """
        Reflect the image across a given axis.
        
        Args:
            image: Input image (numpy array)
            axis: 'x' or 'y' to specify the reflection axis
                 'x' for vertical flip (across x-axis)
                 'y' for horizontal flip (across y-axis)
            
        Returns:
            Reflected image (numpy array)
        """
        if axis == 'x':  # vertical flip
            flipped_image = image[::-1, ...]  # reverse rows
        elif axis == 'y':  # horizontal flip
            flipped_image = image[:, ::-1, ...]  # reverse columns
        
        return flipped_image
    
    @staticmethod
    def shear_image(image, shear_factor_x=0, shear_factor_y=0):
        """
        Shear an image in the X or Y direction using a manual approach.
        
        Args:
            image: Input image (numpy array)
            shear_factor_x: shear factor for horizontal slant
            shear_factor_y: shear factor for vertical slant
            
        Returns:
            Sheared image (numpy array)
        """
        height, width = image.shape[:2]
        
        # create larger canvas to accommodate shearing
        enlarged_height = height + abs(int(shear_factor_y * width))
        enlarged_width = width + abs(int(shear_factor_x * height))
        enlarged_image = np.zeros((enlarged_height, enlarged_width, 3), dtype=image.dtype)
        
        # apply shearing transformations (map to larger canvas)
        for y in range(height):
            for x in range(width):
                new_x = int(x + shear_factor_x * y)
                new_y = int(y + shear_factor_y * x)
                if 0 <= new_x < enlarged_width and 0 <= new_y < enlarged_height:
                    enlarged_image[new_y, new_x] = image[y, x]
        
        # scale transformation back to original size
        sheared_image = cv2.resize(enlarged_image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return sheared_image
