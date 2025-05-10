import numpy as np
import cv2
import matplotlib.pyplot as plt

def salt_and_pepper_noise(img_rgb, density=0.02):
    """
    Add salt and pepper noise to an image.
    Parameters:
    - img_rgb: Input image (numpy array)
    - density: Proportion of pixels to be affected by noise (0.0 to 1.0)
    Returns:
    - img_noisy: Image with salt and pepper noise (numpy array)
    """
    total_pixels = img_rgb.size
    num_salt = int(density * total_pixels/2)
    num_pepper = int(density * total_pixels/2)
    img_noisy = img_rgb.copy()

    # randomly select salty and peppery pixels
    for i in range(num_salt):
        x = np.random.randint(0, img_rgb.shape[0])
        y = np.random.randint(0, img_rgb.shape[1])
        img_noisy[x, y] = 255  
    for i in range(num_pepper):
        x = np.random.randint(0, img_rgb.shape[0])
        y = np.random.randint(0, img_rgb.shape[1])
        img_noisy[x, y] = 0
    
    return img_noisy

def convolution2d(image, kernel, stride=1):
    """
    Helper function to perform a 2D convolution on an image with a given kernel.
    Parameters:
    - image: 2D numpy array representing the input image
    - kernel: 2D numpy array representing the convolution kernel
    - stride: integer, the step size for the convolution (1)
    Returns:
    - output: 2D numpy array representing the convolved image
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

# Gaussian Blur (example of weighted averaging filter (I.12))
def gaussian_blur(image, kernel_size=9, sigma=1.0):
    """
    Apply Gaussian blur to an image using a Gaussian kernel (handles RGB images).
    Parameters:
    - image: 2D numpy array representing the input image
    - kernel_size: size of the Gaussian kernel (must be odd)
    - sigma: standard deviation of the Gaussian distribution
    Returns:
    - blurred_image: 2D numpy array representing the blurred image
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
        return convolution2d(image, gaussian_blur_kernel)
    elif image.ndim == 3:
        blurred_channels = [
            convolution2d(image[:, :, c], gaussian_blur_kernel)
            for c in range(image.shape[2])
        ]
        return np.stack(blurred_channels, axis=-1)

# Load the image using OpenCV (note: this loads in BGR format)
img = cv2.imread('backofcar.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array = np.array(img_rgb)

rainy_img = salt_and_pepper_noise(img_array, density=0.075)

# Display the image
plt.figure(figsize=(10, 8))
plt.imshow(rainy_img)
plt.axis('off')
plt.show()

