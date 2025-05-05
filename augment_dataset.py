import os
import argparse
import cv2
import numpy as np
from filters import ImageAugmentation

def create_output_dir(output_dir):
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir: Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def process_image(image_path, output_dir):
    """
    Apply various filters to an image and save results
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
    """
    # Extract filename and extension
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # Read image (in BGR format)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Dictionary of augmentation functions and their parameters
    augmentations = {
        'bright_high': lambda img: ImageAugmentation.change_brightness(img, 1.5, display=False),
        'bright_low': lambda img: ImageAugmentation.change_brightness(img, 0.5, display=False),
        'contrast_high': lambda img: ImageAugmentation.change_contrast(img, 1.5, display=False),
        'contrast_low': lambda img: ImageAugmentation.change_contrast(img, 0.5, display=False),
        'sat_high': lambda img: ImageAugmentation.change_saturation(img, 1.5, display=False),
        'sat_low': lambda img: ImageAugmentation.change_saturation(img, 0.5, display=False),
        'bw': lambda img: ImageAugmentation.change_saturation(img, 0.0, display=False)
    }
    
    # Apply each augmentation and save
    for aug_name, aug_func in augmentations.items():
        # Apply the augmentation
        augmented_img = aug_func(img_rgb)
        
        # Convert back to BGR for saving
        augmented_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
        
        # Generate output filename
        output_filename = f"{name}_{aug_name}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the augmented image
        cv2.imwrite(output_path, augmented_bgr)
        print(f"Saved {output_path}")
    
    # TODO: include bounding box for each image

def main():
    """Main function to process all images in a directory"""
    parser = argparse.ArgumentParser(description='Apply image augmentation filters to a folder of images')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', help='Directory to save augmented images (default: input_dir + "_augmented")')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.input_dir + "_augmented"
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Get list of image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(args.input_dir) 
                  if os.path.isfile(os.path.join(args.input_dir, f)) and 
                  os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        process_image(image_path, output_dir)
    
    print(f"Augmentation complete. Augmented images saved to {output_dir}")

if __name__ == "__main__":
    main()
