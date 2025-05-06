import os
import argparse
import cv2
import numpy as np
import random
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

def parse_label_string(label_string):
    """
    Parse label file string into class id and bounding box coordinates
    Args:
        label_string: String from label file in YOLO format
    """
    parts = label_string.strip().split()
    class_id = int(parts[0])
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return class_id, xc, yc, w, h

def shear_bounding_boxes(bboxes, shear_factor, img_width, img_height):
    """
    Apply shear transformation to bounding boxes.
    
    Args:
        bboxes: List of bounding boxes as [xmin, ymin, xmax, ymax].
        shear_factor: Shear factor for the transformation.
        img_width: Width of the image.
        img_height: Height of the image.
    
    Returns:
        Updated bounding boxes after shear transformation.
    """
    # Transformation matrix for shear
    M_inv = np.float32([[1, 0, 0], [-shear_factor, 1, 0]])  # Inverse shear matrix

    updated_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        # Transform the four corners of the bounding box
        corners = np.array([[xmin, ymin, 1],
                            [xmax, ymin, 1],
                            [xmin, ymax, 1],
                            [xmax, ymax, 1]], dtype=np.float32)
        
        # Apply shear transformation to corners
        transformed_corners = np.dot(corners, M_inv.T)
        
        # Get new bounding box coordinates
        new_xmin = max(0, min(transformed_corners[:, 0]))
        new_ymin = max(0, min(transformed_corners[:, 1]))
        new_xmax = min(img_width, max(transformed_corners[:, 0]))
        new_ymax = min(img_height, max(transformed_corners[:, 1]))
        
        updated_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return updated_bboxes

def rotate_bounding_boxes(bboxes, angle, img_width, img_height):
    """
    Apply rotation to bounding boxes.
    
    Args:
        bboxes: List of bounding boxes as [xmin, ymin, xmax, ymax].
        angle: Angle in degrees to rotate the bounding boxes.
        img_width: Width of the image.
        img_height: Height of the image.
    
    Returns:
        Updated bounding boxes after rotation.
    """
    angle_rad = np.deg2rad(angle)
    M_inv = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), -angle, 1)  # Inverse rotation matrix

    updated_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        # Rotate the four corners of the bounding box
        corners = np.array([[xmin, ymin],
                            [xmax, ymin],
                            [xmin, ymax],
                            [xmax, ymax]], dtype=np.float32)
        
        # Apply rotation transformation to corners
        rotated_corners = cv2.transform(np.array([corners]), M_inv)[0]

        # Get new bounding box coordinates
        new_xmin = max(0, min(rotated_corners[:, 0]))
        new_ymin = max(0, min(rotated_corners[:, 1]))
        new_xmax = min(img_width, max(rotated_corners[:, 0]))
        new_ymax = min(img_height, max(rotated_corners[:, 1]))
        
        updated_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    
    return updated_bboxes

def process_image(image_path, output_dir, label_path, output_labels_dir):
    """
    Apply various filters to an image and save results
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
        label_path: Path to label file
        output_labels_dir: Directory to save augmented labels
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
        'bright_high': lambda img: ImageAugmentation.change_brightness(img, 1.5),
        'bright_low': lambda img: ImageAugmentation.change_brightness(img, 0.5),
        'contrast_high': lambda img: ImageAugmentation.change_contrast(img, 1.5),
        'contrast_low': lambda img: ImageAugmentation.change_contrast(img, 0.5),
        'sat_high': lambda img: ImageAugmentation.change_saturation(img, 1.5),
        'sat_low': lambda img: ImageAugmentation.change_saturation(img, 0.5),
        'bw': lambda img: ImageAugmentation.change_saturation(img, 0.0),
        'salt_pepper': lambda img: ImageAugmentation.salt_and_pepper_noise(img, density=0.02),
        'gaussian_blur': lambda img: ImageAugmentation.gaussian_blur(img, kernel_size=9, sigma=1.0),
        'sharpen': lambda img: ImageAugmentation.sharpen_image(img),
        # TODO: add change of basis transformations
        'rotate': lambda img: ImageAugmentation.rotate_image(img, angle=random.randint(0, 360)),
        'flip': lambda img: ImageAugmentation.flip_image(img, flip_code=random.choice(['x', 'y'])),
        'shear': lambda img: ImageAugmentation.shear_image(img, shear_factor=random.uniform(-0.5, 0.5)),
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
        
        # Save corresponding label if it exists
        if os.path.exists(label_path):
            # Create output labels directory if needed
            if not os.path.exists(output_labels_dir):
                os.makedirs(output_labels_dir)
                print(f"Created labels directory: {output_labels_dir}")
            
            # Generate augmented label filename
            label_name = os.path.basename(label_path)
            name_without_ext, label_ext = os.path.splitext(label_name)
            dst_label_filename = f"{name}_{aug_name}{label_ext}"
            dst_label_path = os.path.join(output_labels_dir, dst_label_filename)
            
            # TODO: if applying change of basis transformations, apply to bounding boxes as well
            img_height, img_width, _ = augmented_img.shape
            if aug_name == 'rotate' and bboxes:
                bboxes = rotate_bounding_boxes(bboxes, angle=random.randint(0, 360), img_width=img_width, img_height=img_height)
            elif aug_name == 'shear' and bboxes:
                bboxes = shear_bounding_boxes(bboxes, shear_factor=random.uniform(-0.5, 0.5), img_width=img_width, img_height=img_height)
            elif aug_name == 'flip' and bboxes:
                # Flip bounding boxes
                for i, bbox in enumerate(bboxes):
                    xmin, ymin, xmax, ymax = bbox
                    if random.choice([True, False]):
                        bboxes[i] = [img_width - xmax, ymin, img_width - xmin, ymax]  # Horizontal flip
                    else:
                        bboxes[i] = [xmin, img_height - ymax, xmax, img_height - ymin]
            

            # else, Copy the label content
            with open(label_path, 'r') as src_file:
                label_content = src_file.read()
                
            with open(dst_label_path, 'w') as dst_file:
                dst_file.write(label_content)
                
            print(f"Saved label: {dst_label_path}")
    

def main():
    """Main function to process all images in a directory"""
    parser = argparse.ArgumentParser(description='Apply image augmentation filters to a folder of images')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', help='Directory to save augmented images (default: input_dir + "_augmented")')
    parser.add_argument('--labels_dir', help='Directory containing label files (default: same as input_dir)')
    parser.add_argument('--output_labels_dir', help='Directory to save augmented labels (default: output_dir + "_labels")')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.input_dir + "_augmented"
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)

    # Set labels directory
    if args.labels_dir:
        labels_dir = args.labels_dir
    else:
        labels_dir = args.input_dir
    
    # Set output labels directory
    if args.output_labels_dir:
        output_labels_dir = args.output_labels_dir
    else:
        output_labels_dir = output_dir + "_labels"
    
    # Create output labels directory if it doesn't exist
    create_output_dir(output_labels_dir)
    
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
        
        # Get corresponding label file path (assuming same basename but .txt extension)
        image_basename = os.path.splitext(image_file)[0]
        label_file = f"{image_basename}.txt"
        label_path = os.path.join(labels_dir, label_file)
        
        # Process the image and its label
        process_image(image_path, output_dir, label_path, output_labels_dir)
    
    print(f"Augmentation complete. Augmented images saved to {output_dir}")
    print(f"Augmented labels saved to {output_labels_dir}")

if __name__ == "__main__":
    main()
