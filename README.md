# Image Augmentation Tool
This README provides an overview of the augment_dataset.py script, which is designed to generate augmented images and corresponding YOLO-format label files for object detection datasets.

## Overview
The augment_dataset.py script applies various image augmentation techniques to a dataset of images and their corresponding YOLO-format label files. The script handles both pixel-level transformations (like brightness and contrast changes) and geometric transformations (like rotations and shears), adjusting the bounding box coordinates accordingly.

## Features
The script applies the following augmentations to each image:

Pixel-Level Augmentations (Preserves Bounding Boxes)
Brightness: Increased and decreased brightness
Contrast: Increased and decreased contrast
Saturation: Increased and decreased saturation
Black & White: Complete desaturation
Salt & Pepper Noise: Random noise addition
Gaussian Blur: Smoothing filter
Sharpening: Edge enhancement
Geometric Augmentations (Updates Bounding Boxes)
Rotation: Random rotation with angle between 0-360 degrees
Flipping: Random horizontal or vertical flipping
Shearing: Random horizontal and vertical shearing

## Usage
```
python augment_dataset.py input_dir [--output_dir OUTPUT_DIR] [--labels_dir LABELS_DIR] [--output_labels_dir OUTPUT_LABELS_DIR]
```


### Arguments

- **input_dir**: Directory containing input images
- **--output_dir**: Directory to save augmented images (default: input_dir + "_augmented")
- **--labels_dir**: Directory containing label files (default: same as input_dir)
- **--output_labels_dir**: Directory to save augmented labels (default: output_dir + "_labels")

## Label File Format
The script works with YOLO-format label files, where each line represents an object:

``` <class_id> <x_center> <y_center> <width> <height> ```

All values are normalized by image dimensions (values between 0 and 1).

## Examples
Basic usage:

This will:

- Read images from car_sample

- Read labels from the same directory

- Save augmented images to data/car_sample_augmented

- Save augmented labels to data/car_sample_augmented_labels

Specifying custom directories:
```
python augment_dataset.py data/car_sample --output_dir data/car_augmented --labels_dir data/car_labels --output_labels_dir data/car_augmented_labels
```

## Dependencies
- OpenCV (cv2)
- NumPy
- The custom ImageAugmentation class from the filters.py module

## Bounding Box Handling
For geometric transformations (rotate, flip, shear), the script:

Reads the original bounding boxes
Transforms the coordinates according to the applied transformation
Writes new label files with updated coordinates
For other transformations, the original label files are copied without modification.