import numpy as np
import cv2
import matplotlib.pyplot as plt

# Parse text file into array based on line breaks
def parse_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]

# Get label information from string
def parse_label_string(label_string):
    parts = label_string.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Label format error: {label_string}")
    class_id = int(parts[0])
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return class_id, xc, yc, w, h

# Draw bounding boxes on the image
def draw_bbox(image, label_list):
    for label in label_list:
        class_id, xc, yc, w, h = label
        x1 = int((xc - w / 2) * image.shape[1])
        y1 = int((yc - h / 2) * image.shape[0])
        x2 = int((xc + w / 2) * image.shape[1])
        y2 = int((yc + h / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Load labels and image
labels = parse_text_file('presentation_images/car_magazine_labels.txt')
image = cv2.imread('presentation_images/car_magazine.jpg')

# Check if image is loaded correctly
if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Parse labels
label_list = [parse_label_string(label) for label in labels]

# Draw bounding boxes
image_with_boxes = draw_bbox(image, label_list)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
