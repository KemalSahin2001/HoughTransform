import numpy as np
import cv2
import math
import os
import glob

def calculate_padding(resized_image, target_size=1000):
    height, width = resized_image.shape[:2]
    delta_w = target_size - width
    delta_h = target_size - height
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    return top, bottom, left, right

def resize_image(image, max_size=1000):
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image

# Define the Hough Transform function for circles
def hough_circle_transform(edge_image, magnitude, direction, radius_range):
    # Initialize the Hough accumulator array to zeros
    # The dimensions are based on the image size and the range of radii
    hough_space = np.zeros((edge_image.shape[0], edge_image.shape[1], len(radius_range)))

    # Cast votes in the Hough space
    for x in range(edge_image.shape[1]):
        for y in range(edge_image.shape[0]):
            # Check if we have an edge point
            if edge_image[y, x] > 0:
                # Cast votes for potential circle centers
                for radius_index, radius in enumerate(radius_range):
                    # Determine the a and b offsets based on the gradient direction
                    a = int(x - radius * math.cos(direction[y, x]))
                    b = int(y - radius * math.sin(direction[y, x]))
                    # Check if the center would be within bounds
                    if a >= 0 and a < edge_image.shape[1] and b >= 0 and b < edge_image.shape[0]:
                        hough_space[b, a, radius_index] += 1

    return hough_space

def non_maximum_suppression(detected_circles, distance_threshold):
    # Sort the circles by the accumulator value in descending order (high to low)
    sorted_circles = sorted(detected_circles, key=lambda x: x[2], reverse=True)
    # Initialize a list to keep the circles that survive NMS
    nms_circles = []

    # Go through the sorted list and suppress non-maximums
    for target in sorted_circles:
        x, y, radius, _ = target
        # Check if the circle is far enough from circles that have higher votes
        is_max = True
        for candidate in nms_circles:
            candidate_x, candidate_y, candidate_radius, _ = candidate
            distance = math.sqrt((x - candidate_x) ** 2 + (y - candidate_y) ** 2)
            if distance < distance_threshold * (radius + candidate_radius) / 2:
                is_max = False
                break
        if is_max:
            nms_circles.append(target)

    return nms_circles

# Function to perform all steps: edge detection, circle detection, scaling, and drawing
def process_image(image_path, output_dir, distance_threshold):
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(gray_image)

    # Resize the image to half of its original size for faster processing
    if image_path.startswith('Train'):
        # Resize the image to half of its original size for faster processing
        original_image_resize = resize_image(original_image, max_size=200)
        resized_image = resize_image(contrast_enhanced_image, max_size=200)

        # Apply Gaussian Blur to smoothen the image
        blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

        # Apply Canny Edge Detector
        edges = cv2.Canny(blurred_image, 150, 200)

        # Apply Sobel operator to find the gradients
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Define a range of radii to search for circles
        radius_range = np.arange(60, 100)
    else:

        # Resize the image to half of its original size for faster processing
        original_image_resize = resize_image(original_image, max_size=1000)
        resized_image = resize_image(contrast_enhanced_image, max_size=1000)

        # Apply Gaussian Blur to smoothen the image
        blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

        # Apply Canny Edge Detector
        edges = cv2.Canny(blurred_image, 50, 150)

        # Apply Sobel operator to find the gradients
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Define a range of radii to search for circles
        radius_range = np.arange(5, 120)


    # Calculate the gradient magnitude and direction
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x)

    # Apply the Hough Transform for circle detection
    hough_space = hough_circle_transform(edges, magnitude, direction, radius_range)

    # Find the peaks in the Hough space to identify the circles
    threshold = 0.5 * hough_space.max()
    circles = np.where(hough_space >= threshold)

    # Extract the circle parameters and their accumulator values
    detected_circles = []
    for idx in range(len(circles[0])):
        b, a, radius_index = circles[0][idx], circles[1][idx], circles[2][idx]
        radius = radius_range[radius_index]
        vote = hough_space[b, a, radius_index]  # The vote for this circle
        detected_circles.append((a, b, radius, vote))

    # Apply Non-Maximum Suppression
    nms_circles = non_maximum_suppression(detected_circles, distance_threshold)

    # Draw the circles that survived NMS
    output_image_nms = original_image_resize.copy()
    for circle in nms_circles:
        a, b, radius, _ = circle
        cv2.circle(output_image_nms, (a, b), radius, (0, 250, 0), 4)


    size = 1000 if image_path.startswith('Test') else 200

    # Pad the image to make it square
    top, bottom, left, right = calculate_padding(resized_image, target_size=size)

    # Get the corner pixel value
    corner_pixel = original_image[0, 0]

    # Convert numpy array to list
    corner_pixel = corner_pixel.tolist()

    # Pad the image with the corner pixel
    padded_image = cv2.copyMakeBorder(output_image_nms, top, bottom, left, right, cv2.BORDER_CONSTANT, value=corner_pixel)

    # Save the image with drawn circles to the output directory
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, padded_image)


if __name__ == '__main__':

    # Directories
    input_dirs = ['Train', 'TestV', 'TestR']
    output_dirs = ['Train_Hough', 'TestV_Hough', 'TestR_Hough']

    # Create output directories if they do not exist
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Clear the directory if it already exists
            for file in glob.glob(os.path.join(output_dir, '*.jpg')):
                os.remove(file)


    # Process all images in each directory
    for input_dir, output_dir in zip(input_dirs, output_dirs):

        if(input_dir == 'Train'):
            distance_threshold = 6

        else:
            distance_threshold = 0.5


        # Get a list of all image files in the directory
        image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
        
        # Process each image
        for image_path in image_paths:
            process_image(image_path, output_dir, distance_threshold)

