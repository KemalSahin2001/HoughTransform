import numpy as np
import cv2
import os
import glob
import re
from sklearn import svm
from Part1 import resize_image, hough_circle_transform, non_maximum_suppression

def calculate_padding(resized_image, target_size=1000):
    height, width = resized_image.shape[:2] # Get the image dimensions
    delta_w = target_size - width # Calculate the width difference
    delta_h = target_size - height # Calculate the height difference
    top = delta_h // 2 # Calculate the top padding
    bottom = delta_h - top # Calculate the bottom padding
    left = delta_w // 2 # Calculate the left padding
    right = delta_w - left # Calculate the right padding
    return top, bottom, left, right # Return the padding values

def resize_image(image, max_size=1000):
    height, width = image.shape[:2] # Get the image dimensions
    if height > max_size or width > max_size: # Check if resizing is necessary
        if height > width: # Check if the height is greater than the width
            new_height = max_size # Set the new height
            new_width = int(width * (max_size / height)) # Calculate the new width
        else: # Otherwise, the width is greater than the height
            new_width = max_size # Set the new width
            new_height = int(height * (max_size / width)) # Calculate the new height
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA) # Resize the image
    else: # Otherwise, the image is already small enough
        resized_image = image # Return the original image

    return resized_image

def preprocess_images(image_path):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized_image = resize_image(gray_image, max_size=200)

    # Calculate padding
    top, bottom, left, right = calculate_padding(resized_image, target_size=200)

    # Get the corner pixel value
    corner_pixel = original_image[0, 0]

    # Convert numpy array to list
    corner_pixel = corner_pixel.tolist()

    # Pad the image with the corner pixel
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=corner_pixel)
    
    return padded_image

def compute_gradients(image):
    # Compute gradients in the x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction (in degrees)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    return magnitude, direction

def create_histograms(magnitude, direction, cell_size=8, bin_size=20):
    # Initialize histogram array
    cell_rows = int(magnitude.shape[0] / cell_size)
    cell_cols = int(magnitude.shape[1] / cell_size)
    histograms = np.zeros((cell_rows, cell_cols, int(180 / bin_size)))

    # Create a histogram for each cell
    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_magnitude = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] # Get the magnitudes of the pixels in this cell
            cell_direction = direction[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] # Get the directions of the pixels in this cell
            hist, _ = np.histogram(cell_direction, bins=np.arange(0, 181, bin_size), weights=cell_magnitude)
            histograms[i, j, :] = hist

    return histograms

def normalize_histograms(histograms, block_size=2):
    # Normalize histograms over blocks
    e = 1e-5  # Small value to avoid division by zero
    normalized_histograms = []
    for i in range(histograms.shape[0] - block_size + 1):
        for j in range(histograms.shape[1] - block_size + 1):
            block = histograms[i:i+block_size, j:j+block_size, :]
            normalized = block.flatten() / np.sqrt(np.sum(block**2) + e)
            normalized_histograms.append(normalized)

    return np.concatenate(normalized_histograms)

def extract_custom_hog(image):
    # Compute gradients
    magnitude, direction = compute_gradients(image)

    # Create histograms of gradient directions for each cell
    histograms = create_histograms(magnitude, direction)

    # Normalize histograms over blocks and flatten into a feature vector
    hog_features = normalize_histograms(histograms)

    return hog_features

def extract_label_from_filename(filename):
    # Regular expression to extract the complete label from the filename
    match = re.match(r"([a-zA-Z0-9]+_[a-zA-Z]+)_.*", filename)
    if match:
        return match.group(1)
    else:
        return None
    
def detect_circles(image_path, threshold):
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(gray_image)

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
    nms_circles = non_maximum_suppression(detected_circles, 0.5)

    return nms_circles

def pad_and_resize(roi, size=200):
    # Calculate padding to make the image square
    height, width = roi.shape[:2]
    delta = abs(height - width)
    top = bottom = left = right = delta // 2

    if height > width:
        left += delta % 2
    else:
        top += delta % 2

    # Get the corner pixel value for padding
    corner_pixel = roi[0, 0].tolist()

    # Pad the image to make it square
    squared_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=corner_pixel)

    # Resize the squared ROI to the target size
    resized_roi = cv2.resize(squared_roi, (size, size), interpolation=cv2.INTER_AREA)

    return resized_roi

def get_text_placement(text, center_coordinates, font, scale, thickness):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0] # Get text size
    text_x = center_coordinates[0] - text_size[0] // 2 # Calculate text placement
    text_y = center_coordinates[1] + text_size[1] // 2 # Calculate text placement
    return (text_x, text_y)

def draw_fancy_label(img, text, position, font_face, font_scale, text_color, border_color, thickness=1, border_thickness=4):
    x, y = position
    # Draw text border
    cv2.putText(img, text, (x, y), font_face, font_scale, border_color, thickness=border_thickness, lineType=cv2.LINE_AA)
    # Draw the actual text
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness=thickness, lineType=cv2.LINE_AA)

hog_features = []
labels = []

train_dir = 'Train'
image_paths = glob.glob(os.path.join(train_dir, '*.jpg'))
for image_path in image_paths:
    image = preprocess_images(image_path)
    features = extract_custom_hog(image)

    # Extract label from filename
    label = extract_label_from_filename(os.path.basename(image_path))

    if label:
        hog_features.append(features)
        labels.append(label)

# Convert the list of HoG features to a numpy array for scikit-learn
X = np.array(hog_features)
y = np.array(labels)

# Create an SVM classifier
clf = svm.SVC()

# Train the classifier
clf.fit(X, y)

test_dir = ['TestV', 'TestR']
output_dir = ['TestV_HoG', 'TestR_HoG']

for input_dir,output_dir in zip(test_dir,output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clear the directory if it already exists
        for file in glob.glob(os.path.join(output_dir, '*.jpg')):
            os.remove(file)


    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    for image_path in image_paths:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_image = resize_image(original_image, max_size=1000)
        circles = detect_circles(image_path, threshold=0.5)

        # Draw the circles
        for (a, b, radius, _) in circles:
            cv2.circle(original_image, (a, b), radius, (0, 250, 0), 4)

        
        for (x, y, r, _) in circles:
            # Extract the ROI using the bounding box, handling boundary conditions
            x1, y1 = max(x - r, 0), max(y - r, 0)
            x2, y2 = min(x + r, original_image.shape[1]), min(y + r, original_image.shape[0])
            roi = original_image[y1:y2, x1:x2]

            # Pad and resize the extracted ROI
            processed_roi = pad_and_resize(roi)

            # Extract HoG features from the processed ROI
            hog_features = extract_custom_hog(processed_roi)

            # Reshape the features to match the input shape expected by the SVM classifier
            hog_features = hog_features.reshape(1, -1)

            # Predict the class of the ROI using the trained SVM classifier
            class_name = clf.predict(hog_features)[0]

            # Get proper placement for the text
            text_position = get_text_placement(class_name, (x, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Annotate the image with the class name in a fancy way
            draw_fancy_label(original_image, class_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0))

        # Save the annotated image to the output directory
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, original_image)