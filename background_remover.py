# Image Background Remover Script for 'Ta Dirjanu' website product images

# Takes as input the .jpg images stored in the 'images' directory

# Import required packages
import os
import cv2
import numpy as np

# Defining function to convert image to graysacle
def grayscale(image):
    new_image = []
    for row in image:
        new_row = []
        for pixel in row: new_row.append(int((int(pixel[0])+int(pixel[1])+int(pixel[2]))/3))
        new_image.append(new_row)
    return np.array(new_image).astype(np.uint8)

# Hard-coded parameters
# These can be adjusted for different results

blur = 3
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 11
erode_iter = 11

# blur: affects the “smoothness” of the dividing line between the background and foreground
# canny_low: the minimum intensity value along which edges will be drawn
# canny_high: the maximum intensity value along which edges will be drawn
# min_area: the minimum area a contour in the foreground may occupy. Taken as a value between 0 and 1.
# max_area: the maximum area a contour in the foreground may occupy. Taken as a value between 0 and 1.
# dilate_iter: the number of iterations of dilation will take place on the mask.
# erode_iter: the number of iterations of erosion will take place on the mask.
# mask_color: the color of the background once it is removed.

# Defining function which returns the object from the image
def get_object(image):
    new_image = image
    # Convert image to grayscale
    gs = grayscale(image)

    # Edge detection
    edges = cv2.Canny(gs, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    # Get the area of the image as a comparison
    image_area = image.shape[0] * image.shape[1]

    # calculate max and min areas in terms of pixels
    max_area_ = max_area * image_area
    min_area_ = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype = np.uint8)

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area_ and contour[1] < max_area_:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))

    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    # For every pixel in original image
    for i in range(len(new_image)):
         for j in range(len(new_image[i])):
            # Compare pixel value with mask
            if mask[i][j] == 0:
                # Set pixel value to white background
                new_image[i][j] = np.array([255, 255, 255])

    return new_image


# Get images from 'images' folder
images = {}

for x in os.listdir("images"):
    filename = os.path.splitext(x)[0]
    images[filename] = cv2.imread("images/" + x)

# For every image
for key in images:
    new_image = get_object(images[key])
    new_image_name = "result_" + key + ".jpg"

    if cv2.imwrite("result_images/"+new_image_name, new_image) == True:
        print("'"+key+".jpg' converted successfully.")

print("Background removal complete.")
