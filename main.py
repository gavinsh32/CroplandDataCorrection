# main.py
# Gavin Haynes, Ibrahim Monsour
# CS360 Database Systems
# Fall '24
# An algorithm to clean noise from the CroplandCROS dataset

import os
import sys
import cv2 as cv
import numpy as np

ROWS = 0
COLS = 0

def project(img) -> list:
    if img is None:
        print("Image could not be loaded. Check the file path.")
    unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    print("Unique Colors identified")

    individual_color_maps = []

    for color in unique_colors:
        mask = np.all(img == color, axis=-1)
        
        color_image = np.zeros_like(img)
        color_image[mask] = color
        
        individual_color_maps.append(color_image)

    print("Image splitting complete")
    return individual_color_maps

# Apply the first noise filter, removing specks from each image
def filter(img):
    pass

# Find enclosed regions and fill them in
def fill(img):
    pass

# Take a list of images and squash in to one image
def squash(images):
    pass

# Main engine
def main():
    # Load image and check args
    assert len(sys.argv) > 1, "Correct usage: python main.py path-to-input.jpg"
    input = cv.imread(sys.argv[1])
    assert input is not None, "Image " + sys.argv[1] + " failed to load."
    print("Image " + sys.argv[1] + " loaded successfully.")

    # example
    ROWS = input.shape[0]
    COLS = input.shape[1]

    print(repr(ROWS) + " rows and " + repr(COLS) + " columns.")

    quantized_image = kmeans_color_quantization(input, k=20) #reduce total colors

    projections = project(quantized_image)
    cv.imshow('Image', projections[1])
    cv.waitKey(0)

    #print(projections)
    # Ideal main logic:
    # projections = project(input)
    # for img in projections:
    #     img = filter(img)
    #     img = fill(img)
    # output = squash(projections)

# Reduce ambiguity
    
def kmeans_color_quantization(img, k):
    img_reshaped = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(img_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    
    quantized_img = centers[labels.flatten()]
    quantized_img = quantized_img.reshape(img.shape)
    
    return quantized_img


if __name__ == '__main__':
    main()