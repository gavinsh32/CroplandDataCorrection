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
    unique_colors = np.unique(img.reshape(-1, 3), axis=0) #finds unique elements in a 2d array. Turns 3d image into a 2d image because we do not care about location we only care about individual pixels and there color.
    print("Unique Colors identified")

    individual_color_maps = []

    for color in unique_colors:
        mask = np.all(img == color, axis=-1) #find all pixels in the image where the color matches
        
        color_image = np.zeros_like(img) #create a blacked out image
        color_image[mask] = color
        
        individual_color_maps.append(color_image)

    print("Image splitting complete")
    return individual_color_maps

# Apply a morphological opening operation on img to remove small noise splatters
def OpenMorph(img):
    pass

# Apply a morphological closing operation on img to fill in holes and recorrect
# from OpenMorph(img)
def CloseMorph(img):
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

    corrected_image = kmeans_color_correction(input, k=30) #reduce total colors

    projections = project(corrected_image)

    test = gray_scale(projections[1])

    #cv.imshow('Image', projections[1])
    cv.imshow('Image', test)
    cv.waitKey(0)

    # Ideal main logic:
    # projections = project(input)
    # for img in projections:
    #     img = filter(img)
    #     img = fill(img)
    # output = squash(projections)

# Reduce ambiguity
    
def kmeans_color_correction(img, k):
    img_reshaped = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(img_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    
    corrected_img = centers[labels.flatten()]
    corrected_img = corrected_img.reshape(img.shape)
    
    return corrected_img

def gray_scale(img):
    mask = np.any(img != [0,0,0], axis=-1)
        
    color_image = np.zeros_like(img) #create a blacked out image
    color_image[mask] = [255,255,255]

    return color_image


if __name__ == '__main__':
    main()