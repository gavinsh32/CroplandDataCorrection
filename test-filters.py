# main.py
# Gavin Haynes
# CS360 Database Systems
# Fall '24
# Threshold testing for different filters

import os
import sys
import cv2 as cv

ROWS = 0
COLS = 0

# Project the input image in to images with only one color
# Input: image, Output: list of images
# Also take note of the "depth" of the image;
def project(img) -> list:
    # find the first pixel and it's color
    # strip all pixels of that color from the image
    # find a new color
    # repeat
    # return a list of images with all pixels of one color each
    pass

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
    
    # Read input image from command line
    input = cv.imread(sys.argv[1])
    
    # Check that the image was loaded and display dimensions
    assert input is not None, "Image " + sys.argv[1] + " failed to load."
    print("Image " + sys.argv[1] + " loaded with " + repr(input.shape[0])
           + " rows and " + repr(input.shape[1]) + " cols.")

    # Ideal main logic:
    # projections = project(input)
    # for img in projections:
    #     img = filter(img)
    #     img = fill(img)
    # output = squash(projections)

# Reduce ambiguity
if __name__ == '__main__':
    main()