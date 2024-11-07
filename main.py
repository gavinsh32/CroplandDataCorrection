# main.py
# Gavin Haynes, Ibrahim Monsour
# An algorithm to clean noise from the CroplandCROS dataset

import os
import cv2 as cv

inputPath = "sample-input.jpg"
ROWS = 0
COLS = 0

# Engine
def main():
    input = cv.imread(inputPath)
    
    # Store input dimensions
    ROWS = input.shape[0]
    COLS = input.shape[1]
    print("Loaded image with " + repr(ROWS) + " rows and "
        + repr(COLS) + " columns.")

    # cv.imshow("Input", input)
    
    # cv.waitKey(0) # pause the imshow frame
    inputs = projectInput(input)


# Project the input image in to images with only one color
def projectInput(img) -> list:
    for pixel in img:
        pass

def filterNoise1(img):
    pass

if __name__ == '__main__':
    main()