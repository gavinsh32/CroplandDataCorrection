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
        color_image[mask] = color #put all colors of this unique type in the new blacked out image
        
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

    corrected_image = kmeans_color_correction(input, k=8) #reduce total colors

    projections = project(corrected_image)

    test = squash_img([projections[0], projections[1]])

    #cv.imshow('Image', projections[1])
    cv.imshow('Image', test)
    cv.waitKey(0)

    #print(projections)
    # Ideal main logic:
    # projections = project(input)
    # for img in projections:
    #     img = filter(img)
    #     img = fill(img)
    # output = squash(projections)

# Reduce ambiguity
    
def kmeans_color_correction(img, k):
    img_reshaped = img.reshape((-1, 3)).astype(np.float32) #create 2d array

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2) #criteria to stop running the kmeans if any criteria is met
    _, labels, centers = cv.kmeans(img_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS) #run kmeans

    centers = np.uint8(centers)
    
    corrected_img = centers[labels.flatten()] #map each pixel to the closest color center
    corrected_img = corrected_img.reshape(img.shape) #reshape the corrected image back to the same dimensions as the input image
    
    return corrected_img

def gray_scale(img):
    mask = np.any(img != [0,0,0], axis=-1) #create a map of all pixels in the img that are not black and use -1 to compare all RGB values
        
    color_image = np.zeros_like(img) #create a blacked out image
    color_image[mask] = [255,255,255] #turn all non black pixels white

    return color_image

def squash_img(filtered_imgs):
    height, width, channels = filtered_imgs[0].shape #find the shape of the image

    combined_img = np.zeros((height, width, channels), dtype=np.uint8) #create initial blacked out image

    #loop through all pixels and imgs
    for i in range(height):
        for j in range(width):
            for img in filtered_imgs:
                if np.array_equal(combined_img[i,j], [0,0,0]): #check if pixel is black which means it can be changed
                    combined_img[i,j] = img[i,j]
    return combined_img

if __name__ == '__main__':
    main()