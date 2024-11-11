# main.py
# Gavin Haynes, Ibrahim Monsour
# CS360 Database Systems
# Fall '24
# An algorithm to clean noise from the CroplandCROS dataset

import os
import sys
import cv2 as cv
import numpy as np

# Main engine
def main():
    # Load image and check args
    assert len(sys.argv) > 1, "Correct usage: python main.py path-to-input.jpg"
    input = cv.imread(sys.argv[1])
    assert input is not None, "Image " + sys.argv[1] + " failed to load."
    print("Image " + sys.argv[1] + " loaded successfully.")

    # find k most dominant colors in the input
    input = kmeans_color_correction(input, k=9)

    # split the input by each dominant color
    projections = project(input)    # list of images

    # save copies of all projections
    for i in range(0, len(projections)):
        cv.imwrite(f'projections/projection{i}.jpg', projections[i])

    # apply morphological transformations to further reduce noise
    morphs = [] 
    for i in range(0, len(projections)):
        morphed = morph(projections[i], 3)
        morphs.append(morphed)
    
    # save a copy of all the morphed images
    for i in range(0, len(morphs)):
        cv.imwrite(f"./morphs/morph{i}.jpg", morphs[i])

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

# Apply morphological close and open operations on a projection to both remove noise splatter
# and then fill in remaining holes.
def morph(projection, option):
    # make rectangular kernel for uniform results
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) 
    
    output = projection # make a copy of projection

    match option:   # morphing order options
        case 0: # just open
            output = morphOpen(projection, kernel)
        case 1: # just close
            output = morphClose(projection, kernel)
        case 2: # open then close
            output = morphOpen(projection, kernel)
            output = morphClose(projection, kernel)
        case 3: # close then open
            output = morphClose(projection, kernel)
            output = morphOpen(projection, kernel)
        case default:
            pass

    return output

# Apply morphological opening operation to reduce background noise splatter
def morphOpen(projection, kernel):
    return cv.morphologyEx(projection, cv.MORPH_OPEN, kernel)

# Apply morphological close operation to fill in holes 
def morphClose(projection, kernel):
    return cv.morphologyEx(projection, cv.MORPH_CLOSE, kernel)  

# Take a list of images and squash in to one image
def squash(images):
    pass

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


if __name__ == '__main__':
    main()