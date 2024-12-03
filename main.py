# main.py
# Gavin Haynes, Ibrahim Monsour
# CS360 Database Systems
# Fall '24
# An algorithm to clean noise from the CroplandCROS dataset

import os
import sys
import cv2 as cv
import numpy as np
import tkinter as tk
import shutil
from tkinter import ttk
from tkinter import filedialog as fd

inputPath = ""
input = None
name = ""  #make global so we can delete it at the end

# Main engine
def main():
    input = kMeans(input, k=9)      # find k most dominant colors in the input

    projections = project(input)    # split the input by each dominant color

    # save copies of all projections
    for i in range(0, len(projections)):
        cv.imwrite(f'projections/projection{i}.jpg', projections[i])

    # apply morphological transformations to further reduce noise
    morphs = [] 
    for i in range(0, len(projections)):
        morphed = morph(projections[i], 2)
        morphs.append(morphed)
    
    # save a copy of all the morphed images
    for i in range(0, len(morphs)):
        cv.imwrite(f"./morphs/morph{i}.jpg", morphs[i])

    # Save results
    cv.imwrite(f'output.jpg', squash(morphs))

# Open a file and get it's path
def open() -> bool:
    global input, inputPath

    inputPath = fd.askopenfilename(
                    title='Select Input File', 
                    initialdir='.', 
                    filetypes=(
                        ('JPG', '*.jpg'),
                        ('JPEG', '*.jpeg'),
                        ('TIF', '*.tif')            
                    )
                )
    dir = os.path.dirname(inputPath)
    # Load image and check args
    input = cv.imread(inputPath)  
    print("Image " + inputPath + " loaded successfully.")
    return False if input is None else True

# Create a new folder for operating with folders for each intermediate file
# such as morphs, projections, etc.
def setup() -> None:
    global input, inputPath, name

    i = 0
    name = 'run'
    while os.path.exists(name + str(i)):
        i += 1
    name = name + str(i)
    os.mkdir(name)
    os.mkdir(name + '/' + 'clusters')
    os.mkdir(name + '/' + 'projections')
    os.mkdir(name + '/' + 'morphs')
    dir = name + str(i)
    input = cv.imread(name)

def project() -> list:
    global input
    img = input
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

    # Apply morphological opening operation to reduce background noise splatter
    def morphOpen(projection):
        return cv.morphologyEx(projection, cv.MORPH_OPEN, kernel)

    # Apply morphological close operation to fill in holes 
    def morphClose(projection):
        return cv.morphologyEx(projection, cv.MORPH_CLOSE, kernel) 

    match option:   # morphing order options
        case 0:                                 # just open
            output = morphOpen(projection)
        case 1:                                 # just close
            output = morphClose(projection)
        case 2:                                 # open then close
            output = morphOpen(projection)
            output = morphClose(projection)
        case 3:                                 # close then open
            output = morphClose(projection)
            output = morphOpen(projection)
        case default:
            pass

    return output 

# Reduce ambiguity
def kMeans(img=inputPath, k=any):
    img_reshaped = img.reshape((-1, 3)).astype(np.float32) #create 2d array

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2) #criteria to stop running the kmeans if any criteria is met
    _, labels, centers = cv.kmeans(img_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS) #run kmeans

    centers = np.uint8(centers)
    
    corrected_img = centers[labels.flatten()] #map each pixel to the closest color center
    corrected_img = corrected_img.reshape(img.shape) #reshape the corrected image back to the same dimensions as the input image
    
    return corrected_img

def greyscale(img):
    mask = np.any(img != [0,0,0], axis=-1) #create a map of all pixels in the img that are not black and use -1 to compare all RGB values
        
    color_image = np.zeros_like(img) #create a blacked out image
    color_image[mask] = [255,255,255] #turn all non black pixels white

    return color_image

def squash(filtered_imgs):
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