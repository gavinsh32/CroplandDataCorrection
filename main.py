# main.py
# Gavin Haynes, Ibrahim Monsour
# CS360 Database Systems
# Fall '24
# An algorithm to clean noise from the CroplandCROS dataset

# To do:
# squash() doesn't account for reshaping done by canny()
# add run folders to store each session
# add folders like 'filter1' 'filter2' to save copies of each step

import os
import sys
import cv2 as cv
import numpy as np
import tkinter as tk
import shutil
from tkinter import ttk
from tkinter import filedialog as fd
from enum import Enum

inputPath = ""
filters = ['Squash and Exit', 'View All', 'Morphological Closing', 'Morphological Opening', 
           'Canny']

class State(Enum):
    FIRST = 0,
    SECOND = 1,
    INPUT = 2,
    LOAD = 3,
    EXIT = 4

# Main engine
def main():
    welcome()
    inputImg = None         # input image
    results = []            # result of previous filter operation
    output = None           # final squashed output
    
    # Selection Engine
    state = State.LOAD  # intial state: loading
    while True:
        match state:
            case State.LOAD:
                inputImg = open()  # prompt user for input image
                stdin = input("Would you like to use our predefined structure to correct the image?\nEnter an option [yes/no]: ")
                if stdin == 'yes':  # use preset pipeline
                    output = defaultmodel(inputImg)
                    viewCompare(inputImg, output)
                    state = State.EXIT
                    break
                else:
                    state = State.FIRST
            case State.FIRST:   # Applying clustering with custom values
                clustered = pickClusterFunction(inputImg)
                results = project(clustered)
                state = State.SECOND
            case State.SECOND:
                option = promptFilters()
                if option == 0:
                    state = State.EXIT
                elif option == 1:
                    viewGrid(results)
                else:
                    oldresults = results
                    results = []
                    for result in oldresults:
                        result = filter(option, result)
                        results.append(result)
            case State.EXIT:  # squash, save, and exit
                print('\nSquashing and saving...')
                output = squash(results)
                cv.imwrite('output.jpg', output)
                viewCompare(inputImg, output)
                break
            case _:
                print("\nERROR: unexpected state " + repr(state))
                break
    print('Exit successful.')

# Open a file and get it's path
def open():
    global inputPath

    inputPath = fd.askopenfilename(
                    title='Select Input File', 
                    initialdir='./input', 
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
    return input

def welcome() -> None:
    print('\nWelcome to CroplandCROS Data Corrector and Visualizer.\
          \nGavin Haynes & Ibrahim Monsour\nCS360 Database Systems, Fall \'24\
          \nThis tool consists of two stages: clustering and filtering. Dominant colors are\
          identified in an input sample and is split up accordingly. Next, you will be\
          \nprompted with several options to filter the image afterwards.')

def defaultmodel(input):
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

    viewGrid(morphs)

    # Save results
    result = squash(morphs)
    cv.imwrite(f'output.jpg', result)

    return result

# resize img to desired size (dx, dy) using Nearest Neighbor interpolation
def resize(img, dx, dy):
    return cv.resize(img, (dx, dy), cv.INTER_NEAREST)

# Apply Canny edge detector to img with thresholds t1 and t2
def canny(img, t1, t2):
    return cv.Canny(img, t1, t2)

# prompt the user for clustering options and apply to img
def pickClusterFunction(img):
    print("\nNow that you have loaded your image it is time to select your intial cluster function.")
    print("A cluster function is needed because the image initially has significant noise which makes our algorithms view slight rgb differences as being different colors.")
    print("This is important because an image can only have 10 colors but with noise our algorithms will find possibly hundred or thousands of colors.")

    option = int(input("\nOptions:\n 1. kMeans\n*Other options not yet added\
                       \n\nEnter an option [1]: "))

    match option:
        case 1:
            num = int(input("\nEnter the number of dominant colors you want identified: "))
            return kMeans(img, num)
        case _:
            return img

# Display options for filtering, which are defined
# globally. Prompt the user for an option and check input.
def promptFilters() -> int:
    i = 0
    print('\nNow, the input has been split up in to many images with one color each.\
          \nSelect a filter to modify each split image:')
    for filter in filters:
        print(repr(i) + '. ' + filter)
        i += 1
    option = int(input(f'\nEnter an option [0-{i-1}]: '))

    # return option if it's valid or 0 otherwise
    return option if checkInput(option, i) else 0

# pick a filter using option and apply it to a list of images
def filter(option: int, img) -> list:
    match option:
        case 2: # Morph Open
            return morph(img, 0)
        case 3: # Morph Close
            return morph(img, 1)
        case 4: # Canny
            return canny(img, 50, 150)
        case _:
            print('Error: failed to apply filter option ' + repr(option))
            return img

# Check that input is a number and in range
# return num if valid else 0
def checkInput(num: int, max: int) -> bool:
    return True if num >= 0 and num <= max else False

# view a list of images as a grid
def viewGrid(resultList):
    length  = len(resultList)
    rows = int(np.ceil(np.sqrt(length)))
    cols = int(np.ceil(length / rows))

    resultList_with_borders = []
    for img in resultList:
        img = resize(img, 250, 250)
        bordered_img = cv.copyMakeBorder(img, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=[255, 255, 255])
        resultList_with_borders.append(bordered_img)

    rows_images = []
    for i in range(rows):
        row_images = resultList_with_borders[i * cols:(i + 1) * cols]
        row = np.hstack(row_images)
        rows_images.append(row)
    
    grid = np.vstack(rows_images)

    cv.imshow("Image Grid", grid)
    cv.waitKey(0)
    cv.destroyAllWindows()

# View input_img and output_img side-by-side
def viewCompare(input_img, output_img):
    input_img = resize(input_img, 500, 500)
    output_img = resize(output_img, 500, 500)
    output_img = cv.copyMakeBorder(output_img, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=[255, 255, 255])
    input_img = cv.copyMakeBorder(input_img, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=[255, 255, 255])
    side_by_side = np.hstack((input_img, output_img))
    
    cv.imshow("Input vs Output", side_by_side)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Create a new folder for operating with folders for each intermediate file
# such as morphs, projections, etc.
def setup(path) -> None:
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
    return input

def project(input) -> list:
    img = input
    if img is None:
        print("Image could not be loaded. Check the file path.")
    unique_colors = np.unique(img.reshape(-1, 3), axis=0) #finds unique elements in a 2d array. Turns 3d image into a 2d image because we do not care about location we only care about individual pixels and there color.

    individual_color_maps = []

    for color in unique_colors:
        mask = np.all(img == color, axis=-1) #find all pixels in the image where the color matches
        
        color_image = np.zeros_like(img) #create a blacked out image
        color_image[mask] = color #put all colors of this unique type in the new blacked out image
        
        individual_color_maps.append(color_image)

    print("\nImage projection complete.")
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
            print('Error: morph() recieved invalid option ' + repr(option))
            return projection 
    print('Morphological operation complete')
    return output 

# Reduce ambiguity
def kMeans(img, k):
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
    print('\nImage squashing complete.\n')
    return combined_img

if __name__ == '__main__':
    main()