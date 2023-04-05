import numpy as np
import cv2
import os
from PIL import Image

def normalization(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new

def clahe_equalized(image):
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(image)
    
    return imgs_equalized

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_image = cv2.LUT(image, table)
    
    return new_image

def resize_image(input_dir, output_dir, desired_width = 100, desired_height = 100):     
    original_image = Image.open(input_dir)
    resized_image = original_image.resize((desired_width, desired_height))
    resized_image.save(output_dir)

def square_the_image(filename_input_path, filename_output_path):
    img = Image.open(filename_input_path)
    size = max(img.size)
    
    new_img = Image.new("RGBA" if img.mode == "RGBA" else "RGB", (size, size), (0, 0, 0, 0) if img.mode == "RGBA" else (0, 0, 0))
    new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    new_img.save(filename_output_path)