import numpy as np
import cv2
from PIL import Image
from helper import *
import imageio
from IPython.display import display


def normalization(image, max, min):
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new

def clahe_equalized(image):
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(grayimg)
    
    return imgs_equalized

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_image = cv2.LUT(image, table)
    
    return new_image

def resize_image_to_file(input_dir, output_dir, desired_width = 100, desired_height = 100):     
    original_image = Image.open(input_dir)
    resized_image = original_image.resize((desired_width, desired_height))
    resized_image.save(output_dir)

def resize_image(input_np_array, desired_width=1024, desired_height=1024):
    input_image = Image.fromarray(input_np_array)
    resized_image = input_image.resize((desired_width, desired_height))
    return np.array(resized_image)

def make_square(filename_input_path, filename_output_path, min_size=256):
    im = Image.open(filename_input_path)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.convert('RGB')
    new_im.save(filename_output_path)

def monochroming_image_to_file(img_path, target_path):
    image_file = Image.open(img_path)
    image = image_file.convert('L')
    image = np.array(image, dtype=np.uint8)
    mask = image < 128
    image[mask] = 0
    image[~mask] = 255
    new_im = Image.fromarray(image)
    new_im.save(target_path)

def monochroming_image(input_np_array):
    image = Image.fromarray(input_np_array)
    image = image.convert('L')
    image = np.array(image, dtype=np.uint8)
    mask = input_np_array < 128
    input_np_array[mask] = 0
    input_np_array[~mask] = 255

    return input_np_array

def changeMaskColor(input_np_array):
    colored_mask = np.zeros((input_np_array.shape[0], input_np_array.shape[1], 3), dtype=np.uint8)
    colored_mask[:, :, 0] = 0  # Komponen merah diatur ke 0
    colored_mask[:, :, 1] = 0  # Komponen hijau diatur sesuai dengan masker hitam putih
    colored_mask[:, :, 2] = input_np_array  # Komponen biru diatur ke 255

    return colored_mask