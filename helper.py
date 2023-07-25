import numpy as np
import os
import shutil
from PIL import Image
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from functools import reduce


def split_and_remove():
    directory = "temp-augmented-dataset/images/output"
    output_dir = "augmented-dataset"

    if not os.path.exists(os.path.join(output_dir, 'images/')):
        os.makedirs(os.path.join(output_dir, 'images/'))
        os.makedirs(os.path.join(output_dir, 'masking/'))

    total_iterations = len(os.listdir(directory))
    pbar = tqdm(total=total_iterations, desc="split augmentation result", unit="iter")

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            if "_groundtruth_" in filename:
                shutil.copy(os.path.join(directory, filename), os.path.join(output_dir, "masking/" + filename))
            elif "images_original_" in filename:
                shutil.copy(os.path.join(directory, filename), os.path.join(output_dir, "images/" + filename))
        
        pbar.update(1)

    pbar.close()

    shutil.rmtree("temp-augmented-dataset")

def rename_augmented_file(directory):
    total_iterations = len(os.listdir(directory))
    pbar = tqdm(total=total_iterations, desc="renaming augmented data", unit="iter")

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            new_filename = filename.replace("images_original_", "")
            new_filename = new_filename.replace("_groundtruth_(1)_images_", "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        pbar.update(1)

    pbar.close()

def print_image_numpy_array_not_zero(image):
    image_file = Image.open(image)
    image_array = np.array(image_file, dtype=np.uint8)

    np.set_printoptions(threshold=np.inf)
    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            if (image_array[row][col] != 0).any():
                print(image_array[row])
                break

def split_to_80_and_20(input_dir, type):
    output_dir = "final-dataset"

    if not os.path.exists(os.path.join(output_dir, 'train/')):
        os.makedirs(os.path.join(output_dir, 'train/images'))
        os.makedirs(os.path.join(output_dir, 'train/masking'))
    if not os.path.exists(os.path.join(output_dir, 'val/')):
        os.makedirs(os.path.join(output_dir, 'val/images'))
        os.makedirs(os.path.join(output_dir, 'val/masking'))

    total_iterations = len(input_dir)
    pbar = tqdm(total=total_iterations, desc=type, unit="iter")

    for index, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith(".jpg"):
            data_length = len(os.listdir(input_dir))
            n_80_percent = int(0.8 * data_length)
            
            # second dataset split to get training and validation dataset
            if index < n_80_percent:
                filename_output_path = os.path.join(output_dir, 'train/')
            else:
                filename_output_path = os.path.join(output_dir, 'val/')

            shutil.copy(os.path.join(input_dir, filename), os.path.join(filename_output_path, type + "/" + filename))

        pbar.update(1)

    pbar.close()