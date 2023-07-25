from PIL import Image
from preprocessing import *
from tqdm.notebook import tqdm
from PIL import Image
import os


input_dir = "dataset/"
output_dir = "preprocessed-dataset/"
final_dir = "final-dataset/"

if not os.path.exists(os.path.join(output_dir, 'images/')):
    os.makedirs(os.path.join(output_dir, 'images/'))
    os.makedirs(os.path.join(output_dir, 'masking/'))
if not os.path.exists(os.path.join(final_dir, 'test/')):
    os.makedirs(os.path.join(final_dir, 'test/images'))
    os.makedirs(os.path.join(final_dir, 'test/masking'))

for dir_type in os.listdir(input_dir):
    dir_type_input_path = os.path.join(input_dir, dir_type + '/')
    dir_type_output_path = os.path.join(output_dir, dir_type + '/')

    total_iterations = len(dir_type_input_path)
    pbar = tqdm(total=total_iterations, desc=dir_type, unit="iter")
    
    for index, filename in enumerate(os.listdir(dir_type_input_path)):
        data_length = len(os.listdir(dir_type_input_path))
        n_90_percent = int(0.9 * data_length)

        filename_input_path = os.path.join(dir_type_input_path, filename)
        
        # first dataset split to get test dataset
        if index < n_90_percent:
            filename_output_path = os.path.join(dir_type_output_path, filename)
        else:
            filename_output_path = os.path.join(final_dir, 'test/' + dir_type + '/' + filename)

        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):            
            square_the_image(filename_input_path, filename_output_path)
            resize_image(filename_output_path, filename_output_path, 224, 224) # for testing new dataset model
            # resize_image(filename_output_path, filename_output_path, 1024, 1024)

            if dir_type == "masking":
                monochroming_image(filename_output_path, filename_output_path)

        pbar.update(1)

    pbar.close()