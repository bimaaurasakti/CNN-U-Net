from PIL import Image
from pre_processing import *

input_dir = "dataset/"
output_dir = "preprocessed-dataset/"

for process_type in os.listdir(input_dir):
    process_type_input_path = os.path.join(input_dir, process_type + '/')
    process_type_output_path = os.path.join(output_dir, process_type + '/')
    for dir_type in os.listdir(process_type_input_path):
        dir_type_input_path = os.path.join(process_type_input_path, dir_type + '/')
        dir_type_output_path = os.path.join(process_type_output_path, dir_type + '/')
        for filename in os.listdir(dir_type_input_path):
            filename_input_path = os.path.join(dir_type_input_path, filename)
            filename_output_path = os.path.join(dir_type_output_path, filename)
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                square_the_image(filename_input_path, filename_output_path)
                resize_image(filename_output_path, filename_output_path)