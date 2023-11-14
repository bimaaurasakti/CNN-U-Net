from augmentation import *
from preprocessing import *

# Split image process
# input_directory = 'dataset'
# output_directory = 'splitted-dataset'

# input_directory_images = os.path.join(input_directory, 'images/')
# input_directory_masking = os.path.join(input_directory, 'masking/')
# output_directory_images = os.path.join(output_directory, 'images/')
# output_directory_masking = os.path.join(output_directory, 'masking/')

# if not os.path.exists(output_directory_images):
#     os.makedirs(output_directory_images)
#     os.makedirs(output_directory_masking)

# image_files = os.listdir(input_directory_images)
# mask_files = os.listdir(input_directory_masking)

# total_iterations = len(image_files)
# pbar = tqdm(total=total_iterations, desc="Split into Four Images", unit="iter")

# for image_file, mask_file in zip(image_files, mask_files):
#     filename_input_images_path = os.path.join(input_directory_images, image_file)
#     filename_input_masking_path = os.path.join(input_directory_masking, mask_file)

#     split_into_four_parts(filename_input_images_path, output_directory_images)
#     split_into_four_parts(filename_input_masking_path, output_directory_masking)

#     pbar.update(1)

# pbar.close()


# Preprocessing process
input_directory = 'dataset'
output_directory = 'preprocessed-dataset'
final_test_directory = 'final-dataset/test'

input_directory_images = os.path.join(input_directory, 'images/')
input_directory_masking = os.path.join(input_directory, 'masking/')
output_directory_images = os.path.join(output_directory, 'images/')
output_directory_masking = os.path.join(output_directory, 'masking/')
final_test_directory_images = os.path.join(final_test_directory, 'images/')
final_test_directory_masking = os.path.join(final_test_directory, 'masking/')

if not os.path.exists(output_directory_images):
    os.makedirs(output_directory_images)
    os.makedirs(output_directory_masking)
    os.makedirs(final_test_directory_images)
    os.makedirs(final_test_directory_masking)   

image_files = os.listdir(input_directory_images)
mask_files = os.listdir(input_directory_masking)
data_length = len(image_files)
n_90_percent = int(0.9 * data_length)

pbar = tqdm(total=data_length, desc="Preprocessing Radiograf", unit="iter")
for index, image_file in enumerate(image_files):
    filename_input_images_path = os.path.join(input_directory_images, image_file)
    
    # first dataset split to get test dataset
    if index < n_90_percent:
        filename_output_images_path = os.path.join(output_directory_images, image_file)
    else:
        filename_output_images_path = os.path.join(final_test_directory_images, image_file)

    make_square(filename_input_images_path, filename_output_images_path, 224)
    resize_image_to_file(filename_output_images_path, filename_output_images_path, 224, 224)
    # shutil.copyfile(filename_input_images_path, filename_output_images_path)

    pbar.update(1)

pbar.close()

pbar = tqdm(total=data_length, desc="Preprocessing Ground Truth", unit="iter")
for index, mask_file in enumerate(mask_files):
    filename_input_masking_path = os.path.join(input_directory_masking, mask_file)

    # first dataset split to get test dataset
    if index < n_90_percent:
        filename_output_masking_path = os.path.join(output_directory_masking, mask_file)
    else:
        filename_output_masking_path = os.path.join(final_test_directory_masking, mask_file)

    make_square(filename_input_masking_path, filename_output_masking_path, 224)
    resize_image_to_file(filename_output_masking_path, filename_output_masking_path, 224, 224)
    monochroming_image_to_file(filename_output_masking_path, filename_output_masking_path)
    # shutil.copyfile(filename_input_masking_path, filename_output_masking_path)

    pbar.update(1)

pbar.close()