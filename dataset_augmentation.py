from augmentation import *
from helper import *
import shutil


if not os.path.exists("temp-augmented-dataset"):
    shutil.copytree("preprocessed-dataset", "temp-augmented-dataset")
augment_images_in_directory()

# split the output data (because original image and groundtruth in the same directory)
# include second split to get training and validation dataset
split_and_remove()

# renaming file for equalizing original and groundtruth image
rename_augmented_file("augmented-dataset/images")
rename_augmented_file("augmented-dataset/masking")

# # split 90% of original data that successfully augmented to 5000 data to 80% and 20%
# # 80% --> training | 20% --> validation
split_to_80_and_20("augmented-dataset/images", "images")
split_to_80_and_20("augmented-dataset/masking", "masking")