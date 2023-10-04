import Augmentor
import os

from tqdm.notebook import tqdm
from PIL import Image


def augment_images_in_directory():
    input_directory_images = 'temp-augmented-dataset/images'
    input_directory_masking = 'temp-augmented-dataset/masking'

    # Membuat objek pipeline untuk direktori gambar
    p = Augmentor.Pipeline(input_directory_images)

    # Menambahkan transformasi yang diinginkan
    p.ground_truth(input_directory_masking)
    p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)

    # Menentukan jumlah sampel yang dihasilkan
    p.sample(10000)  # Misalnya, menghasilkan 10000 sampel gambar yang telah diaugmentasi

def split_into_four_parts(image_file, output_location):
    image = Image.open(image_file)
    _, nama_file = os.path.split(image_file)

    width, height = image.size
    half_width, half_height = width // 2, height // 2

    # Crop the four parts for image
    top_left_image = image.crop((0, 0, half_width, half_height))
    top_right_image = image.crop((half_width, 0, width, half_height))
    bottom_left_image = image.crop((0, half_height, half_width, height))
    bottom_right_image = image.crop((half_width, half_height, width, height))

    # Save the cropped parts to the output directory
    top_left_image.save(os.path.join(output_location, f"{nama_file}_top_left.jpg"))
    top_right_image.save(os.path.join(output_location, f"{nama_file}_top_right.jpg"))
    bottom_left_image.save(os.path.join(output_location, f"{nama_file}_bottom_left.jpg"))
    bottom_right_image.save(os.path.join(output_location, f"{nama_file}_bottom_right.jpg"))