import Augmentor


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
    p.sample(5000)  # Misalnya, menghasilkan 5000 sampel gambar yang telah diaugmentasi