import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from functools import reduce
from loss import dice_loss


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

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

# def tranform_binary(inp):
#     new_inp = torch.zeros((inp.shape[0], 1, inp.shape[2], inp.shape[3]))
#     for batch in range(new_inp.shape[0]):
#         for i in range(new_inp.shape[2]):
#             for j in range(new_inp.shape[3]):
#                 if inp[batch, 0, i, j] > 0:
#                     new_inp[batch, 0, i, j] = 1
#                 else:
#                     new_inp[batch, 0, i, j] = 0
#     return new_inp

def transform_binary(inp):
    return (inp > 0).astype(int)

def calculate_accuracy(predicts, labels):
    # Mengonversi tensor PyTorch ke array NumPy
    predicts = predicts.cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)

    # Transformasi biner
    predicts = transform_binary(predicts)

    # Menghitung jumlah piksel yang benar dalam setiap batch
    correct_predictions = (predicts == labels).sum(axis=(1, 2, 3))

    # Menghitung total piksel dalam setiap batch
    total_pixels = predicts.shape[1] * predicts.shape[2] * predicts.shape[3]

    # Menghitung akurasi untuk setiap batch
    accuracies = correct_predictions / total_pixels

    return accuracies.tolist()