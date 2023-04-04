"""# Prepare Dataset and DataLoader"""

import glob
import numpy as np
import torch

from torchvision.transforms.functional import adjust_gamma
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from pre_processing import *

class SimDataset(Dataset):
  def __init__(self, image_path, mask_path):
    self.input_images = glob.glob(str(image_path) + str('/*'))
    self.target_masks =  glob.glob(str(mask_path) + str('/*'))

  def __len__(self):
    return len(self.target_masks)

  def __getitem__(self, idx):
    image = self.input_images[idx]
    image_as_image = Image.open(image) 
    image_as_np = np.asarray(image_as_image)

    # Normalize the image
    image_as_np = clahe_equalized(image_as_np)
    image_as_np = adjust_gamma(image_as_np, 1.2)
    image_as_np = normalization(image_as_np, max=1, min=0)
    image_as_np = np.expand_dims(image_as_np, axis=0)
    image_as_tensor = torch.from_numpy(image_as_np).float()

    mask = self.target_masks[idx]
    mask_as_mask = Image.open(mask)
    mask_as_np = np.asarray(mask_as_mask)
    
    mask_as_np = mask_as_np/255
    mask_as_np = np.expand_dims(mask_as_np[..., 0], axis=0)
    mask_as_tensor = torch.from_numpy(mask_as_np).float()

    return (image_as_tensor, mask_as_tensor)

train_set = SimDataset('./dataset/train/images', './dataset/train/masks')
val_set = SimDataset('./dataset/val/images', './dataset/val/masks')
test_set = SimDataset('./dataset/test/images', './dataset/test/masks')

image_datasets = {
  'train': train_set, 'val': val_set, 'test': test_set
}

batch_size = 25

dataloaders = {
    # 'train': DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0),
    # 'val': DataLoader(val_set, batch_size=10, shuffle=True, num_workers=0),
    # 'test': DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)
    'train': DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
}

n_train = len(image_datasets['train'])
n_val = len(image_datasets['val'])
n_test = len(image_datasets['test'])

print("n_train=",n_train)
print("n_val=",n_val)
print("n_val=",n_test)

"""# Create U-NET Model Function"""

import torch.nn as nn
from torchsummary import summary

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

"""# Train U-NET"""

import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from collections import defaultdict
from torch.optim import lr_scheduler
from tqdm import tqdm


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
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

def train_model(model, optimizer, scheduler, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),'Unet_Model.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_class = 1
model = UNet(num_class).to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=100)