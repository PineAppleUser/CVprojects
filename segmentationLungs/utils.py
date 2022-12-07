from dataset import my_dataset
from torch.utils.data import DataLoader, random_split
import os
import glob
import numpy as np
import scipy
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


# helping function to load dicom files from folders
def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


# function to get imgs and masks and write them into arrays
def get_img_msk(path):
    imgs = []
    masks = []

    for patient in os.listdir(path + 'subset'):
        subfold = glob.glob(os.path.join((path + 'subset/' + patient), '*'))
        sub_subfold = glob.glob(os.path.join(subfold[0], '*'))
        dirr = glob.glob(os.path.join(sub_subfold[0], '*'))
        for unit in dirr:
            if unit[-4:] !='json':
                dirr = unit
                break
        img_zyx = load_dicom(dirr)
        # value got from ploting HU values
        img_zyx = np.clip(img_zyx, a_min=-250, a_max=250)

        mask = nib.load(path + 'subset_masks/' + patient + '/' + patient + '_effusion_first_reviewer.nii.gz')
        mask = mask.get_fdata().transpose(2, 0, 1)
        mask = scipy.ndimage.rotate(mask, 90, (1, 2))

        for i in range(len(img_zyx)):
            imgs.append(img_zyx[i])
            # approximate binorization based on the given dataset values in the mask where background values are close to 0
            masks.append(np.where(mask[i] > 0.05, 1.0, 0.0))
    return np.float32(imgs), np.float32(masks)

#getting both train and val loader based on the train_size percentage
def get_loaders(path, train_size=70, btch_size=64):
    # getting arrays of images and masks
    imgs, masks = get_img_msk(path)

    data_transf = A.Compose([A.Resize(height=512, width=512),
                             ToTensorV2()])
    # turning arrays into custom dataset for pytorch
    data = my_dataset(imgs, masks, data_transf)

    # splitting based in the train_size value which is percent out of 100
    train_size = (len(data) * train_size) // 100
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    # getting train and val data into DataLoader, and shuffle only train_loader
    train_loader = DataLoader(train_data, batch_size=btch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=btch_size, shuffle=False)
    return train_loader, val_loader


# custom loss function which is Dice loss with binary cross-entropy which showed the best results in the papers
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        # flattening
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = bce + dice_loss

        return dice_bce

#func to save model checkpoint
def save_checkpoint(state, filename="saved_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

#func to load model checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

#checking dice coefficient during training as an evaluation metric
def check_accuracy(loader, model, device="mps"):
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

#function to save prediction as an image
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="mps"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
