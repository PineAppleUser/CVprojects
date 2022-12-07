from torch.utils.data import Dataset
import torch

device = torch.device("mps" if torch.has_mps else "cpu")

class my_dataset(Dataset):

  def __init__(self, imgs, masks, transform = None):
    self.imgs = imgs
    self.masks = masks
    self.transform = transform

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, index):
    image = self.imgs[index]
    mask = self.masks[index]

    if self.transform is not None:
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]
    return image.to(device), mask.to(device)