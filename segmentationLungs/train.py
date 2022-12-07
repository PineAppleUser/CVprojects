import torch
from tqdm import tqdm
import torch.optim as optim
from utils import DiceBCELoss, load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs, get_loaders
from model import unet
from model_2 import UNET

device_2use = torch.device("mps" if torch.has_mps else "cpu")
load_model = False

def train_fun(loader, model, optimizer, loss_func, scaler):
  loop = tqdm(loader)

  device_2use = torch.device("mps" if torch.has_mps else "cpu")
  for batch_idx, (data, targets) in enumerate(loop):
    data = data.to(device = device_2use)
    targets = targets.unsqueeze(1).to(device = device_2use)

    #forward and optimized for faster perfrormance
    #with torch.cuda.amp.autocast():
    predictions = model(data)
    loss = loss_func(predictions, targets)

    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # update tqdm loop
    loop.set_postfix(loss=loss.item())

def main():
    #model = unet().to(device_2use)
    model = UNET(in_channels=1, out_channels=1).to(device_2use)
    loss_func = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    path = 'segmentationVolume/'
    train_loader, val_loader = get_loaders(path = path)

    if load_model:
        load_checkpoint(torch.load("saved_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=device_2use)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(100):
        train_fun(train_loader, model, optimizer, loss_func, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device_2use)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="/content/", device=device_2use
        )
if __name__ == "__main__":
    main()