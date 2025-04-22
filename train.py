
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from model.arch import ModelBuilder
from model.loss import DetectionLoss
from data.loader import CustomDataset, collate_fn
from data.augment import CustomAugmentations
from utils.utils import create_target_map , save_checkpoint

def init_weights(m):
    """Initialize only unfrozen layers"""
    if not hasattr(m, 'weight') or not m.weight.requires_grad:
        return  # Skip frozen layers
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight.requires_grad:  # Only init trainable BatchNorm
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss= 0 

    num_classes = model.head.num_classes
    for images, targets, _ in tqdm(dataloader):
        images = images.to(device)

        preds = model(images)

        _, _, H, W = preds.shape
        target_map = create_target_map(
            targets, 
            grid_size=(H, W),  # Grid size (H, W) from your model output
            num_classes=num_classes,
            device= device
        )
        loss = criterion(preds, target_map) # make sure from the shape
        print("Loss components (xy, wh, obj, cls):", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            preds = model(images)

            _, _, H, W = preds.shape
            target_map = create_target_map(
                targets,
                grid_size=(H, W),
                num_classes=model.head.num_classes,
                device=device
            )

            loss = criterion(preds, target_map)
            val_loss += loss.item()

    return val_loss / len(dataloader)

def train(model_config, data_config, criterion, optimizer ,device, num_epochs, lr, batch_size, file_path=None ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelBuilder.from_config("./configs/model_v1.yaml").to(device)
    # model.apply(init_weights)

    augmentations = CustomAugmentations(p_flip=0.5)
    # train_dataset = CustomDataset('./dataset/filtered_data/train/images',
    #                               './dataset/filtered_data/train/labels',
    #                                shuffle=True, normalize=True, augmentations=augmentations)
    # val_dataset = CustomDataset('./dataset/filtered_data/valid/images',
    #                             './dataset/filtered_data/valid/labels',
    #                             shuffle=False, normalize=True)
    train_dataset = CustomDataset('./dataset/samples/valid/images',
                                  './dataset/samples/valid/labels',
                                   shuffle=True, normalize=True, augmentations=augmentations)
    val_dataset = CustomDataset('./dataset/samples/valid/images',
                                './dataset/samples/valid/labels',
                                shuffle=False, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Start smaller
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val_loss = float('inf')

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_loss= train_one_epoch(model,train_loader, optimizer, criterion, device)

        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch == 0:
            save_checkpoint(model, optimizer, epoch,val_loss, f'checkpoints/model_epoch_{epoch+1}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/Exp1_best_model.pth')
            print(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")  # Log current LR


if __name__ == "__main__":
    criterion = DetectionLoss()
    train(model_config= '.\configs\model_v1.yaml',
          data_config='', criterion=criterion, optimizer=None,
          device=None,  num_epochs=300, lr = 1e-3, batch_size=1 )

          #1e-4