import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from model.arch import ModelBuilder
from model.loss import DetectionLoss
from torch.utils.data import DataLoader
from data.augment import CustomAugmentations
from data.loader import CustomDataset, collate_fn
from utils.utils import create_target_map , save_checkpoint

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
        # print("Loss components (xy, wh, obj, cls):", loss.item())

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

def train(model_config, data_config, criterion ,device, num_epochs, lr, batch_size, save_dir, Exp_name ):
    model = ModelBuilder.from_config(model_config).to(device)
    # model.apply(init_weights)

    augmentations = CustomAugmentations(p_flip=0.5)
    train_dataset = CustomDataset(data_config['train_images'], data_config['train_labels'],
                                   shuffle=True, normalize=True, augmentations=augmentations)
    val_dataset = CustomDataset(data_config['val_images'], data_config['val_labels'],
                                shuffle=False, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val_loss = float('inf')

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_loss= train_one_epoch(model,train_loader, optimizer, criterion, device)

        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch == 0:
            save_checkpoint(model, optimizer, epoch,val_loss, f'{save_dir}/{Exp_name}_first.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, f'{save_dir}/{Exp_name}_best_Checkpoint.pth')
            print(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")  # Log current LR

def get_args():
    parser = argparse.ArgumentParser(description="Object Detector Training Script")

    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training images')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels (YOLO format)')
    parser.add_argument('--val_data', type=str, required=False, help='Path to validation images')
    parser.add_argument('--val_labels', type=str, required=False, help='Path to validation labels (YOLO format)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--Exp_name', type=str, default='Exp1', help='The name of Exp used in the model name')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_config= {'train_images': args.train_data, 'train_labels': args.train_labels,
                  'val_images': args.val_data, 'val_labels': args.val_labels}

    criterion = DetectionLoss()
    train(model_config= args.config, data_config=data_config, criterion=criterion,
          device=args.device,  num_epochs=args.epochs, lr = args.lr,
          batch_size=args.batch_size, save_dir=args.save_dir, Exp_name= args.Exp_name )


