import torch
from model.arch import ModelBuilder
from torch.utils.data import DataLoader
from data.loader import CustomDataset, collate_fn
from utils.utils import load_model_from_checkpoint

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for imgs, labels, filenames in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            print(f"Predictions for {filenames}: {preds.shape}")  # Placeholder


if __name__ == "__main__":
    config_path = "configs/simple_model.yaml"
    check_path= "./model_epoch_1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint= load_model_from_checkpoint(config_path=config_path, checkpoint_path=check_path, device=device)

    val_dataset = CustomDataset(
        img_dir="./dataset/filtered_data/valid/images",
        labels_dir="./dataset/filtered_data/valid/labels",
        shuffle=False,
        normalize=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn,)

    evaluate(model, val_loader, device)
