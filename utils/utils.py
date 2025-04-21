import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from model.arch import ModelBuilder


def create_target_map(targets, grid_size, num_classes, device):
    """
    Create target map for each image.
    
    Args:
        targets: A list of tensors, each representing a batch of ground truth boxes.
        grid_size: The size of the grid (H, W).
        num_classes: The number of classes.
        device: The device to move tensors to.
    
    Returns:
        target_map: A tensor with shape [B, 5 + num_classes, H, W].
    """
    B = len(targets) 
    H, W = grid_size
    target_map = torch.zeros(B, 5 + num_classes, H, W, device=device)

    for i, target in enumerate(targets):
        for t in target:
            class_id, x_center, y_center, w, h = t  # Assuming each target is of the form [class_id, x_center, y_center, w, h]

            # Convert to grid cell coordinates (normalize by grid size)
            grid_x = min(int(x_center * W), W - 1)
            grid_y = min(int(y_center * H), H - 1)

            # Fill the target map for this specific grid cell
            target_map[i, 0, grid_y, grid_x] = x_center
            target_map[i, 1, grid_y, grid_x] = y_center
            target_map[i, 2, grid_y, grid_x] = w
            target_map[i, 3, grid_y, grid_x] = h
            target_map[i, 4, grid_y, grid_x] = 1  # Objectness score

            # Fill class probabilities
            # Ensure the class_id is within bounds
            if 0 <= int(class_id) < num_classes:
                target_map[i, 5 + int(class_id), grid_y, grid_x] = 1  # Set class probability to 1
    return target_map
    
def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Latest loss value
        path (str): output path to save checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at: {path}")

def load_model_from_checkpoint(config_path, checkpoint_path, device=None):
    """
    Loads a model and its weights from a checkpoint file.

    Args:
        config_path (str): Path to the model config YAML.
        checkpoint_path (str): Path to the saved checkpoint file.
        device (torch.device or str): Target device ('cpu' or 'cuda').

    Returns:
        model (nn.Module): Model with loaded weights.
        checkpoint (dict): Full checkpoint dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Rebuild model architecture
    model = ModelBuilder.from_config(config_path)
    model.to(device)

    # Step 2: Load checkpoint and apply weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"=Model loaded from {checkpoint_path} onto {device}")
    return model, checkpoint


if __name__ == "__main__":

    #to test target map 
    # targets = (
    #     torch.tensor([[3.0, 0.1320, 0.2031, 0.0805, 0.1133],  # Object 1
    #                 [3.0, 0.3133, 0.0984, 0.0961, 0.1297],  # Object 2
    #                 [3.0, 0.5672, 0.1586, 0.0672, 0.1297]]),  # Object 3
    #     torch.tensor([[3.0, 0.0781, 0.3844, 0.0562, 0.0852],  # Object 1
    #                 [3.0, 0.1977, 0.3750, 0.0477, 0.0828],  # Object 2
    #                 [8.0, 0.5672, 0.3758, 0.0383, 0.0750]]),  # Object 3
    # )

    # target_map = create_target_map(targets, grid_size=(30, 30), num_classes=2, device='cpu')

    # print(target_map.shape)  # Should print: torch.Size([2, 7, 30, 30]) for a batch of size 2

    # test model loading 

    model, checkpoint= load_model_from_checkpoint('./configs/simple_model.yaml', './model_epoch_1.pth')
    print(model, checkpoint)