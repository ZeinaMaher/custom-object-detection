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

def xywh_to_xyxy(box):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = box
    return [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]

def iou(box1, box2):
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def extract_pred(preds, conf_thresh=0.5):
    """
    Extract the prediction from the model output for each image

    Arg:
    preds: Tensor of shape [B, 5 + num_classes, H, W]
    conf_thresh: Confidence threshold to filter predictions

    Returns:
    detections: List of detections per image, each as [x1, y1, x2, y2, score, class_id]
    """
    B, C, H, W = preds.shape
    num_classes = C - 5
    preds = preds.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, 5 + num_classes]

    all_detections= []
    for b in range(B):
        pred= preds[b]
        
        bboxes = pred[:, :4]
        obj_scores= pred[:, 4]
        cls_scores= pred[:, 5:]

        scores = obj_scores.unsqueeze(1) * cls_scores  # shape: [H*W, num_classes]

        # Get the best class and its score
        conf, class_ids = scores.max(dim=1)  # shape: [H*W]


        mask= conf >= conf_thresh
        bboxes= bboxes[mask]
        conf= conf[mask]
        class_ids= class_ids[mask]

        boxes_xyxy = torch.stack([torch.tensor(xywh_to_xyxy(box), device=box.device) for box in bboxes], dim=0)

        # Final format: [x1, y1, x2, y2, score, class_id]
        detections = torch.cat([
            boxes_xyxy,
            conf.unsqueeze(1),
            class_ids.unsqueeze(1).float()
        ], dim=1)

        all_detections.append(detections)

    return all_detections

def nms(bboxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        bboxes (Tensor): [N, 4] in format [x1, y1, x2, y2]
        scores (Tensor): [N] confidence scores
        iou_threshold (float): IoU threshold for suppression

    Returns:
        keep (List[int]): indices of boxes to keep
    """
    if bboxes.size(0) == 0:
        return []

    x1= bboxes[:, 0]
    y1= bboxes[:, 1]
    x2= bboxes[:, 2]
    y2= bboxes[:, 3]

    # Areas of the boxes
    areas = (x2 - x1) * (y2 - y1)

    # sort scores 
    _, order = scores.sort(descending=True)

    keep_boxes=[]

    while order.size(0):
        i = order[0].item()
        keep_boxes.append(i)

        if order.size(0) == 1:
            break
        
        # Compute IoU of the highest score box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # Compute intersection
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU < threshold
        order = order[1:][iou < iou_threshold]

    return keep_boxes

def apply_nms(detections, iou_thresh=0.5):
    """
    Apply Non-Maximum Suppression (NMS) on the predicted bounding boxes.

    Args:
        detections: Tensor of shape [num_boxes, 6], where each box has the format [x1, y1, x2, y2, score, class_id].
        iou_thresh: The threshold for IoU to consider boxes as duplicates.

    Returns:
        kept_boxes: Tensor of kept boxes after applying NMS.
        kept_scores: Scores of the kept boxes.
        kept_class_ids: Class IDs of the kept boxes.
    """
    # Extract coordinates, scores, and class IDs
    boxes = detections[:, :4]  # [x1, y1, x2, y2]
    scores = detections[:, 4]  # Confidence score
    class_ids = detections[:, 5]  # Class IDs

    # Apply NMS for each class
    unique_class_ids = torch.unique(class_ids)
    
    kept_boxes = []
    kept_scores = []
    kept_class_ids = []

    for class_id in unique_class_ids:
        # Get the boxes corresponding to the current class
        class_mask = (class_ids == class_id)
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Perform NMS
        keep = nms(class_boxes, class_scores, iou_thresh)
        
        # Collect the kept boxes, scores, and class_ids
        kept_boxes.append(class_boxes[keep])
        kept_scores.append(class_scores[keep])
        kept_class_ids.append(class_ids[class_mask][keep])
    
    # Concatenate the results from all classes
    kept_boxes = torch.cat(kept_boxes, dim=0)
    kept_scores = torch.cat(kept_scores, dim=0)
    kept_class_ids = torch.cat(kept_class_ids, dim=0)
    
    return kept_boxes, kept_scores, kept_class_ids


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
#==================================================================================================
    # test model loading 

    # model, checkpoint= load_model_from_checkpoint('./configs/simple_model.yaml', './model_epoch_1.pth')
    # print(model, checkpoint)

#==================================================================================================
# Test postprocessing function 

# Fake prediction tensor [B, 5 + num_classes, H, W]
    B, C, H, W = 1, 5 + 3, 2, 2  # batch of 1 image, 3 classes, 2x2 grid
    preds = torch.rand(B, C, H, W)

    # Force high confidence on some cells to simulate real detections
    preds[0, 4, 0, 0] = 0.9  # objectness
    preds[0, 5:, 0, 0] = torch.tensor([0.1, 0.8, 0.1])  # class probs

    preds[0, 4, 1, 1] = 0.85
    preds[0, 5:, 1, 1] = torch.tensor([0.6, 0.2, 0.2])

    # Step 1: Extract predictions
    detections_per_image = extract_pred(preds, conf_thresh=0.5)

    # Step 2: Apply NMS
    for i, detections in enumerate(detections_per_image):
        print(f"\nImage {i}")
        print("Raw Detections:", detections)

        boxes, scores, class_ids = apply_nms(detections, iou_thresh=0.5)

        print("\nAfter NMS:")
        for b, s, c in zip(boxes, scores, class_ids):
            print(f"Box: {b.tolist()}, Score: {s.item():.2f}, Class: {int(c.item())}")