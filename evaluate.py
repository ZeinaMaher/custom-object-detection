import torch
from tqdm import tqdm
from model.arch import ModelBuilder
from torch.utils.data import DataLoader
from data.loader import CustomDataset, collate_fn
from utils.metrics import compute_detection_metrics
from utils.utils import load_model_from_checkpoint, postprocess_pred, postprocess_gt

def evaluate(model, dataloader, device, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    model.eval()
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for imgs, labels, filenames in tqdm(dataloader):
            imgs = imgs.to(device)

            # Get model predictions and apply postprocessing
            preds = model(imgs)
            batch_detections = postprocess_pred(preds, conf_thresh=conf_threshold, iou_thresh=iou_threshold)

            gt_boxes = postprocess_gt(labels, device)

            # Store predictions and ground truths for metric computation
            all_preds.extend(batch_detections)  # Assuming detections are in the format [x1, y1, x2, y2, score, class_id]
            all_gts.extend(gt_boxes)  # Store ground truth labels

            # print(f"Predictions for {filenames}: {batch_detections}")  # For debugging/inspection

    # Flatten lists of predictions and ground truths
    # only if all_preds is a list of list of tensors
    all_preds = [torch.cat(dets, dim=0) if isinstance(dets, list) and len(dets) > 0 else torch.empty((0, 6), device=device) for dets in all_preds]
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Compute metrics
    metrics = compute_detection_metrics(all_preds, all_gts, num_classes, iou_threshold=iou_threshold)

    print("Evaluation Metrics:")
    print(f"Overall Precision: {metrics['overall']['precision']}")
    print(f"Overall Recall: {metrics['overall']['recall']}")
    print(f"Overall F1: {metrics['overall']['f1']}")
    print(f"Overall mAP: {metrics['overall']['mAP']}")

    # Optionally print per-class metrics
    for i, class_metrics in enumerate(metrics['per_class']):
        print(f"Class {i} - Precision: {class_metrics['precision']}, Recall: {class_metrics['recall']}, F1: {class_metrics['f1']}, AP: {class_metrics['AP']}")

if __name__ == "__main__":
    config_path = "configs/simple_model.yaml"
    check_path = "./model_epoch_1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(config_path=config_path, checkpoint_path=check_path, device=device)

    val_dataset = CustomDataset(
        img_dir="./dataset/samples/valid/images",
        labels_dir="./dataset/samples/valid/labels",
        shuffle=False,
        normalize=True
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Set the number of classes based on your dataset
    num_classes = 2  # Adjust accordingly
    evaluate(model, val_loader, device, num_classes, 0)
