import torch
import argparse
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

    for i, class_metrics in enumerate(metrics['per_class']):
        print(f"Class {i} - Precision: {class_metrics['precision']}, Recall: {class_metrics['recall']}, F1: {class_metrics['f1']}, AP: {class_metrics['AP']}")

def get_args():
    parser = argparse.ArgumentParser(description="Object Detector Evaluation Script")

    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to YOLO-format label directory')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for NMS')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model, checkpoint = load_model_from_checkpoint(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device)

    dataset = CustomDataset(
        img_dir=args.images,
        labels_dir=args.labels,
        shuffle=False,
        normalize=True
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    evaluate(model= model, dataloader=dataloader , device= args.device,
             num_classes=args.num_classes, conf_threshold= args.conf_thresh,
             iou_threshold= args.iou_thresh)
