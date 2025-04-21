from utils.utils import iou
import torch

def cal_detection_result(preds, gts, num_classes, iou_threshold=0.5):
    """
    calculate FP, FN, TP

    Args:
        preds: List of detections per image, each as [x1, y1, x2, y2, score, class_id].
        gts: List of ground truth boxes per image, each as [x1, y1, x2, y2, class_id].
        num_classes: number of classes
        iou_threshold: IoU threshold for considering a detection as a TP.
    
    Returns:
     dict: {"tp": Tensor[num_classes], "fp": Tensor[num_classes], "fn": Tensor[num_classes]}

    """
    tp = torch.zeros(num_classes, dtype=torch.int)
    fp = torch.zeros(num_classes, dtype=torch.int)
    fn = torch.zeros(num_classes, dtype=torch.int)

    # Iterate over each image
    for det, gt in zip(preds, gts):
        pred_boxes = det[:, :4]  # [x1, y1, x2, y2]
        scores = det[:, 4]  # Confidence scores
        pred_class_ids = det[:, 5].long()  # Predicted class ids

        gt_boxes = gt[:, :4]  # Ground truth boxes [x1, y1, x2, y2]
        gt_class_ids = gt[:, 4].long()  # Ground truth class ids
        
        matched_pred_indices = set()

        # Iterate through each detected box
        for i, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_class_ids)):
            matched = False
            for j, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_class_ids)):
                if j in matched_pred_indices or pred_class != gt_class:
                    continue

                iou_score = iou(gt_box.tolist(), pred_box.tolist())
                if iou_score >= iou_threshold:
                    tp[gt_class] += 1
                    matched_pred_indices.add(j)
                    matched = True
                    break

            if not matched:
                fn[gt_class] += 1

        # Any unmatched prediction is a false positive
        for j, pred_class in enumerate(pred_class_ids):
            if j not in matched_pred_indices:
                fp[pred_class] += 1

    return {"tp": tp, "fp": fp, "fn": fn}

def precision_score(tp, fp):
    return tp.float() / (tp + fp).clamp(min=1)

def recall_score(tp, fn):
    return tp.float() / (tp + fn).clamp(min=1)

def f1_score(tp, fp, fn):
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    return 2 * (precision * recall) / (precision + recall).clamp(min=1e-8)

def compute_ap(tp, fp, num_gt):
    # Cumulative sum of TP and FP
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # 11-point interpolation (Pascal VOC style)
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        prec = precision[recall >= t].max() if (recall >= t).any() else 0
        ap += prec / 11
    return ap

def compute_detection_metrics(preds, gts, num_classes, iou_threshold=0.5, epsilon=1e-6):
    """
    Computes detection metrics (precision, recall, F1) per class and overall.

    Args:
        preds: List of detections per image, each as [x1, y1, x2, y2, score, class_id].
        gts: List of ground truth boxes per image, each as [x1, y1, x2, y2, class_id].
        num_classes: Total number of classes.
        iou_threshold: IoU threshold to consider TP.
        epsilon: Small value to avoid division by zero.

    Returns:
        dict with:
            - 'per_class': List of dicts per class with 'precision', 'recall', 'f1'
            - 'overall': Dict with overall 'precision', 'recall', 'f1'
    """
    result = cal_detection_result(preds, gts, num_classes, iou_threshold)
    tp, fp, fn = result['tp'].float(), result['fp'].float(), result['fn'].float()

    precision_per_class = precision_score(tp, fp)
    recall_per_class = recall_score(tp, fn)
    f1_per_class = f1_score(tp, fp, fn)

    # Compute AP for each class
    ap_per_class = []
    for cls in range(num_classes):
        # Number of ground truths for this class
        num_gt = tp[cls] + fn[cls]
        ap = compute_ap(tp[cls], fp[cls], num_gt if num_gt > 0 else 1)
        ap_per_class.append(ap)

    per_class_metrics = []
    for i in range(num_classes):
        per_class_metrics.append({
            "precision": precision_per_class[i].item(),
            "recall": recall_per_class[i].item(),
            "f1": f1_per_class[i].item(),
            "AP": ap_per_class[i].item()
        })

    # Compute mAP as the mean of AP across classes
    mAP = torch.tensor(ap_per_class).mean()

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()

    overall_precision = precision_score(total_tp, total_fp)
    overall_recall = recall_score(total_tp, total_fn)
    overall_f1 =f1_score(total_tp, total_fp,total_fn)

    return {
        "per_class": per_class_metrics,
        "overall": {
            "precision": overall_precision.item(),
            "recall": overall_recall.item(),
            "f1": overall_f1.item(),
            "mAP": mAP.item(),
        }
    }


if __name__ == "__main__":
# test cal_detection_result function
    # preds = [
    # torch.tensor([
    #     [10, 10, 50, 50, 0.9, 0],  # matches GT
    #     [60, 60, 100, 100, 0.8, 1],  # no GT match (FP)
    # ]),
    # torch.tensor([
    #     [20, 20, 60, 60, 0.95, 0],  # matches GT
    #     [30, 30, 70, 70, 0.6, 0],   # duplicate prediction (FP)
    # ])
    # ]

    # gts = [
    #     torch.tensor([
    #         [10, 10, 50, 50, 0],  # TP
    #         [100, 100, 140, 140, 2],  # FN, no prediction for class 2
    #     ]),
    #     torch.tensor([
    #         [20, 20, 60, 60, 0],  # TP
    #     ])
    # ]

    # result = cal_detection_result(preds, gts, num_classes=3, iou_threshold=0.5)

    # print("TP:", result["tp"])
    # print("FP:", result["fp"])
    # print("FN:", result["fn"])

#============================================================================================

# test compute metrics function 
# Predictions (preds) - [x1, y1, x2, y2, score, class_id]
    preds = [
        torch.tensor([[10, 10, 20, 20, 0.9, 0], [30, 30, 40, 40, 0.8, 1]]),  # image 1 predictions
        torch.tensor([[50, 50, 60, 60, 0.7, 0]])  # image 2 prediction
    ]

    # Ground truths (gts) - [x1, y1, x2, y2, class_id]
    gts = [
        torch.tensor([[10, 10, 20, 20, 0], [30, 30, 40, 40, 1]]),  # image 1 ground truths
        torch.tensor([[50, 50, 60, 60, 0]])  # image 2 ground truth
    ]

    # Number of classes
    num_classes = 2

    # Compute the metrics
    metrics = compute_detection_metrics(preds, gts, num_classes, iou_threshold=0.5)
    print(metrics)