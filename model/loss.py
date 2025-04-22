import torch 
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')  # For coordinates
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')  # For class/obj scores
        self.lambda_coord = lambda_coord  # Weight for box coordinates
        self.lambda_noobj = lambda_noobj  # Weight for no-object regions

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor [B, 5+C, H, W] (x, y, w, h, obj_score, class_scores)
            targets: Tensor [B, 5+C, H, W] (same format as preds)
        Returns:
            loss: Scalar tensor
        """
        B, _, H, W = preds.shape
        
        # --- Mask for cells with objects ---
        obj_mask = targets[..., 4] == 1  # Shape [B, H, W]
        noobj_mask = ~obj_mask

        # --- 1. Coordinate Loss (x, y, w, h) ---
        pred_xy = torch.sigmoid(preds[..., 0:2])  # x,y are sigmoid-activated
        pred_wh = preds[..., 2:4]                # w,h are direct predictions
        
        # Only compute for cells with objects
        loss_x = self.mse(pred_xy[obj_mask], targets[..., 0:2][obj_mask])
        loss_w = self.mse(pred_wh[obj_mask], targets[..., 2:4][obj_mask])
        coord_loss = (loss_x + loss_w) * self.lambda_coord

        # --- 2. Objectness Loss ---
        pred_obj = preds[..., 4]  # Raw logits
        # For cells WITH objects
        loss_obj = self.bce(pred_obj[obj_mask], targets[..., 4][obj_mask])
        # For cells WITHOUT objects (downweighted)
        loss_noobj = self.bce(pred_obj[noobj_mask], targets[..., 4][noobj_mask])
        obj_loss = loss_obj + self.lambda_noobj * loss_noobj

        # --- 3. Class Loss ---
        pred_cls = preds[..., 5:]  # Class logits
        loss_cls = self.bce(pred_cls[obj_mask], targets[..., 5:][obj_mask])

        # --- Total Loss ---
        total_loss = (coord_loss + obj_loss + loss_cls) / B  # Normalize by batch size
        return total_loss