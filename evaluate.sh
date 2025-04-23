python3 ./evaluate.py --config './configs/model_v1.yaml' --checkpoint './checkpoints/Exp1_best_model.pth' \
                      --images './dataset/filtered_data/valid/images' --labels './dataset/filtered_data/valid/labels' \
                      --batch_size 8 --num_classes 4 --conf_thresh 0.5 --iou_thresh 0.5