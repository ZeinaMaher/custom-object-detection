python3 ./train.py --config './configs/model_v1.yaml' --epochs 300 --batch_size 64 --lr 0.001 \
                   --train_data './dataset/filtered_data/train/images' \
                   --train_labels './dataset/filtered_data/train/labels' \
                   --val_data './dataset/filtered_data/valid/images' \
                   --val_labels './dataset/filtered_data/valid/labels' \
                   --save_dir './checkpoints' --Exp_name 'Exp1'