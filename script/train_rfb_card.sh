#!/usr/bin/env bash

save_folder="work_space/card"
datasets_root1="/data3/panjinquan/dataset/card_datasets/yolo_det/CardData4det/train.txt"
val_path="/data3/panjinquan/dataset/card_datasets/yolo_det/CardData4det/val.txt"


datasets=$datasets_root1
echo $datasets
echo $val_path


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u train.py \
    --train_path $datasets \
    --val_path $val_path \
    --data_type "VOC" \
    --net_type "RFB" \
    --priors_type "card" \
    --num_workers 8 \
    --gpu_id 6 \
    --input_size 320 320 \
    --batch_size 64 \
    --max_epoch 200 \
    --lr 0.1 \
    --milestones "60,100,150" \
    --save_folder $save_folder \
    --flag  ""\
    --check \


