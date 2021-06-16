#!/usr/bin/env bash

save_folder="work_space"
datasets_root1="/data3/panjinquan/MPII/trainval.txt"
datasets_root2="/data3/panjinquan/VOCdevkit/VOC2012/trainval.txt"
datasets_root3="/data3/panjinquan/VOCdevkit/VOC2007/trainval.txt"
datasets_root4="/data3/panjinquan/COCO/VOC/trainval.txt"
datasets_root5="/data3/panjinquan/crowdhuman/trainval.txt"

#val_path="/data3/panjinquan/COCO/VOC"
#val_path="/data3/panjinquan/VOCdevkit/VOC2007"
val_path="/data3/panjinquan/MPII/test.txt"

#datasets=$datasets_root1
train_path=$datasets_root1" "$datasets_root2" "$datasets_root3
#train_path=$datasets_root1" "$datasets_root4
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3" "$datasets_root4" "$datasets_root5
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3" "$datasets_root4
#train_path=$datasets_root1" "$datasets_root5
#train_path=$datasets_root1" "$datasets_root5


echo $train_path


#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u train.py \
python3 -u train.py \
    --train_path $train_path \
    --val_path $val_path \
    --data_type "VOC" \
    --network "RFB_person" \
    --num_classes 2 \
    --num_workers 4 \
    --gpu_id 7 \
    --input_size 480 360 \
    --batch_size 64 \
    --max_epoch 200 \
    --lr 0.1 \
    --milestones "60,100,150" \
    --save_folder $save_folder \
    --check \
    --polyaxon

