#!/usr/bin/env bash

save_folder="work_space/RFB_face_person"

datasets_root1="/data3/panjinquan/dataset/face_person/MPII/trainval.txt"
datasets_root2="/data3/panjinquan/dataset/face_person/VOCdevkit/VOC2012/trainval.txt"
datasets_root3="/data3/panjinquan/dataset/face_person/VOCdevkit/VOC2007/trainval.txt"
datasets_root4="/data3/panjinquan/dataset/face_person/COCO/VOC/trainval.txt"
datasets_root5="/data3/panjinquan/dataset/face_person/SMTC/trainval.txt"
#datasets_root6="/data3/panjinquan/dataset/face_person/crowdhuman/trainval.txt"

#val_path="/data3/panjinquan/dataset/face_person/COCO/VOC"
#val_path="/data3/panjinquan/dataset/face_person/VOCdevkit/VOC2007"
val_path="/data3/panjinquan/dataset/face_person/MPII/test.txt"

train_path=$datasets_root1
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3
#train_path=$datasets_root1" "$datasets_root4
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3" "$datasets_root5" "$datasets_root5" "$datasets_root5" "$datasets_root5" "$datasets_root5
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3" "$datasets_root4" "$datasets_root5" "$datasets_root5" "$datasets_root5
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3" "$datasets_root4
#train_path=$datasets_root1" "$datasets_root5
#train_path=$datasets_root1" "$datasets_root5


echo $train_path

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u train.py \
    --train_path $train_path \
    --val_path $val_path \
    --data_type "VOC" \
    --net_type "RFB" \
    --priors_type "face_person" \
    --num_workers 4 \
    --gpu_id 3 \
    --input_size 320 320 \
    --batch_size 128 \
    --max_epoch 200 \
    --lr 0.1 \
    --milestones "60,100,150" \
    --save_folder $save_folder \
    --flag  "v2_ssd"\
    --check \
#    --resume "work_space/RFB_landms/rfb1.0_face_320_320_wider_face_add_lm_10_10_anchor_v4_20210329174602/model/best_model_rfb_160_loss6.6319.pth"


