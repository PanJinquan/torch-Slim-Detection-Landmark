#!/usr/bin/env bash

save_folder="work_space/RFB_landms_v2"


datasets_root1="/data3/panjinquan/dataset/face_person/wider_face_add_lm_10_10/trainval.txt"
datasets_root2="/data3/panjinquan/dataset/face_person/dmai_data/trainval.txt"
datasets_root3="/data3/panjinquan/dataset/face_person/FDDB/trainval.txt"


val_path="/data3/panjinquan/dataset/face_person/wider_face_add_lm_10_10/test.txt"
train_path=$datasets_root1
#train_path=$datasets_root1" "$datasets_root2" "$datasets_root3
echo $train_path

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u train_for_landm.py \
    --train_path $train_path \
    --val_path $val_path \
    --data_type "VOCLandm" \
    --net_type "RFB_landm" \
    --priors_type "face" \
    --num_workers 4 \
    --gpu_id 3 \
    --input_size 320 320 \
    --batch_size 128 \
    --max_epoch 200 \
    --lr 0.1 \
    --milestones "60,100,150" \
    --save_folder $save_folder \
    --flag  "v3"\
    --check \
#    --resume "work_space/RFB_landms/rfb1.0_face_320_320_wider_face_add_lm_10_10_anchor_v4_20210329174602/model/best_model_rfb_160_loss6.6319.pth"


