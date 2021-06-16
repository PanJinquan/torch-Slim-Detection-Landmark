# config.py
import numpy as np

mnet_face_config = {
    'name': 'mobilenet0.25',
    'min_sizes': [[10, 20], [32, 64], [128, 256]],
    'shrinkage': [8, 16, 32],
    # 'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'decay1': 190,
    'decay2': 220,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

slim_face_config = {
    'name': 'slim',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'shrinkage': [8, 16, 32, 64],
    # 'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'decay1': 190,
    'decay2': 220,
}

rfb_face_config = {
    'name': 'RFB',
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    "shrinkage": [8, 16, 32, 64],
    "aspect_ratios": [[1.0, 1.0]],
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'landm_weight': 1.0,
    'decay1': 190,
    'decay2': 220,
    "class_names": ['BACKGROUND', 'face'],
}

face_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    "shrinkage": [8, 16, 32, 64],
    "aspect_ratios": [[1.0, 1.0]],
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    'clip': False,
    'loc_weight': 2.0,
    'landm_weight': 1.0,
    'decay1': 190,
    'decay2': 220,
    "class_names": ['BACKGROUND', 'face'],
}

face_person_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    "shrinkage": [8, 16, 32, 64],
    # "aspect_ratios": [[1.0, 1.0]],
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]],
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    'clip': False,
    'loc_weight': 2.0,
    'landm_weight': 1.0,
    'decay1': 190,
    'decay2': 220,
    "class_names": ['BACKGROUND', 'face', 'person'],
}
