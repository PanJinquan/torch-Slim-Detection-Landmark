# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: ultra-light-fast-generic-face-detector-1MB
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
from __future__ import print_function
import os, sys

sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
sys.path.append("../..")
sys.path.append(os.getcwd())

import argparse
import torch
import cv2
import numpy as np
import models.config.config as config
import demo
from models import onnx_detector
from models.layers.functions.prior_box import PriorBox
from utils import box_utils
from utils.nms.py_cpu_nms import py_cpu_nms
from utils import image_processing, debug, file_processing


def get_parser():
    # network = "RFB"
    # conf_th = 0.5
    # nms_th = 0.3
    # top_k = 500
    # keep_top_k = 750
    input_size = [320, 320]
    # image_path = "./test.jpg"
    # image_path = "./data/image/14.jpg"
    # image_dir = "data/image/7.jpg"
    # image_dir = "data/test_images"
    image_dir = "data/test_image/0.jpg"
    model_path = "work_space/RFB_landms/rfb1.0_face_320_320_wider_face_add_lm_10_10_dmai_data_FDDB_RandomCropLarge_20210608121819/model/rfb1.0_face_320_320.onnx"
    net_type = "rfb"
    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=input_size)
    parser.add_argument('--num_classes', help="num_classes", type=int, default=2)
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--device', default="cpu", type=str, help='device')
    args = parser.parse_args()
    print(args)
    return args


class Detector(demo.Detector):
    def __init__(self,
                 model_path,
                 net_type="RFB",
                 priors_type="face",
                 input_size=[320, 320],
                 num_classes=2,
                 prob_threshold=0.6,
                 iou_threshold=0.4,
                 top_k=5000,
                 keep_top_k=750,
                 device="cuda:0"):
        self.device = device
        self.net_type = net_type
        self.priors_type = priors_type
        self.num_classes = num_classes
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.input_size = input_size
        self.model_path = model_path
        self.net, self.prior_boxes = self.build_net(self.net_type, self.priors_type)
        self.priors_cfg = self.prior_boxes.get_prior_cfg()
        self.priors = self.prior_boxes.priors
        print('Finished loading model!')

    def build_net(self, net_type, priors_type):
        priorbox = PriorBox(input_size=self.input_size, priors_type=priors_type)
        net = onnx_detector.ONNXModel(onnx_path=self.model_path)
        return net, priorbox

    @debug.run_time_decorator("detect-inference")
    def inference(self, img_tensor):
        img_tensor = np.asarray(img_tensor)
        loc, conf, landms = self.net(img_tensor)  # boxes,scores,landms
        return loc, conf, landms

    def pose_process(self, loc, conf, landms, image_size):
        loc = torch.from_numpy(loc)
        conf = torch.from_numpy(conf)
        landms = torch.from_numpy(landms)
        return super().pose_process(loc, conf, landms, image_size)


if __name__ == '__main__':
    args = get_parser()
    model_path = args.model_path
    net_type = args.net_type
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    top_k = args.top_k
    keep_top_k = args.keep_top_k
    image_dir = args.image_dir
    device = args.device
    input_size = args.input_size
    num_classes = args.num_classes

    det = Detector(model_path,
                   net_type=net_type,
                   num_classes=num_classes,
                   prob_threshold=prob_threshold,
                   iou_threshold=iou_threshold,
                   top_k=top_k,
                   keep_top_k=keep_top_k,
                   input_size=input_size,
                   device=device)
    det.detect_image_dir(image_dir, isshow=True)
