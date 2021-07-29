# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys

sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
sys.path.append("../..")
sys.path.append(os.getcwd())

import argparse
import torch
import numpy as np
import demo
import demo_for_landm
from models import onnx_detector
from models.anchor_utils.prior_box import PriorBox
from utils import debug


def get_parser():
    input_size = [320, 320]
    # image_dir = "data/test_image"
    # # model_path = "data/pretrained/onnx/rfb_landm_face_320_320.onnx"
    # model_path = "data/pretrained/onnx/rfb_landm_face_320_320_freeze.onnx"
    # net_type = "rfb_landm"
    # priors_type = "face"

    # image_dir = "/home/dm/data3/dataset/card_datasets/card_test/card"
    image_dir = "/home/dm/project/SDK/object-detection-tnn-sdk/data/test_image/card/1599129060517.png"
    model_path = "data/pretrained/onnx/rfb_card_320_320_freeze.onnx"
    # model_path = "data/pretrained/onnx/rfb_card_320_320.onnx"
    net_type = "rfb"
    priors_type = "card"


    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default=priors_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.3, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=input_size)
    parser.add_argument('--num_classes', help="num_classes", type=int, default=2)
    parser.add_argument('--device', default="cpu", type=str, help='device')
    args = parser.parse_args()
    return args


class Detector(demo.Detector):
# class Detector(demo_for_landm.Detector):
    def __init__(self,
                 model_path,
                 net_type="RFB",
                 priors_type="face",
                 input_size=[320, 320],
                 prob_threshold=0.6,
                 iou_threshold=0.4,
                 freeze_header=True,
                 device="cpu"):
        """
        :param model_path:
        :param net_type:"RFB",
        :param input_size:input_size,
        :param network: Backbone network mobile0.25 or slim or RFB
        :param prob_threshold: confidence_threshold
        :param iou_threshold: nms_threshold
        :param device:
        """
        self.device = device
        self.net_type = net_type
        self.priors_type = priors_type
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.top_k = 5000
        self.keep_top_k = 750
        self.input_size = input_size
        self.freeze_header = freeze_header
        self.model, self.prior_boxes = self.build_net(self.net_type, self.priors_type, model_path)
        self.class_names = self.prior_boxes.class_names
        self.priors_cfg = self.prior_boxes.get_prior_cfg()
        self.priors = self.prior_boxes.priors.to(self.device)
        print('Finished loading model!')

    def build_net(self, net_type, priors_type, model_path):
        priorbox = PriorBox(input_size=self.input_size, priors_type=priors_type, freeze_header=self.freeze_header)
        model = onnx_detector.ONNXModel(onnx_path=model_path)
        return model, priorbox

    @debug.run_time_decorator("detect-inference")
    def inference(self, img_tensor):
        img_tensor = np.asarray(img_tensor)
        output = self.model(img_tensor)  # boxes,scores,landms
        return output

    def pose_process(self, output, image_size):
        output = [torch.from_numpy(o) for o in output]
        return super().pose_process(output, image_size)


if __name__ == '__main__':
    args = get_parser()
    model_path = args.model_path
    net_type = args.net_type
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    image_dir = args.image_dir
    priors_type = args.priors_type
    device = args.device
    input_size = args.input_size
    num_classes = args.num_classes

    det = Detector(model_path,
                   net_type=net_type,
                   priors_type=priors_type,
                   prob_threshold=prob_threshold,
                   iou_threshold=iou_threshold,
                   input_size=input_size,
                   device=device)
    det.detect_image_dir(image_dir, isshow=True)
