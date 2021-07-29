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
import numpy as np
import demo
from models.anchor_utils import anchor_utils
from models.anchor_utils.nms import py_cpu_nms
from utils import debug


def get_parser():
    input_size = [320, 320]
    image_dir = "data/test_image"
    model_path = "/home/dm/data3/FaceDetector/torch-Slim-Detection-Landmark/work_space/RFB_landms_v2/RFB_landm1.0_face_320_320_wider_face_add_lm_10_10_dmai_data_FDDB_v2_ssd_20210624145405/model/best_model_RFB_landm_183_loss7.6508.pth"
    net_type = "rfb_landm"
    priors_type = "face"

    # image_dir = "data/test_image"
    # model_path = "/home/dm/data3/FaceDetector/torch-Slim-Detection-Landmark/data/weights/v0.0/mobilenet0.25_Final.pth"
    # net_type = "mnet_landm"
    # priors_type = "mnet_face"
    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default=priors_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.3, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=input_size)
    parser.add_argument('--num_classes', help="num_classes", type=int, default=2)
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    args = parser.parse_args()
    print(args)
    return args


class Detector(demo.Detector):
    def __init__(self,
                 model_path,
                 net_type="RFB",
                 priors_type="face",
                 input_size=[320, 320],
                 prob_threshold=0.6,
                 iou_threshold=0.4,
                 freeze_header=False,
                 device="cpu"
                 ):
        super(Detector, self).__init__(model_path,
                                       net_type=net_type,
                                       priors_type=priors_type,
                                       input_size=input_size,
                                       prob_threshold=prob_threshold,
                                       iou_threshold=iou_threshold,
                                       freeze_header=freeze_header,
                                       device=device)

    def build_net(self, net_type, priors_type, version="v2"):
        return super().build_net(net_type, priors_type, version)

    def pose_process(self, output, image_size):
        """
        bboxes, conf, landms = output
        bboxes = torch.Size([1, num_anchors, 4])
        bboxes = torch.Size([1, num_anchors, 2])
        bboxes = torch.Size([1, num_anchors, 10])
        """
        bboxes, scores, landms = output
        bboxes_scale = np.asarray(image_size * 2)
        landms_scale = np.asarray(image_size * 5)
        if not self.prior_boxes.freeze_header:
            # get boxes
            variances = [self.prior_boxes.center_variance, self.prior_boxes.size_variance]
            bboxes = anchor_utils.decode(bboxes, self.priors, variances)
            # get landmarks
            landms = anchor_utils.decode_landm(landms, self.priors, variances)

        bboxes = bboxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        landms = landms[0].cpu().numpy()
        bboxes = bboxes * bboxes_scale
        landms = landms * landms_scale
        scores = scores[:, 1:]  # scores[:, 0:]是背景，无需nms
        dets, labels, landms = py_cpu_nms.bboxes_landm_nms(bboxes, scores, landms,
                                                           prob_threshold=self.prob_threshold,
                                                           iou_threshold=self.iou_threshold,
                                                           top_k=self.top_k,
                                                           keep_top_k=self.keep_top_k)
        labels = labels + 1  # index+1
        return dets, labels, landms

    @debug.run_time_decorator("predict")
    def predict(self, rgb_image, isshow=False):
        """
        :param rgb_image:
        :return:
        bboxes: <np.ndarray>: (num_boxes, 4)
        scores: <np.ndarray>: (num_boxes, 1)
        scores: <np.ndarray>: (num_boxes, 5, 2)
        """
        shape = rgb_image.shape
        input_tensor = self.pre_process(rgb_image, input_size=self.input_size)
        output = self.inference(input_tensor)
        dets, labels, landms = self.pose_process(output, image_size=[shape[1], shape[0]])
        if isshow:
            self.show_image(rgb_image, dets, labels, landms)
        return dets, labels, landms


if __name__ == '__main__':
    args = get_parser()
    model_path = args.model_path
    net_type = args.net_type
    priors_type = args.priors_type
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    image_dir = args.image_dir
    device = args.device
    input_size = args.input_size

    det = Detector(model_path,
                   net_type=net_type,
                   priors_type=priors_type,
                   prob_threshold=prob_threshold,
                   iou_threshold=iou_threshold,
                   input_size=input_size,
                   device=device)
    det.detect_image_dir(image_dir, isshow=True)
