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
import demo
from models import nets
from models.layers.functions.prior_box import PriorBox
from utils import box_utils
from utils.nms.py_cpu_nms import py_cpu_nms
from utils import image_processing, debug, file_processing, torch_tools

print(torch.cuda.device_count())


def get_parser():
    input_size = [320, 320]
    image_dir = "data/test_image"
    model_path = "/home/dm/data3/FaceDetector/Face-Detector-1MB-with-landmark/work_space/RFB_landms_v2/RFB_landm1.0_face_320_320_wider_face_add_lm_10_10_no_RandomAffineResizePadding2_20210615104418/model/best_model_RFB_landm_198_loss7.4634.pth"
    # model_path ="work_space/rfb_ldmks_face_320_320.pth"
    net_type = "rfb_landm"
    priors_type = "face"
    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default=priors_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.3, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=input_size)
    parser.add_argument('--num_classes', help="num_classes", type=int, default=2)
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
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
                 top_k=5000,
                 keep_top_k=750,
                 device="cpu"):
        """
        :param model_path:
        :param net_type:"RFB",
        :param input_size:input_size,
        :param network: Backbone network mobile0.25 or slim or RFB
        :param prob_threshold: confidence_threshold
        :param iou_threshold: nms_threshold
        :param top_k:
        :param keep_top_k:
        :param device:
        """
        self.device = device
        self.net_type = net_type
        self.priors_type = priors_type
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.input_size = input_size
        self.net, self.prior_boxes = self.build_net(self.net_type, self.priors_type)
        self.priors_cfg = self.prior_boxes.get_prior_cfg()
        self.priors = self.prior_boxes.priors.to(self.device)
        self.net = self.load_model(self.net, model_path)
        print('Finished loading model!')

    @debug.run_time_decorator("post_process")
    def pose_process(self, output, image_size):
        """
        :param loc:
        :param conf:
        :param width: orig image width
        :param height:orig image height
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return:
        """
        loc, conf, landms = output
        if self.priors is None:
            priorbox = PriorBox(self.input_size, self.priors_type)
            self.priors = priorbox.priors.to(self.device)
        bboxes_scale = np.asarray(image_size * 2)
        landms_scale = np.asarray(image_size * 5)
        # get boxes
        boxes = box_utils.decode(loc.data.squeeze(0), self.priors,
                                 [self.prior_boxes.center_variance, self.prior_boxes.size_variance])
        boxes = boxes.cpu().numpy()
        conf = conf.squeeze(0).data.cpu().numpy()
        boxes = boxes * bboxes_scale
        # get landmarks
        # landms = box_utils.decode_landm(landms.data.squeeze(0), self.priorbox, variance)
        landms = box_utils.decode_landm(landms.data.squeeze(0), self.priors,
                                        [self.prior_boxes.center_variance, self.prior_boxes.size_variance])
        landms = landms.squeeze(0)
        landms = landms.cpu().numpy()
        landms = landms * landms_scale

        picked_box_probs = []
        picked_landms = []
        picked_labels = []
        num_classes = conf.shape[1]
        for class_index in range(1, num_classes):
            sub_probs = conf[:, class_index]
            if sub_probs.shape[0] == 0:
                continue
            dets, landm = self.nms_process(boxes,
                                           sub_probs,
                                           landms,
                                           prob_threshold=self.prob_threshold,
                                           iou_threshold=self.iou_threshold,
                                           top_k=self.top_k,
                                           keep_top_k=self.keep_top_k)
            picked_box_probs.append(dets)
            picked_landms.append(landm)
            picked_labels.extend([class_index] * dets.shape[0])
        if len(picked_box_probs) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([])
        dets = np.concatenate(picked_box_probs)
        landms = np.concatenate(picked_landms)
        labels = np.asarray(picked_labels)
        return dets, labels, landms

    @staticmethod
    @debug.run_time_decorator("nms_process")
    def nms_process(boxes, scores, landms, prob_threshold, iou_threshold, top_k, keep_top_k):
        """
        :param boxes: (num_boxes, 4)
        :param scores:(num_boxes,)
        :param landms:(num_boxes, 10)
        :param prob_threshold:
        :param iou_threshold:
        :param top_k:
        :param keep_top_k:
        :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
                 landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
        """
        # ignore low scores
        inds = np.where(scores > prob_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, iou_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        return dets, landms

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
    top_k = args.top_k
    keep_top_k = args.keep_top_k
    image_dir = args.image_dir
    device = args.device
    input_size = args.input_size

    det = Detector(model_path,
                   net_type=net_type,
                   priors_type=priors_type,
                   prob_threshold=prob_threshold,
                   iou_threshold=iou_threshold,
                   top_k=top_k,
                   keep_top_k=keep_top_k,
                   input_size=input_size,
                   device=device)
    det.detect_image_dir(image_dir, isshow=True)
