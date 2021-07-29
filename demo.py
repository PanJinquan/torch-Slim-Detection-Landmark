# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Slim-Detection-Landmark
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
from models import nets
from models.anchor_utils.prior_box import PriorBox
from models.anchor_utils import anchor_utils
from models.anchor_utils.nms import py_cpu_nms
from utils import image_processing, file_processing, torch_tools


def get_parser():
    input_size = [320, 320]
    image_dir = "/home/dm/data3/dataset/card_datasets/card_test/card"
    model_path = "work_space/card/RFB1.0_card_320_320_CardData4det_20210701114842/model/best_model_RFB_199_loss0.4026.pth"
    net_type = "rfb"
    priors_type = "card"

    image_dir = "data/test_image"
    model_path = "work_space/RFB_face_person/RFB1.0_face_person_320_320_MPII_v2_ssd_20210624100518/model/best_model_RFB_168_loss2.8330.pth"
    net_type = "rfb"
    priors_type = "face_person"

    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default=priors_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=input_size)
    parser.add_argument('--num_classes', help="num_classes", type=int, default=2)
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    args = parser.parse_args()
    print(args)
    return args


class Detector(object):
    def __init__(self,
                 model_path,
                 net_type="RFB",
                 priors_type="face",
                 input_size=[320, 320],
                 prob_threshold=0.6,
                 iou_threshold=0.4,
                 freeze_header=True,
                 device="cuda:0"):
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
        self.model, self.prior_boxes = self.build_net(self.net_type, self.priors_type)
        self.class_names = self.prior_boxes.class_names
        self.priors_cfg = self.prior_boxes.get_prior_cfg()
        self.priors = self.prior_boxes.priors.to(self.device)
        self.model = self.load_model(self.model, model_path)
        print('Finished loading model!')

    def build_net(self, net_type, priors_type, version="v2"):
        priorbox = PriorBox(input_size=self.input_size, priors_type=priors_type, freeze_header=self.freeze_header)
        if version.lower() == "v1".lower():
            model = nets.build_net_v1(net_type, priorbox, width_mult=1.0, phase='test', device=self.device)
        else:
            model = nets.build_net_v2(net_type, priorbox, width_mult=1.0, phase='test', device=self.device)
        model = model.to(self.device)
        return model, priorbox

    def load_model(self, model, model_path):
        """
        :param model:
        :param model_path:
        :param load_to_cpu:
        :return:
        """
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    # @debug.run_time_decorator("pre_process")
    def pre_process(self, image, input_size, mean=(127.0, 127.0, 127.0), std=(128.0, 128.0, 128.0)):
        """
        :param image:
        :param input_size: model input size [W,H]
        :param mean:
        :return:image_tensor: out image tensor[1,channels,W,H]
                input_size  : model new input size [W,H]
        """
        out_image = image_processing.resize_image(image, resize_height=input_size[1], resize_width=input_size[0])
        out_image = np.float32(out_image)
        out_image -= mean
        out_image /= std
        out_image = out_image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(out_image).unsqueeze(0)
        return image_tensor

    def pose_process(self, output, image_size):
        """
        bboxes, scores = output
        """
        bboxes, scores = output
        bboxes_scale = np.asarray(image_size * 2)
        # get boxes
        if not self.prior_boxes.freeze_header:
            variances = [self.prior_boxes.center_variance, self.prior_boxes.size_variance]
            bboxes = anchor_utils.decode(bboxes, self.priors, variances)
        bboxes = bboxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        bboxes = bboxes * bboxes_scale
        scores = scores[:, 1:]  # scores[:, 0:]是背景，无需nms
        dets, labels = py_cpu_nms.bboxes_nms(bboxes, scores,
                                             prob_threshold=self.prob_threshold,
                                             iou_threshold=self.iou_threshold,
                                             top_k=self.top_k,
                                             keep_top_k=self.keep_top_k)
        labels = labels + 1  # index+1
        return dets, labels

    # @debug.run_time_decorator("inference")
    def inference(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            # loc, conf, landms-> boxes,scores,landms
            output = self.model(input_tensor)
        return output

    # @debug.run_time_decorator("predict")
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
        dets, labels = self.pose_process(output, image_size=[shape[1], shape[0]])
        if isshow:
            self.show_image(rgb_image, dets, labels)
        return dets, labels

    def detect_image_dir(self, image_dir, isshow=True):
        """
        :param image_dir: directory or image file path
        :param isshow:<bool>
        :return:
        """
        image_list = file_processing.get_files_lists(image_dir)
        for img_path in image_list:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = image_processing.resize_image(image, 800)
            self.predict(image, isshow=isshow)

    def show_image(self, image, dets, labels, landms=None):
        """
        :param image
        :param dets
        :return:
        """
        if not landms is None and len(landms) > 0:
            landms = landms.reshape(len(landms), -1, 2)
            image = image_processing.draw_landmark(image, landms, vis_id=False)
        print("dets:{}".format(dets))
        print("landms:{}".format(landms))
        if len(dets) > 0:
            bboxes = dets[:, 0:4]
            scores = dets[:, 4:5]
            image = image_processing.draw_image_detection_bboxes(image, bboxes, scores, labels)
        image_processing.cv_show_image("image", image)


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
