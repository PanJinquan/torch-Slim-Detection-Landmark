# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Slim-Detection-Landmark
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
import os, sys

sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
sys.path.append("../..")
sys.path.append(os.getcwd())
import argparse
import torch
import cv2
import numpy as np
import MNN
import demo
from utils import image_processing, debug


def get_parser():
    image_dir = "data/person"
    # model_path = "./data/pretrained/mnn/rbf_600_600.mnn"
    model_path = "./data/pretrained/mnn/rbf_300_300.mnn"
    net_type = "RFB"
    parser = argparse.ArgumentParser(description='Face Detection Test')
    parser.add_argument('-m', '--model_path', default=model_path, type=str, help='model file path')
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='confidence_threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--input_size', nargs='+', help="--input size [600(W),600(H)]", type=int, default=[300, 300])
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    args = parser.parse_args()
    print(args)
    return args


class MNNDetector(demo.Detector):
    def __init__(self,
                 model_path,
                 net_type="RFB",
                 input_size=[300, 300],
                 prob_threshold=0.6,
                 iou_threshold=0.4,
                 top_k=5000,
                 keep_top_k=750,
                 device="cuda:0"):
        """
        :param model_path:
        :param net_type:"RFB",
        :param input_size:input_size,
        :param network: Backbone network mobile0.25 or slim or RFB
        :param prob_threshold: confidence_threshold
        :param nms_threshold: nms_threshold
        :param top_k:
        :param keep_top_k:
        :param device:
        """
        super(MNNDetector, self).__init__(model_path,
                                          net_type,
                                          input_size,
                                          prob_threshold,
                                          iou_threshold,
                                          top_k,
                                          keep_top_k,
                                          device=device)

    def build_net(self, cfg, net_type, model_path):
        """
        :param net_type: <dict> mobile0.25,slim or RFB
        :return:net,cfg,prior_data
        """
        interpreter = MNN.Interpreter(model_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        net = {
            "interpreter": interpreter,
            "session": session,
            "input_tensor": input_tensor,
        }
        return net

    @debug.run_time_decorator("inference")
    def inference(self, img_tensor):
        img_tensor = np.asarray(img_tensor, dtype=np.float32)
        session = self.net["session"]
        interpreter = self.net["interpreter"]
        input_tensor = self.net["input_tensor"]
        mnn_tensor = MNN.Tensor((1, 3, self.input_size[1], self.input_size[0]),
                                MNN.Halide_Type_Float, img_tensor,
                                MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(mnn_tensor)
        interpreter.runSession(session)
        conf = interpreter.getSessionOutput(session, "scores").getData()  # <class 'tuple'>: (1, 4420, 2)
        loc = interpreter.getSessionOutput(session, "boxes").getData()  # <class 'tuple'>: (1, 4420, 4)
        landms = interpreter.getSessionOutput(session, "ldmks").getData()  # <class 'tuple'>: (1, 4420, 10)
        conf = torch.from_numpy(conf).to(device)
        loc = torch.from_numpy(loc).to(device)
        landms = torch.from_numpy(landms).to(device)
        return loc, conf, landms


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

    det = MNNDetector(model_path,
                      net_type=net_type,
                      prob_threshold=prob_threshold,
                      iou_threshold=iou_threshold,
                      top_k=top_k,
                      keep_top_k=keep_top_k,
                      input_size=input_size,
                      device=device)
    det.detect_image_dir(image_dir, isshow=True)
