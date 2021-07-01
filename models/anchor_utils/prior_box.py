import torch
from itertools import product as product
import numpy as np
from math import ceil
from models.config import config
import math


class PriorBox(object):
    def __init__(self, input_size: list, priors_type="face", freeze_header=True):
        super(PriorBox, self).__init__()
        self.priors_type = priors_type
        if priors_type == "face":
            cfg = config.face_config
        elif priors_type == "face_person":
            cfg = config.face_person_config
        elif priors_type == "rfb_face":
            cfg = config.rfb_face_config
        elif priors_type == "mnet_face":
            cfg = config.mnet_face_config
        elif priors_type == "slim_face":
            cfg = config.slim_face_config
        elif priors_type == "card":
            cfg = config.card_config
        else:
            raise Exception("Error:{}".format(priors_type))

        self.class_names = cfg["class_names"]
        self.image_mean = cfg["image_mean"]
        self.image_std = cfg["image_std"]
        self.iou_threshold = cfg["iou_threshold"]
        self.center_variance = cfg["center_variance"]
        self.size_variance = cfg["size_variance"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.shrinkage = cfg["shrinkage"]
        self.input_size = input_size
        self.min_sizes = cfg['min_sizes']
        self.shrinkage = cfg['shrinkage']
        self.clip = cfg['clip']
        self.prior_cfg = cfg
        # self.priors = self.get_priors()
        self.priors = self.get_priors_v2()
        self.freeze_header = freeze_header
        self.num_classes = self.get_numclass()

    def get_numclass(self):
        """
        :return:
        """
        if isinstance(self.class_names, list):
            class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        elif isinstance(self.class_names, dict):
            class_dict = self.class_names
        else:
            raise Exception("Error:{}".format(self.class_names))
        num_classes = max(list(class_dict.values())) + 1
        return num_classes

    def get_prior_cfg(self):
        return self.prior_cfg

    def get_priors(self):
        self.feature_maps = [[ceil(self.input_size[0] / step), ceil(self.input_size[1] / step)] for step in
                             self.shrinkage]
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.input_size[1]
                    s_ky = min_size / self.input_size[0]
                    dense_cx = [x * self.shrinkage[k] / self.input_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.shrinkage[k] / self.input_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def get_priors_v2(self, ):
        """
        input_size = [width,height]
        16：9:
        960: [960, 540],
        640: [640, 360],
        :param size:
        :param priors_type:
        :return:
        """
        print("input_size:{}".format(self.input_size))
        # 多尺度特征检测:共4层,对应的缩减率[8, 16, 32, 64]
        # shrinkage = [8, 16, 32, 64]
        shrinkage = self.shrinkage
        feature_map_w_h_list = []  # 每个尺度输出特征的大小:[[f0_w,f1_w,f2_w,f3_w],[f0_h,f1_h,f2_h,f3_h]]
        for size in self.input_size:
            s = [math.ceil(size / s) for s in shrinkage]  #
            feature_map_w_h_list.append(s)

        # 计算特征缩减率:shrinkage_list近似等于[shrinkage,shrinkage],要求更精确
        # shrinkage_list = [shrinkage, shrinkage]
        shrinkage_list = []
        for i in range(0, len(self.input_size)):
            item_list = []
            for k in range(0, len(feature_map_w_h_list[i])):
                item_list.append(self.input_size[i] / feature_map_w_h_list[i][k])
            shrinkage_list.append(item_list)

        # create priors anchor
        priors = self.generate_priors(feature_map_w_h_list,
                                      shrinkage_list,
                                      self.input_size,
                                      min_boxes=self.min_sizes,
                                      aspect_ratios=self.aspect_ratios)
        # anchors_t = np.asarray(priors).reshape(-1, 4)
        print("priors nums:{},priors_type:{}".format(len(priors), self.priors_type))
        return priors

    @staticmethod
    def generate_priors(feature_map_list,
                        shrinkage_list,
                        input_size,
                        min_boxes,
                        aspect_ratios,
                        clamp=True) -> torch.Tensor:
        """
        :param feature_map_list: 每个尺度输出特征的大小:[[f0_w,f1_w,f2_w,f3_w],[f0_h,f1_h,f2_h,f3_h]]
        :param shrinkage_list: 特征缩减率
        :param input_size: model input size
        :param min_boxes:  anchor size: [[32], [48], [64, 96], [128, 192, 256]]
        :param aspect_ratios: anchor rate= width:height,aspect_ratios=[[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]]
        :param clamp:是否限制上下限范围[0.0, 1.0]
        :return:
        """
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = input_size[0] / shrinkage_list[0][index]
            scale_h = input_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / input_size[0]
                        h = min_box / input_size[1]
                        # priors.append([x_center, y_center, w, h])
                        for ratio in aspect_ratios:
                            # ratio = math.sqrt(ratio)
                            priors.append([x_center, y_center, w * ratio[0], h * ratio[1]])

        print("priors nums:{}".format(len(priors)))
        priors = torch.tensor(priors)
        if clamp:
            torch.clamp(priors, 0.0, 1.0, out=priors)
        return priors


if __name__ == "__main__":
    from utils import image_processing

    # 17625,29375
    input_size = [320, 320]  # W,H
    priors_type = "card"
    prior_boxes = PriorBox(input_size, priors_type=priors_type)
    image = np.ones(shape=(input_size[1], input_size[0], 3), dtype=np.uint8)
    anchors = prior_boxes.priors.detach().numpy()
    anchors = image_processing.center2bboxes(anchors)
    boxes_list = image_processing.convert_anchor(anchors, height=input_size[1], width=input_size[0])
    for i, b in enumerate(boxes_list):
        image_processing.show_image_boxes("anchors", image, [b], color=(255, i * 20 % 255, i * 20 % 255))
