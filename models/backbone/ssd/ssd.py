# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.anchor_utils import box_utils, anchor_utils


class SSD(nn.Module):
    def __init__(self,
                 num_classes: int,
                 backbone: nn.ModuleList,
                 extra_layers: nn.ModuleList,
                 feature_index: List[int],
                 class_headers: nn.ModuleList,
                 bbox_headers: nn.ModuleList,
                 is_test=False,
                 prior_boxes=None,
                 device=None):
        """Compose a SSDLandmark model using the given components."""
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.feature_index = feature_index
        self.extra_layer = extra_layers
        self.class_headers = class_headers
        self.bbox_headers = bbox_headers
        self.is_test = is_test
        self.device = device
        if is_test:
            self.center_variance = prior_boxes.center_variance
            self.size_variance = prior_boxes.size_variance
            self.priors = prior_boxes.priors.to(self.device)
            self.freeze_header = prior_boxes.freeze_header

    def compute_header(self, i, x):
        """
        input:320
        x=torch.Size([1, 64, 40, 40]),location:torch.Size([1, 12, 40, 40])
        x=torch.Size([1, 128, 20, 20]),location:torch.Size([1, 8, 20, 20])
        x=torch.Size([1, 256, 10, 10]),location:torch.Size([1, 8, 10, 10])
        x=torch.Size([1, 256, 5, 5]),location:torch.Size([1, 12, 5, 5])
        :param i:
        :param x:
        :return:
        """
        conf = self.class_headers[i](x)
        conf = conf.permute(0, 2, 3, 1).contiguous()
        conf = conf.view(conf.size(0), -1, self.num_classes)

        loc = self.bbox_headers[i](x)
        # print("x={},location:{}".format(x.shape, location.shape))
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(loc.size(0), -1, 4)
        return conf, loc

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        header_index = 0
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_index:
                conf, loc = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(conf)
                locations.append(loc)
        for layer in self.extra_layer:
            x = layer(x)
            conf, loc = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(conf)
            locations.append(loc)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            if self.freeze_header:
                locations = anchor_utils.decode(locations.data.squeeze(0), self.priors,
                                                [self.center_variance, self.size_variance])
        return locations, confidences


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors,
                                                self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes,
                                                         self.center_form_priors,
                                                         self.center_variance,
                                                         self.size_variance)
        return locations, labels
