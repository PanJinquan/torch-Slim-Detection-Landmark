# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.anchor_utils import box_utils, anchor_utils


class SSDLandmark(nn.Module):
    def __init__(self,
                 num_classes: int,
                 backbone: nn.ModuleList,
                 extra_layers: nn.ModuleList,
                 feature_index: List[int],
                 class_headers: nn.ModuleList,
                 bbox_headers: nn.ModuleList,
                 landm_headers: nn.ModuleList,
                 is_test=False,
                 prior_boxes=None,
                 device=None):
        """Compose a SSDLandmark model using the given components."""
        super(SSDLandmark, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.feature_index = feature_index
        self.extra_layer = extra_layers
        self.class_headers = class_headers
        self.bbox_headers = bbox_headers
        self.landm_headers = landm_headers
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
        # print("location-view:{}".format(location.shape))

        landm = self.landm_headers[i](x)
        # print("x={},location:{}".format(x.shape, location.shape))
        landm = landm.permute(0, 2, 3, 1).contiguous()
        landm = landm.view(landm.size(0), -1, 10)
        # print("location-view:{}".format(location.shape))
        return conf, loc, landm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        landmarks = []
        header_index = 0
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_index:
                conf, loc, landm = self.compute_header(header_index, x)
                header_index += 1
                confidences.append(conf)
                locations.append(loc)
                landmarks.append(landm)
        for layer in self.extra_layer:
            x = layer(x)
            conf, loc, landm = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(conf)
            locations.append(loc)
            landmarks.append(landm)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        landmarks = torch.cat(landmarks, 1)
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            if self.freeze_header:
                locations = anchor_utils.decode(locations.data.squeeze(0), self.priors,
                                                [self.center_variance, self.size_variance])
                landmarks = anchor_utils.decode_landm(landmarks.data.squeeze(0), self.priors,
                                                      [self.center_variance, self.size_variance])
        return locations, confidences, landmarks


class MatchPriorLandms(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        """
        :param center_form_priors: priors [cx,cy,w,h]
        :param center_variance:
        :param size_variance:
        :param iou_threshold:
        """
        self.center_form_priors = center_form_priors  # [cx,cy,w,h]
        # [cx,cy,w,h]->[xmin,ymin,xmax,ymax]
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels, gt_landms):
        """
        if landms=-1,preproc will set landms=0, labels=-1
        :param gt_boxes:
        :param gt_labels:
        :param gt_landms:
        :return:
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        if type(gt_landms) is np.ndarray:
            gt_landms = torch.from_numpy(gt_landms)
        boxes, labels, landms = box_utils.assign_priors_landms(gt_boxes,
                                                               gt_labels,
                                                               gt_landms,
                                                               self.corner_form_priors,
                                                               self.iou_threshold)
        # [xmin,ymin,xmax,ymax]-> [cx,cy,w,h]
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes,
                                                         self.center_form_priors,
                                                         self.center_variance,
                                                         self.size_variance)
        landms = box_utils.encode_landm(landms,
                                        self.center_form_priors,
                                        variances=[self.center_variance, self.size_variance])
        return locations, labels, landms
