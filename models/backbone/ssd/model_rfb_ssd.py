# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""

from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from models.backbone.nn.model_rfb import Mb_Tiny_RFB
from models.backbone.ssd.ssd import SSD


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(True),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_model_rfb_ssd(prior_boxes, num_classes, is_test=False, width_mult=1.0, device="cuda:0"):
    """
    create_Mb_Tiny_RFB_fd_predictor
    min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]  # for Face
    x=torch.Size([24, 64, 30, 40]), location:torch.Size([24, 12, 30, 40])location-view:torch.Size([24, 3600, 4])
    x=torch.Size([24, 128, 15, 20]),location:torch.Size([24, 8, 15, 20]) location-view:torch.Size([24, 600, 4])
    x=torch.Size([24, 256, 8, 10]), location:torch.Size([24, 8, 8, 10])  location-view:torch.Size([24, 160, 4])
    x=torch.Size([24, 256, 4, 5]),  location:torch.Size([24, 12, 4, 5])  location-view:torch.Size([24, 60, 4])
    :param num_classes:
    :param is_test:
    :param device:
    :return:
    """
    base_net = Mb_Tiny_RFB(num_classes)
    backbone = base_net.model  # disable dropout layer
    # feature_index = [8, 11, 13]
    feature_index = [7, 10, 12]
    extra_layers = ModuleList([Sequential(Conv2d(in_channels=base_net.base_channel * 16,
                                                 out_channels=base_net.base_channel * 4,
                                                 kernel_size=1),
                                          ReLU(inplace=True),
                                          SeperableConv2d(in_channels=base_net.base_channel * 4,
                                                          out_channels=base_net.base_channel * 16,
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1),
                                          ReLU(inplace=True))])

    boxes_expand = [len(boxes) * (len(prior_boxes.aspect_ratios)) for boxes in prior_boxes.min_sizes]
    bbox_headers = ModuleList([SeperableConv2d(in_channels=base_net.base_channel * 4,
                                               out_channels=boxes_expand[0] * 4,
                                               kernel_size=3,
                                               padding=1),
                               SeperableConv2d(in_channels=base_net.base_channel * 8,
                                               out_channels=boxes_expand[1] * 4,
                                               kernel_size=3,
                                               padding=1),
                               SeperableConv2d(in_channels=base_net.base_channel * 16,
                                               out_channels=boxes_expand[2] * 4,
                                               kernel_size=3,
                                               padding=1),
                               Conv2d(in_channels=base_net.base_channel * 16,
                                      out_channels=boxes_expand[3] * 4,
                                      kernel_size=3,
                                      padding=1)])

    class_headers = ModuleList([SeperableConv2d(in_channels=base_net.base_channel * 4,
                                                out_channels=boxes_expand[0] * num_classes,
                                                kernel_size=3,
                                                padding=1),
                                SeperableConv2d(in_channels=base_net.base_channel * 8,
                                                out_channels=boxes_expand[1] * num_classes,
                                                kernel_size=3,
                                                padding=1),
                                SeperableConv2d(in_channels=base_net.base_channel * 16,
                                                out_channels=boxes_expand[2] * num_classes,
                                                kernel_size=3,
                                                padding=1),
                                Conv2d(in_channels=base_net.base_channel * 16,
                                       out_channels=boxes_expand[3] * num_classes,
                                       kernel_size=3,
                                       padding=1)])
    return SSD(num_classes,
               backbone,
               extra_layers,
               feature_index,
               class_headers,
               bbox_headers,
               is_test=is_test,
               prior_boxes=prior_boxes,
               device=device)
