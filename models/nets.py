# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-03-12 10:05:31
"""
from models.backbone.nn.net_retinaface_landm import RetinaFaceLandm
from models.backbone.nn.net_slim import SlimLandm
from models.backbone.nn.net_rfb import RFBLandm, RFB
from models.backbone.ssd.mb_tiny_fd import create_mb_tiny_slim_fd
from models.backbone.ssd.mb_tiny_RFB_fd import create_mb_tiny_rfb_fd
from models.backbone.ssd.mb_tiny_RFB_landms import create_mb_tiny_rfb_landms
from models.backbone.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from models.backbone.ssd.mobilenet_v2_ssd_landms import create_mobilenetv2_ssd_landms


def build_net_v1(net_type: str, prior_boxes, width_mult=1.0, phase='train', device="cuda:0"):
    if net_type.lower() == "mobile0.25_landm".lower():
        # prior_cfg = mnet_face_config
        net = RetinaFaceLandm(cfg=prior_boxes.prior_cfg, phase=phase)
    elif net_type.lower() == "slim_landm".lower():
        net = SlimLandm(prior_boxes=prior_boxes, phase=phase)
    elif net_type.lower() == "RFB_landm".lower():
        # prior_cfg = rfb_face_config
        net = RFBLandm(prior_boxes=prior_boxes, phase=phase)
    elif net_type.lower() == "RFB".lower():
        # prior_cfg = rfb_face_config
        net = RFB(prior_boxes=prior_boxes, phase=phase)
    else:
        raise Exception("Error:{}".format(net_type))
    return net


def build_net_v2(net_type: str, prior_boxes, width_mult=1.0, phase='train', device="cuda:0"):
    if net_type.lower() == 'slim'.lower():
        create_net = create_mb_tiny_slim_fd
    elif net_type.lower() == 'RFB'.lower():
        create_net = create_mb_tiny_rfb_fd
    elif net_type.lower() == 'mbv2'.lower():
        create_net = create_mobilenetv2_ssd_lite
    elif net_type.lower() == 'RFB_landm'.lower():
        create_net = create_mb_tiny_rfb_landms
    elif net_type.lower() == 'MBV2_Landm'.lower():
        create_net = create_mobilenetv2_ssd_landms
    else:
        create_net = None
        raise Exception("The net type is wrong.")
    is_test = not phase == 'train'
    net = create_net(prior_boxes=prior_boxes,
                     num_classes=prior_boxes.num_classes,
                     is_test=is_test,
                     device=device,
                     width_mult=width_mult)
    return net
