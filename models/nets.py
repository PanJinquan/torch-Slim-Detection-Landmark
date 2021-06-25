# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-03-12 10:05:31
"""
from models.backbone.nn.net_retinaface_landm import RetinaFaceLandm
from models.backbone.nn.net_slim import SlimLandm
from models.backbone.nn.net_rfb import RFBLandm, RFB
from models.backbone.ssd.model_slim_ssd import create_model_slim_ssd
from models.backbone.ssd.model_rfb_ssd import create_model_rfb_ssd
from models.backbone.ssd.model_rfb_landms_ssd import create_model_rfb_landm_ssd
from models.backbone.ssd.model_mb_v2_ssd_ssd import create_mobilenetv2_ssd
from models.backbone.ssd.model_mb_v2_landms_ssd import create_mobilenetv2_landm_ssd


def build_net_v1(net_type: str, prior_boxes, width_mult=1.0, phase='train', device="cuda:0"):
    if net_type.lower() == "mnet_landm".lower():
        # prior_cfg = mnet_face_config
        net = RetinaFaceLandm(prior_boxes=prior_boxes, phase=phase)
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
        create_net = create_model_slim_ssd
    elif net_type.lower() == 'RFB'.lower():
        create_net = create_model_rfb_ssd
    elif net_type.lower() == 'mbv2'.lower():
        create_net = create_mobilenetv2_ssd
    elif net_type.lower() == 'RFB_landm'.lower():
        create_net = create_model_rfb_landm_ssd
    elif net_type.lower() == 'MBV2_Landm'.lower():
        create_net = create_mobilenetv2_landm_ssd
    else:
        raise Exception("The net type is wrong.")
    is_test = not phase == 'train'
    net = create_net(prior_boxes=prior_boxes,
                     num_classes=prior_boxes.num_classes,
                     is_test=is_test,
                     device=device,
                     width_mult=width_mult)
    return net
