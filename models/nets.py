# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-03-12 10:05:31
"""
import os
from models.backbone.retinaface_landm import RetinaFaceLandm
from models.backbone.net_slim import SlimLandm
from models.backbone.net_rfb import RFBLandm,RFB


def build_net(net_type:str, prior_cfg, width_mult=1.0, phase='train'):
    if net_type.lower() == "mobile0.25_landm".lower():
        # prior_cfg = mnet_face_config
        net = RetinaFaceLandm(cfg=prior_cfg, phase=phase)
    elif net_type.lower() == "slim_landm".lower():
        # prior_cfg = slim_face_config
        net = SlimLandm(cfg=prior_cfg, phase=phase)
    elif net_type.lower() == "RFB_landm".lower():
        # prior_cfg = rfb_face_config
        net = RFBLandm(cfg=prior_cfg, phase=phase)
    elif net_type.lower() == "RFB".lower():
        # prior_cfg = rfb_face_config
        net = RFB(cfg=prior_cfg, phase=phase)
    else:
        raise Exception("Error:{}".format(net_type))
    return net
