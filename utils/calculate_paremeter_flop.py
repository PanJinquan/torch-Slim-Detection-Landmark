from __future__ import print_function
import argparse
import torch
from models.dataloader import cfg_mnet, cfg_slim, cfg_rfb
from thop import profile
from thop import clever_format
from models import RetinaFace
from models import Slim
from models import RFB

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')

args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)
    long_side = int(args.long_side)
    short_side = int(args.long_side/4*3)
    img = torch.randn(1, 3, long_side, short_side)
    flops, params = profile(net, inputs=(img, ))
    
    flops, params = clever_format([flops, params], "%.3f")
    print("param:", params, "flops:", flops)





