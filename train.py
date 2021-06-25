# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Slim-Detection-Landmark
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.utils.data as data
from models.dataloader.parser_voc_landmark import VOCLandmarkDataset
from models.dataloader.parser_voc import VOCDataset
# from models.dataloader.parser_voc_landmark_preproc import VOCLandmarkDataset
from models.transforms.data_transforms import TrainTransform
from models.dataloader import WiderFaceDetection, detection_collate
from models.backbone.multibox_loss import MultiBoxLoss
from models.anchor_utils.prior_box import PriorBox
from models import nets
from utils import file_processing, debug, torch_tools
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    # net_type = "rfb_face_person"

    # train_path = '/home/dm/panjinquan3/widerface/train/train.txt'
    # val_path = '/home/dm/panjinquan3/widerface/train/val.txt'
    # data_type = "WiderFace"
    # net_type = "rfb"
    net_type = "RFB"

    # train_path1 = "/home/dm/panjinquan3/MPII/trainval.txt"
    # val_path = "/home/dm/panjinquan3/MPII/test.txt"
    # train_path = "/home/dm/panjinquan3/wider_face_add_lm_10_10/trainval.txt"
    # val_path = "/home/dm/panjinquan3/wider_face_add_lm_10_10/test.txt"
    train_path = "/home/dm/data3/dataset/face_person/MPII/test.txt"
    val_path = "/home/dm/data3/dataset/face_person/MPII/test.txt"
    data_type = "VOC"
    priors_type = "mnet_face_config"
    batch_size = 8
    train_path = [train_path]
    val_path = [val_path]
    parser.add_argument('--train_path', nargs='+', help='Dataset directory path', default=train_path)
    parser.add_argument('--val_path', nargs='+', help='val dataset directory', default=val_path)
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--data_type', default=data_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default=priors_type, type=str, help='priors type:face or person')
    parser.add_argument('--input_size', nargs='+', help="define network input size", type=int, default=[320, 320])
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--max_epoch', default=150, type=int, help='gpu_id')
    parser.add_argument('--gpu_id', default="0", type=str, help='gpu_id')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='batch_size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume', default="", type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='work_space/test_model', help='Location to save checkpoint models')
    parser.add_argument('--start_save', default=None, type=int, help='number of epochs')
    parser.add_argument('--log_freq', default=20, type=int, help='print log freq')
    parser.add_argument('--check', action='store_true', help='check dataset', default=False)
    parser.add_argument('--optimizer_type', default="SGD", type=str, help='optimizer_type')
    parser.add_argument('--last_epoch', default=-1, type=int, help='last_epoch')
    parser.add_argument('--milestones', default="60,100", type=str, help="milestones for MultiStepLR")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")
    parser.add_argument("--flag", default="", type=str, help='flag')
    parser.add_argument('--width_mult', default=1.0, type=float, help='width_mult')
    parser.add_argument('--polyaxon', action='store_true', help='flag', default=False)

    args = parser.parse_args()
    # args.polyaxon=True
    if args.polyaxon:
        from utils import rsync_data

        print("use polyaxon")
        args.train_path = rsync_data.get_polyaxon_datasets(root="ceph", dir_list=args.train_path)
        args.val_path = rsync_data.get_polyaxon_datasets(root="ceph", dir_list=args.val_path)
        args.checkpoint_folder = os.path.join(rsync_data.get_polyaxon_output(), args.checkpoint_folder)
    return args


class Trainer(object):
    def __init__(self, args):
        torch_tools.set_env_random_seed()
        self.num_workers = args.num_workers
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.initial_lr = args.lr
        self.gamma = args.gamma
        self.start_save = args.start_save
        self.save_folder = args.save_folder
        self.last_epoch = args.last_epoch
        self.width_mult = args.width_mult
        self.net_type = args.net_type
        self.priors_type = args.priors_type
        self.log_freq = args.log_freq
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.data_type = args.data_type
        self.num_epochs = args.max_epoch
        self.flag = args.flag
        self.resume = args.resume
        self.optimizer_type = args.optimizer_type
        self.scheduler = args.scheduler
        self.milestones = args.milestones

        dataset_name = [os.path.basename(os.path.dirname(path)) for path in args.train_path]
        flag = [self.net_type + str(self.width_mult), self.priors_type, self.input_size[0], self.input_size[1],
                "_".join(dataset_name), str(self.flag), str(file_processing.get_time())]
        flag = [str(f) for f in flag if f]

        self.save_folder = os.path.join(args.save_folder, "_".join(flag))
        self.model_dir = os.path.join(self.save_folder, "model")
        self.log_dir = os.path.join(self.save_folder, "log")
        self.logfile = os.path.join(self.log_dir, "log.txt")
        self.writer = SummaryWriter(self.log_dir)
        file_processing.create_dir(self.model_dir)
        file_processing.create_file_path(self.logfile)
        self.logfile = os.path.join(self.save_folder, "log.txt")
        self.logging = debug.set_logger(logfile=self.logfile)
        self.logging.info("{}".format(args))
        self.logging.info("save_folder:{}".format(self.save_folder))

        self.gpu_id = [int(v.strip()) for v in args.gpu_id.split(",")]
        self.device = torch.device("cuda:{}".format(self.gpu_id[0]))
        self.net, self.prior_boxes = self.build_net(self.net_type, self.priors_type)
        self.priors_cfg = self.prior_boxes.get_prior_cfg()
        self.priors = self.prior_boxes.priors.to(self.device)
        # self.class_names = self.prior_cfg.class_names
        # self.num_classes = self.prior_cfg.num_classes
        self.rgb_mean = self.prior_boxes.image_mean  # bgr order
        self.rgb_std = self.prior_boxes.image_std  # bgr order
        self.class_names = self.prior_boxes.class_names
        self.num_classes = self.prior_boxes.num_classes

        self.train_loader, self.val_loader = self.load_trainval_dataset(args.train_path,
                                                                        args.val_path,
                                                                        self.data_type,
                                                                        args.check)
        self.optimizer = optim.SGD(self.net.parameters(),
                                   lr=self.initial_lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        self.criterion = MultiBoxLoss(self.num_classes, 0.35, True, 0, True, 7, 0.35, False, self.device)
        self.lr_scheduler = self.get_lr_scheduler(self.scheduler, self.optimizer_type, self.milestones)
        self.logging.info("net_type   :{}".format(self.net_type))
        self.logging.info("priors_type:{}".format(self.priors_type))
        self.logging.info("priors nums:{}".format(len(self.priors)))

    def build_net(self, net_type, priors_type, version="v2"):
        priorbox = PriorBox(input_size=self.input_size, priors_type=priors_type)
        if version.lower() == "v1".lower():
            net = nets.build_net_v1(net_type, priorbox, width_mult=self.width_mult, phase='train', device=self.device)
        else:
            net = nets.build_net_v2(net_type, priorbox, width_mult=self.width_mult, phase='train', device=self.device)
        self.logging.info("build_net:{},version:{}".format(net_type, version))
        if self.resume:
            self.logging.info(f"Resume from the model {self.resume}")
            state_dict = torch_tools.load_state_dict(self.resume, module=False)
            net.load_state_dict(state_dict)
        net = torch.nn.DataParallel(net, device_ids=self.gpu_id)
        net = net.to(self.device)
        return net, priorbox

    def load_dataset(self, files, data_type, transform, phase, check=True):
        datasets = []
        for path in files:
            self.logging.info('Loading {} Data:{}'.format(phase, path))
            if data_type == "WiderFace":
                dataset = WiderFaceDetection(path, transform)
            elif data_type == "VOC":
                dataset = VOCDataset(filename=path,
                                     class_names=self.prior_boxes.class_names,
                                     transform=transform,
                                     check=check,
                                     shuffle=False)
            elif data_type == "VOCLandm":
                dataset = VOCLandmarkDataset(filename=path,
                                             class_names=self.prior_boxes.class_names,
                                             transform=transform,
                                             check=check)
                assert self.num_classes == len(dataset.class_names)
            else:
                raise Exception("Error:{}".format(data_type))
            datasets.append(dataset)
        datasets = data.ConcatDataset(datasets)
        self.logging.info("have {} data: {}".format(phase, len(datasets)))
        return datasets

    def load_trainval_dataset(self, train_path, val_path, data_type="VOC", check=True):
        self.logging.info("===" * 15)
        # train_transform = preproc(self.input_size, self.rgb_mean, self.rgb_std)
        # test_transform = val_preproc(self.input_size, self.rgb_mean, self.rgb_std)
        train_transform = TrainTransform(self.input_size, self.rgb_mean, self.rgb_std)
        test_transform = TrainTransform(self.input_size, self.rgb_mean, self.rgb_std)
        train_dataset = self.load_dataset(train_path, data_type, train_transform, phase="train", check=True)
        train_loader = data.DataLoader(train_dataset, self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, collate_fn=detection_collate)
        self.logging.info("---" * 10)
        val_dataset = self.load_dataset(val_path, data_type, test_transform, phase="test", check=True)
        val_loader = data.DataLoader(val_dataset, self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, collate_fn=detection_collate)
        self.logging.info("===" * 15)
        return train_loader, val_loader

    def get_lr_scheduler(self, scheduler, optimizer_type, milestones):
        lr_scheduler = None
        if optimizer_type != "Adam":
            if scheduler == 'multi-step':
                self.logging.info("Uses MultiStepLR scheduler.")
                milestones = [int(v.strip()) for v in milestones.split(",")]
                lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones,
                                           gamma=0.1, last_epoch=self.last_epoch)
            elif scheduler == 'cosine':
                self.logging.info("Uses CosineAnnealingLR scheduler.")
                t_max = len(self.train_loader) * self.num_epochs
                lr_scheduler = CosineAnnealingLR(self.optimizer, t_max, last_epoch=self.last_epoch)
            elif scheduler == 'poly':
                self.logging.info("Uses PolyLR scheduler.")
            else:
                self.logging.fatal(f"Unsupported Scheduler: {scheduler}.")
        return lr_scheduler

    def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (self.initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = self.initial_lr * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def start_train(self):
        self.min_loss = sys.maxsize
        epoch_size = len(self.train_loader)
        self.logging.info("Start training from epoch {}".format(self.last_epoch + 1))
        self.logging.info("work_dir:{}".format(self.save_folder))
        for epoch in range(self.last_epoch + 1, self.num_epochs):
            if self.optimizer_type != "Adam" and self.scheduler != "poly":
                self.lr_scheduler.step()
            self.train_epoch(epoch, epoch_size)
            val_loss = self.val_epoch(epoch)
            self.save_model(self.net, epoch, val_loss)
            self.logging.info("===" * 10)

    def train_epoch(self, epoch, epoch_size):
        self.net.train()
        sum_loss = 0.0
        sum_loss_l = 0.0
        sum_loss_c = 0.0
        step = 0
        for images, targets in self.train_loader:
            step += 1
            # load train data
            images = images.to(self.device)
            targets = [anno.to(self.device) for anno in targets]
            # forward
            out = self.net(images)
            # backprop
            self.optimizer.zero_grad()
            loss_l, loss_c = self.criterion(out, self.priors, targets)
            loss_l = self.priors_cfg['loc_weight'] * loss_l
            loss = loss_l + loss_c
            loss.backward()
            self.optimizer.step()

            sum_loss += loss.item()
            sum_loss_l += loss_l.item()
            sum_loss_c += loss_c.item()
            if step % self.log_freq == 0 or step == 1:
                num = min(self.log_freq, step)
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = sum_loss / num
                avg_loss_l = sum_loss_l / num
                avg_loss_c = sum_loss_c / num
                self.logging.info('Epoch:{:0=3}/{:0=3}\t Step: {}/{}\t '
                                  'Total Loss: {:.4f}\t Loc: {:.4f}\t Cla: {:.4f}\t '
                                  'LR: {:.6f}'.format(epoch, self.num_epochs, step, epoch_size,
                                                      avg_loss, avg_loss_l, avg_loss_c, lr))
                self.writer.add_scalars(main_tag="Train-loss",
                                        tag_scalar_dict={"total_loss": avg_loss,
                                                         "Cla_loss": avg_loss_c,
                                                         "Loc_loss": avg_loss_l,
                                                         },
                                        global_step=epoch * epoch_size + step)
                sum_loss = 0.0
                sum_loss_l = 0.0
                sum_loss_c = 0.0

    def val_epoch(self, epoch):
        self.logging.info("val_epoch...")
        self.logging.info("work_dir:{}".format(self.save_folder))
        self.net.eval()
        avg_loss = 0.0
        avg_loss_l = 0.0
        avg_loss_c = 0.0
        num = 0
        for images, targets in self.val_loader:
            # self.show_image(images[0, :], targets[0], transpose=True, normal=True)
            # load train data
            num += 1
            images = images.to(self.device)
            targets = [anno.to(self.device) for anno in targets]
            # forward
            out = self.net(images)
            loss_l, loss_c = self.criterion(out, self.priors, targets)
            loss_l = self.priors_cfg['loc_weight'] * loss_l
            loss = loss_l + loss_c

            avg_loss += loss.item()
            avg_loss_l += loss_l.item()
            avg_loss_c += loss_c.item()
        avg_loss = avg_loss / num
        avg_loss_l = avg_loss_l / num
        avg_loss_c = avg_loss_c / num
        lr = self.optimizer.param_groups[0]['lr']
        self.logging.info(
            'validation Epoch:{:0=3}/{:0=3}\t Total Loss: {:.4f}\t Loc: {:.4f}\t Cla: {:.4f}\t LR: {:.6f}'.
                format(epoch, self.num_epochs, avg_loss, avg_loss_l, avg_loss_c, lr))

        self.writer.add_scalars(main_tag="Val-loss",
                                tag_scalar_dict={"total_loss": avg_loss,
                                                 "Cla_loss": avg_loss_c,
                                                 "Loc_loss": avg_loss_l,
                                                 },
                                global_step=epoch)
        self.writer.add_scalar("lr", lr, epoch)
        return avg_loss

    def save_model(self, model, epoch, loss):
        """
        :param model:
        :param out_layer:
        :param epoch:
        :param logs:
        :param start_save:
        :return:
        """
        start_save = self.start_save if self.start_save else self.num_epochs - 10
        if epoch >= start_save:
            model_name = "model_{}_{:03d}_loss{:.4f}.pth".format(self.net_type, epoch, loss)
            model_path = os.path.join(self.model_dir, model_name)
            torch.save(model.module.state_dict(), model_path)
            self.logging.info("save model in:{}".format(model_path))

        if self.min_loss >= loss:
            self.min_loss = loss
            model_name = "best_model_{}_{:03d}_loss{:.4f}.pth".format(self.net_type, epoch, loss)
            best_model_path = os.path.join(self.model_dir, model_name)
            file_processing.remove_prefix_files(self.model_dir, "best_model_*")
            torch.save(model.module.state_dict(), best_model_path)
            self.logging.info("save best_model_path in:{}".format(best_model_path))


if __name__ == '__main__':
    args = get_parser()
    t = Trainer(args)
    t.start_train()
