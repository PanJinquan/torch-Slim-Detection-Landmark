import os
import argparse
import torch.utils.data as data
import train
from models.transforms.data_transforms import TrainLandmsTransform, TestLandmsTransform
from models.dataloader import detection_collate
from models.backbone.multibox_loss import MultiBoxLandmLoss


def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    # train_path = '/home/dm/panjinquan3/widerface/train/train.txt'
    # val_path = '/home/dm/panjinquan3/widerface/train/val.txt'
    # data_type = "WiderFace"
    net_type = "rfb_landm"
    # train_path1 = "/home/dm/panjinquan3/MPII/trainval.txt"
    # val_path = "/home/dm/panjinquan3/MPII/test.txt"
    # train_path = "/home/dm/panjinquan3/wider_face_add_lm_10_10/trainval.txt"
    # val_path = "/home/dm/panjinquan3/wider_face_add_lm_10_10/test.txt"
    train_path = "/home/dm/data3/dataset/face_person/wider_face_add_lm_10_10/test.txt"
    val_path = "/home/dm/data3/dataset/face_person/wider_face_add_lm_10_10/test.txt"
    data_type = "VOCLandm"
    num_workers = 4
    batch_size = 64
    train_path = [train_path]
    val_path = [val_path]
    parser.add_argument('--train_path', nargs='+', help='Dataset directory path', default=train_path)
    parser.add_argument('--val_path', nargs='+', help='val dataset directory', default=val_path)
    parser.add_argument('--net_type', default=net_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--data_type', default=data_type, help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--priors_type', default="face", type=str, help='priors type:face or person')
    parser.add_argument('--input_size', nargs='+', help="define network input size", type=int, default=[320, 320])
    parser.add_argument('--num_workers', default=num_workers, type=int, help='Number of workers used in dataloading')
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


class Trainer(train.Trainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.criterion = MultiBoxLandmLoss(self.num_classes, 0.35, True, 0, True, 7, 0.35, False, self.device)

    def load_trainval_dataset(self, train_path, val_path, data_type="VOC", check=True):
        self.logging.info("===" * 15)
        # train_transform = preproc(self.input_size, self.rgb_mean, self.rgb_std)
        # test_transform = val_preproc(self.input_size, self.rgb_mean, self.rgb_std)
        train_transform = TrainLandmsTransform(self.input_size, self.rgb_mean, self.rgb_std)
        test_transform = TestLandmsTransform(self.input_size, self.rgb_mean, self.rgb_std)
        train_dataset = self.load_dataset(train_path, data_type, train_transform, phase="train", check=True)
        train_loader = data.DataLoader(train_dataset, self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, collate_fn=detection_collate)
        self.logging.info("---" * 10)
        val_dataset = self.load_dataset(val_path, data_type, test_transform, phase="test", check=True)
        val_loader = data.DataLoader(val_dataset, self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, collate_fn=detection_collate)
        self.logging.info("===" * 15)
        return train_loader, val_loader

    def train_epoch(self, epoch, epoch_size):
        self.net.train()
        sum_loss = 0.0
        sum_loss_l = 0.0
        sum_loss_c = 0.0
        sum_loss_landm = 0.0
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
            loss_l, loss_c, loss_landm = self.criterion(out, self.priors, targets)
            loss_l = self.priors_cfg['loc_weight'] * loss_l
            loss_landm = self.priors_cfg['landm_weight'] * loss_landm
            loss = loss_l + loss_c + loss_landm
            loss.backward()
            self.optimizer.step()

            sum_loss += loss.item()
            sum_loss_l += loss_l.item()
            sum_loss_c += loss_c.item()
            sum_loss_landm += loss_landm.item()
            if step % self.log_freq == 0 or step == 1:
                num = min(self.log_freq, step)
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = sum_loss / num
                avg_loss_l = sum_loss_l / num
                avg_loss_c = sum_loss_c / num
                avg_loss_landm = sum_loss_landm / num
                self.logging.info('Epoch:{:0=3}/{:0=3}\t Step: {}/{}\t '
                                  'Total Loss: {:.4f}\t Loc: {:.4f}\t Cla: {:.4f}\t Landm: {:.4f}\t '
                                  'LR: {:.6f}'.format(epoch, self.num_epochs, step, epoch_size,
                                                      avg_loss, avg_loss_l, avg_loss_c, avg_loss_landm, lr))
                self.writer.add_scalars(main_tag="Train-loss",
                                        tag_scalar_dict={"total_loss": avg_loss,
                                                         "Cla_loss": avg_loss_c,
                                                         "Loc_loss": avg_loss_l,
                                                         "Lam_loss": avg_loss_landm,
                                                         },
                                        global_step=epoch * epoch_size + step)
                sum_loss = 0.0
                sum_loss_l = 0.0
                sum_loss_c = 0.0
                sum_loss_landm = 0.0

    def val_epoch(self, epoch):
        self.logging.info("val_epoch...")
        self.logging.info("work_dir:{}".format(self.save_folder))
        self.net.eval()
        avg_loss = 0.0
        avg_loss_l = 0.0
        avg_loss_c = 0.0
        avg_loss_landm = 0.0
        num = 0
        for images, targets in self.val_loader:
            # self.show_image(images[0, :], targets[0], transpose=True, normal=True)
            # load train data
            num += 1
            images = images.to(self.device)
            targets = [anno.to(self.device) for anno in targets]
            # forward
            out = self.net(images)
            loss_l, loss_c, loss_landm = self.criterion(out, self.priors, targets)
            loss_l = self.priors_cfg['loc_weight'] * loss_l
            loss_landm = self.priors_cfg['landm_weight'] * loss_landm
            loss = loss_l + loss_c + loss_landm

            avg_loss += loss.item()
            avg_loss_l += loss_l.item()
            avg_loss_c += loss_c.item()
            avg_loss_landm += loss_landm.item()
        avg_loss = avg_loss / num
        avg_loss_l = avg_loss_l / num
        avg_loss_c = avg_loss_c / num
        avg_loss_landm = avg_loss_landm / num
        lr = self.optimizer.param_groups[0]['lr']
        self.logging.info(
            'validation Epoch:{:0=3}/{:0=3}\t Total Loss: {:.4f}\t Loc: {:.4f}\t Cla: {:.4f}\t Landm: {:.4f}\tLR: {:.6f}'.
                format(epoch, self.num_epochs, avg_loss, avg_loss_l, avg_loss_c, avg_loss_landm, lr))

        self.writer.add_scalars(main_tag="Val-loss",
                                tag_scalar_dict={"total_loss": avg_loss,
                                                 "Cla_loss": avg_loss_c,
                                                 "Loc_loss": avg_loss_l,
                                                 "Lam_loss": avg_loss_landm,
                                                 },
                                global_step=epoch)
        self.writer.add_scalar("lr", lr, epoch)
        return avg_loss


if __name__ == '__main__':
    args = get_parser()
    t = Trainer(args)
    t.start_train()
