# -*- coding:utf-8 -*-

import os
import sys

sys.path.append(os.getcwd())
import cv2
import numpy as np
import demo
from tqdm import tqdm


class Validation(demo.Detector):
    def __init__(self,
                 model_path,
                 net_type,
                 input_size,
                 class_names,
                 priors_type,
                 prob_threshold=0.5,
                 iou_threshold=0.01,
                 device="cuda:0"):
        '''

        :param model_path:
        :param basenet:
        :param class_names:
        :param gt_dir: ground_truth dir lable_file.txt is
                label1 x y w h\n
                lable2 x y w h
        '''
        self.model_path = model_path
        self.net_type = net_type
        self.input_size = input_size
        self.priors_type = priors_type
        self.class_names = class_names
        self.candidate_size = 500
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.device = device
        super(Validation, self).__init__(model_path,
                                         self.net_type,
                                         self.input_size,
                                         self.class_names,
                                         self.priors_type,
                                         prob_threshold=self.prob_threshold,
                                         iou_threshold=self.iou_threshold,
                                         device=self.device)

    def detect_image(self, rgb_image, isshow=True):
        """
        :param rgb_image:  input RGB Image
        :param isshow:
        :return:
        """
        boxes, labels, probs = super().detect_image(rgb_image, isshow)

        return boxes, labels, probs


def fddb_dataset_evaluation(fddb_dir, fddb_result_dir=None):
    """
    :param fddb_dir:
    :param fddb_result_dir:
    :return:
    """
    fddb_img_dir = os.path.join(fddb_dir, 'images')
    fddb_fold_dir = os.path.join(fddb_dir, 'FDDB-folds')
    if not fddb_result_dir:
        fddb_result_dir = os.path.join(fddb_dir, 'face_person')
    fddb_result_img_dir = os.path.join(fddb_result_dir, 'images')
    if not os.path.exists(fddb_result_img_dir):
        os.makedirs(fddb_result_img_dir)
    counter = 0
    num_fddb_foldellipseList = 10
    for i in range(num_fddb_foldellipseList):
        txt_in = os.path.join(fddb_fold_dir, 'FDDB-fold-%02d.txt' % (i + 1))
        txt_out = os.path.join(fddb_result_dir, 'fold-%02d-out.txt' % (i + 1))
        answer_in = os.path.join(fddb_fold_dir, 'FDDB-fold-%02d-ellipseList.txt' % (i + 1))
        print("processing:{}".format(answer_in))
        with open(txt_in, 'r') as fr:
            lines = fr.readlines()
        fout = open(txt_out, 'w')
        ain = open(answer_in, 'r')
        for line in tqdm(lines):
            line = line.strip()
            img_file = os.path.join(fddb_img_dir, line + '.jpg')
            if not os.path.exists(img_file):
                print("no path:{}".format(img_file))
                continue
            out_file = os.path.join(fddb_result_img_dir, line.replace('/', '_') + '.jpg')
            counter += 1
            orig_image = cv2.imread(img_file, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            bboxes, labels, scores = det.detect_image(rgb_image, isshow=False)
            bboxes_scores = np.hstack((bboxes, scores[:, np.newaxis]))
            fout.write('%s\n' % line)
            fout.write('%d\n' % len(bboxes_scores))
            for bbox in bboxes_scores:
                x1, y1, x2, y2, score = bbox
                w, h = x2 - x1, y2 - y1
                fout.write('%d %d %d %d %lf\n' % (x1, y1, w, h, score))
            ain.readline()
            n = int(ain.readline().strip())
            for i in range(n):
                line = ain.readline().strip()
                line_data = [float(_) for _ in line.split(' ')[:5]]
                major_axis_radius, minor_axis_radius, angle, center_x, center_y = line_data
                angle = angle / 3.1415926 * 180.
                center_x, center_y = int(center_x), int(center_y)
                major_axis_radius, minor_axis_radius = int(
                    major_axis_radius), int(minor_axis_radius)
                cv2.ellipse(orig_image, (center_x, center_y),
                            (major_axis_radius, minor_axis_radius),
                            angle, 0, 360, (255, 0, 0), 2)
            for bbox in bboxes_scores:
                x1, y1, x2, y2, score = bbox
                cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imwrite(out_file, orig_image)
        fout.close()
        ain.close()


if __name__ == '__main__':
    fddb_dir = '/home/dm/panjinquan3/dataset/FDDB/SRC'
    model_path = "/home/dm/panjinquan3/modes/face/RFB1.0_face_640_640_MPII_VOC2012_VOC2007_COCO_wider_face_add_lm_10_10_20200807140630/model/rfb-face-model-640-640.pth"
    # model_path = "pretrained/RFB_person_960_540_MPII_VOC_20200628135227/model/RFB-Epoch-196-Loss-1.9992448091506958.pth"
    # class_names = ["BACKGROUND", "fingernail"]
    # class_names = ["BACKGROUND", "person"]
    # priors_type = "person"
    priors_type = "face"
    # priors_type = "face_person"
    net_type = "RFB"
    # net_type = "mbv2"
    # class_names = ["BACKGROUND", "face", "person"]
    class_names = ["BACKGROUND", "face"]
    input_size = [640, 640]
    dataroot = "/home/dm/panjinquan3/dataset/FDDB"
    image_dir = os.path.join(dataroot, "JPEGImages")
    filename = os.path.join(dataroot, "test.txt")
    save_dir = os.path.join(dataroot, "eval")
    label_dir = os.path.join(dataroot, "labels")
    device = "cuda:0"
    det = Validation(model_path,
                     net_type=net_type,
                     input_size=input_size,
                     class_names=class_names,
                     priors_type=priors_type,
                     iou_threshold=0.5,
                     prob_threshold=0.05,
                     device=device)
    flag = [net_type, priors_type, "{}_{}".format(input_size[0], input_size[1])]
    fddb_result_dir = os.path.join(fddb_dir, "_".join(flag))
    fddb_dataset_evaluation(fddb_dir, fddb_result_dir)
