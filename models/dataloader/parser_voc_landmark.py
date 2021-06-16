# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import xmltodict
import numpy as np
import cv2
import glob
import random
from tqdm import tqdm
from utils import file_processing
from models.dataloader.parser_voc import VOCDataset, ConcatDataset


class VOCLandmarkDataset(VOCDataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_names=None,
                 transform=None,
                 target_transform=None,
                 color_space="RGB",
                 keep_difficult=False,
                 shuffle=False,
                 check=False):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param color_space:
        :param keep_difficult:
        :param shuffle:
        """
        super(VOCLandmarkDataset, self).__init__(filename=filename,
                                                 data_root=data_root,
                                                 anno_dir=anno_dir,
                                                 image_dir=image_dir,
                                                 class_names=class_names,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 color_space=color_space,
                                                 keep_difficult=keep_difficult,
                                                 shuffle=shuffle,
                                                 check=check)

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        dst_ids = []
        # image_id = image_id[:100]
        # image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            bboxes, labels, landms, is_difficult = self.get_annotation(annotation_file)
            if not self.keep_difficult:
                bboxes = bboxes[is_difficult == 0]
                # labels = labels[is_difficult == 0]
            if ignore_empty and (len(bboxes) == 0 or len(labels) == 0):
                print("illegal annotation:{}".format(annotation_file))
                continue
            dst_ids.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def fileter_landms(self, boxes, labels, landms):
        for i in range(len(boxes)):
            landm = landms[i, :]
            # if landm=-1,label=-1
            if landm[0] < 0:
                landms[i, :] = 0.0
                # labels[i] = -1.0
                # labels[i] = 0.0
                if labels[i] == 1:
                    labels[i] = -1
        # keypoints[:, 0::2] /= width
        # keypoints[:, 1::2] /= height
        return boxes, labels, landms

    def convert_target(self, boxes, labels, landms):
        annotations = []
        for i in range(len(boxes)):
            bbox = boxes[i, :].tolist()
            label = labels[i].tolist()
            landm = landms[i, :].tolist()
            anno = list()
            anno += bbox
            anno += landm
            # if landm=-1,label=-1
            if landm[0] < 0:
                label = -1
            anno += [label]
            assert len(anno) == 15
            annotations.append(anno)
        target = np.array(annotations)
        return target

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.get_image_anno_file(image_id)
        # bboxes, labels, is_difficult = self.get_annotation(annotation_file)
        bboxes, labels, landms, is_difficult = self.get_annotation(annotation_file)
        image = self.read_image(image_file, color_space=self.color_space)
        if self.transform:
            # rgb_image, bboxes, labels = self.transform(rgb_image, bboxes, labels)
            bboxes, labels, landms = self.fileter_landms(bboxes, labels, landms)
            image, bboxes, labels, landms = self.transform(image, bboxes, labels, landms=landms)
            landms = landms["landms"]
        # show_landmark_image(image, bboxes, landms, labels, normal=True, transpose=True)
        num_boxes = len(bboxes)
        if self.target_transform and num_boxes > 0:
            bboxes, labels, landms = self.target_transform(bboxes, labels, landms)
        labels = labels.reshape(-1)
        target = self.convert_target(bboxes, labels, landms)
        if num_boxes == 0:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        # return image, bboxes, labels, landms
        return image, target

    def get_annotation(self, xml_file):
        """
        :param xml_file:
        :param class_vertical_formula: class_vertical_formula = {class_name: i for i, class_name in enumerate(class_names)}
        :return:
        """
        try:
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])
            filename = annotation["filename"]
            objects = annotation["object"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            objects = []
        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            name = object["name"]
            if self.class_names and name not in self.class_names:
                continue
            difficult = int(object["difficult"])
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            # rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            if not self.check_bbox(width, height, bbox):
                print("illegal annotation:{}".format(xml_file))
                continue

            # get person keypoints ,if exist
            if 'lm' in object:
                lm = object["lm"]
                landms = [lm["x1"], lm["y1"], lm["x2"], lm["y2"], lm["x3"],
                          lm["y3"], lm["x4"], lm["y4"], lm["x5"], lm["y5"]]
            else:
                landms = [-1] * 5 * 2
            item = {}
            item["bbox"] = bbox
            item["keypoints"] = landms
            item["difficult"] = difficult
            if self.class_dict:
                name = self.class_dict[name]
            item["name"] = name
            objects_list.append(item)
        boxes, labels, landms, is_difficult = self.get_objects_items(objects_list)
        return boxes, labels, landms, is_difficult

    def get_objects_items(self, objects_list):
        """
        :param objects_list:
        :return:
        """
        bboxes = []
        labels = []
        landms = []
        is_difficult = []
        for item in objects_list:
            bboxes.append(item["bbox"])
            labels.append(item['name'])
            landms.append(item['keypoints'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        # labels = np.array(labels, dtype=np.int64)
        labels = np.asarray(labels).reshape(-1, 1)
        landms = np.array(landms, dtype=np.float32)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, landms, is_difficult


def show_landmark_image(image, bboxes, landms, labels, normal=False, transpose=False):
    """
    :param image:
    :param targets_t:
                bboxes = targets[idx][:, :4].data
                keypoints = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from utils import image_processing
    image = np.asarray(image)
    bboxes = np.asarray(bboxes)
    landms = np.asarray(landms)
    labels = np.asarray(labels)
    print("===" * 10)
    print("image:{}".format(image.shape))
    print("bboxes:{}".format(bboxes))
    print("landms:{}".format(landms))
    print("labels:{}".format(labels))
    nums = len(landms)
    if transpose:
        image = image_processing.untranspose(image)
    h, w, _ = image.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        bboxes = bboxes * bboxes_scale
        landms = landms * landms_scale
    landms = landms.reshape(nums, -1, 2)
    # tmp_image = image_processing.untranspose(tmp_image)
    tmp_image = image_processing.convert_color_space(image, colorSpace="BGR")
    tmp_image = image_processing.draw_landmark(tmp_image, landms, vis_id=True)
    image_processing.show_image_boxes("train", tmp_image, bboxes)


def show_targets_image(image, targets, normal=False, transpose=False):
    """
    :param image:
    :param targets:
                bboxes = targets[idx][:, :4].data
                landms = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)
    bboxes = targets[:, :4]
    landms = targets[:, 4:14]
    labels = targets[:, -1]
    show_landmark_image(image, bboxes, landms, labels, transpose=transpose, normal=normal)


if __name__ == "__main__":
    from models.transforms.data_transforms import TrainLandmsTransform, TestLandmsTransform
    from models.dataloader import WiderFaceDetection, detection_collate, preproc, val_preproc
    # from models.dataloader import voc_parser, collate_fun
    import torch.utils.data as data

    image_mean = np.array([0, 0, 0]),
    image_std = 255.0
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    input_size = [320, 320]
    target_transform = None
    check = False
    rgb_mean = [0, 0, 0]
    rgb_std = [255, 255, 255]
    batch_size = 1
    shuffle = False
    # filename = "/home/dm/data3/dataset/face_person/dmai_data/trainval.txt"
    filename = "/home/dm/data3/dataset/face_person/FDDB/trainval.txt"
    # filename = "/home/dm/data3/dataset/face_person/wider_face_add_lm_10_10/test.txt"
    class_names = {'BACKGROUND': 0, 'face': 1}
    # transform = val_preproc(input_size, rgb_mean, rgb_std)
    # transform = preproc(input_size, rgb_mean, rgb_std)
    transform = TrainLandmsTransform(input_size, rgb_mean, rgb_std)
    # transform = TestLandmsTransform(input_size, rgb_mean, rgb_std)
    train_dataset = VOCLandmarkDataset(filename,
                                       data_root=None,
                                       class_names=class_names,
                                       transform=transform,
                                       keep_difficult=False,
                                       check=False)
    # train_dataset = ConcatDataset([train_dataset, train_dataset])
    train_dataset = data.ConcatDataset([train_dataset])
    train_loader = data.DataLoader(train_dataset,
                                   batch_size,
                                   num_workers=0,
                                   shuffle=True,
                                   collate_fn=detection_collate)
    for i, inputs in enumerate(train_loader):
        print(i)
        img, target = inputs
        show_targets_image(img[0], target[0], normal=True, transpose=True)
        # show_landmark_image(img, target, normal=True, transpose=True)
