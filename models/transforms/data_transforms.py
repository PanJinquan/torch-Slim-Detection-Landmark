# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# url     : https://blog.csdn.net/u010397980/article/details/88088025
# --------------------------------------------------------
"""

import cv2
import numpy as np
import torch
from models.transforms import augment_bbox
from models.transforms import augment_bbox_landm


class ToTensor(object):
    def __call__(self, img, boxes=None, labels=None, **kwargs):
        """
        :param img:
        :param boxes:
        :param labels:
        :return:
        """
        # [H,W,C]->[C,H,W]: (360, 480, 3)->(3, 360, 480)
        # img = img.astype(np.float32)
        # boxes = boxes.astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        if kwargs:
            return img, boxes, labels, kwargs
        return img, boxes, labels


class TrainTransform:
    """
    ref: https://blog.csdn.net/u010397980/article/details/88088025
    """

    def __init__(self, size, mean=0.0, std=1.0, norm=False):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        # bg_dir="/home/dm/data3/dataset/finger_keypoint/finger/val/images"
        bg_dir = "/data3/panjinquan/dataset/finger_keypoint/finger/val/images"
        self.mean = mean
        self.size = size
        self.transform = augment_bbox.Compose([
            # augment_bbox.RandomRot90(),  # 随机横屏和竖屏
            # augment_bbox.RandomRotation(degrees=15),
            augment_bbox.ProtectBoxes(norm=False),
            augment_bbox.RandomHorizontalFlip(),
            # augment_bbox.RandomBoxesPaste(bg_dir=bg_dir),
            # augment_bbox.RandomVerticalFlip(),
            augment_bbox.RandomCrop(),
            # augment_bbox.RandomCropLarge(min_size=self.size),
            # augment_bbox.RandomContrastBrightness(),
            # augment_bbox.ResizePadding(self.size),
            # augment_bbox.ResizeRandomPadding(self.size, p=1.0),
            augment_bbox.Resize(size),
            augment_bbox.RandomColorJitter(),
            augment_bbox.SwapChannels(),
            augment_bbox.NormalizeBoxesCoords(),
            augment_bbox.ProtectBoxes(norm=True),
            augment_bbox.Normalize(mean=mean, std=std, norm=norm),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.transform(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0, norm=False):
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        self.transform = augment_bbox.Compose([
            augment_bbox.Resize(size),
            # augment_bbox.ResizePadding(size),
            augment_bbox.NormalizeBoxesCoords(),
            augment_bbox.Normalize(mean=mean, std=std, norm=norm),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class TrainLandmsTransform:
    def __init__(self, size, mean=0, std=1.0, flip_index=[], norm=False):
        """
        :param size: 输出size大小(W,H)
        :param mean: Normalize mean
        :param std:  Normalize std
        :param flip_index: 翻转后，对于的关键点也进行翻转，如果输入的关键点没有左右关系的，
                           请设置flip_index，以便翻转时，保证index的关系
                           如，对于人脸关键点：flip_index=[2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
        """
        if not flip_index:
            flip_index = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        self.mean = mean
        self.size = size
        self.augment = augment_bbox_landm.Compose([
            # augment_bbox_landm.ProtectBoxes(),
            # augment_bbox.RandomRot90(),  # 随机横屏和竖屏
            augment_bbox_landm.RandomRotation(degrees=15),
            augment_bbox_landm.ProtectBoxes(norm=False),
            augment_bbox_landm.RandomHorizontalFlip(flip_index=flip_index),
            # augment_bbox_landm.RandomBoxesPaste(bg_dir=bg_dir),
            # augment_bbox_landm.RandomVerticalFlip(),
            # augment_bbox_landm.RandomCrop(),
            augment_bbox_landm.RandomCropLarge(min_size=self.size),
            # augment_bbox_landm.RandomContrastBrightness(),
            # augment_bbox_landm.ResizePadding(self.size),
            # augment_bbox_landm.ResizeRandomPadding(self.size, p=1.0),
            # augment_bbox_landm.RandomAffineResizePadding(output_size=self.size, degrees=15),
            augment_bbox_landm.Resize(size),
            # augment_bbox_landm.ResizePadding(size),
            augment_bbox_landm.RandomColorJitter(),
            # augment_bbox_landm.SwapChannels(),
            augment_bbox_landm.NormalizeBoxesCoords(),
            augment_bbox_landm.ProtectBoxes(norm=True),
            augment_bbox_landm.Normalize(mean=mean, std=std, norm=norm),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, **kwargs):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels, **kwargs)


class TestLandmsTransform:
    def __init__(self, size, mean=0.0, std=1.0, norm=False):
        """
        :param size: 输出size大小(W,H)
        :param mean: Normalize mean
        :param std:  Normalize std
        """
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        self.augment = augment_bbox_landm.Compose([
            augment_bbox_landm.Resize(size),
            # augment_bbox_landm.ResizePadding(size),
            # augment_bbox_landm.RandomAffineResizePadding(output_size=size, degrees=0),
            augment_bbox_landm.NormalizeBoxesCoords(),
            augment_bbox_landm.Normalize(mean=mean, std=std, norm=norm),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, **kwargs):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels, **kwargs)


class DemoTransform:
    def __init__(self, size, mean=0.0, std=1.0, norm=False):
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        bg_dir = "/home/dm/data3/dataset/finger_keypoint/finger/val/images"
        self.transform = augment_bbox.Compose([
            # augment_bbox.RandomRot90(),  # 随机横屏和竖屏
            # augment_bbox.RandomRotation(degrees=15),
            augment_bbox.ProtectBoxes(norm=False),
            augment_bbox.RandomHorizontalFlip(),
            # augment_bbox.RandomBoxesPaste(bg_dir=bg_dir),
            augment_bbox.RandomMosaic(size, p=0.5),
            # augment_bbox.RandomVerticalFlip(),
            augment_bbox.RandomCrop(),
            # augment_bbox.RandomCropLarge(min_size=self.size),
            # augment_bbox.RandomContrastBrightness(),
            # augment_bbox.ResizePadding(self.size),
            # augment_bbox.ResizeRandomPadding(self.size, p=1.0),
            augment_bbox.Resize(size),
            augment_bbox.RandomColorJitter(),
            augment_bbox.SwapChannels(),
            augment_bbox.NormalizeBoxesCoords(),
            augment_bbox.ProtectBoxes(norm=True),
            augment_bbox.Normalize(mean=mean, std=std, norm=norm),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        if not (isinstance(size, list) or isinstance(size, tuple)):
            size = (size, size)
        self.transform = augment_bbox.Compose([
            augment_bbox.Resize(size),
            augment_bbox.Normalize(mean=mean, std=std, norm=False),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


def torch_transforms(input_size, RGB_MEAN, RGB_STD):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])
    return transform


def check_bboxes(boxes):
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        assert xmax > xmin
        assert ymax > ymin


def demo_for_landmark():
    from utils import image_processing
    image_path = "test.jpg"
    src_boxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    land_mark = [[[122.44442, 54.193676],
                  [147.6293, 56.77364],
                  [135.35794, 74.66961],
                  [120.94379, 83.858765],
                  [143.35617, 86.417175]],
                 [[258.14902, 287.81662],
                  [281.83157, 281.46664],
                  [268.39877, 306.3493],
                  [265.5242, 318.80936],
                  [286.5602, 313.99652]]]
    classes = [1, 1]
    input_size = [400, 400]
    src_image = image_processing.read_image(image_path)
    src_boxes = np.asarray(src_boxes, dtype=np.float32)
    src_classes = np.asarray(classes, dtype=np.int32)
    src_land_mark = np.asarray(land_mark).reshape(-1, 10)
    augment = TrainLandmsTransform(input_size, mean=0, std=255)
    # augment = TestLandmsTransform(input_size)
    for i in range(1000):
        # dst_image, boxes, classes, kwargs = augment(src_image, src_boxes.copy(), classes)
        dst_image, boxes, classes, kwargs = augment(src_image.copy(),
                                                    src_boxes.copy(),
                                                    src_classes.copy(),
                                                    land_mark=src_land_mark.copy())
        land_mark = kwargs["land_mark"]
        augment_bbox_landm.show_landmark_image(dst_image, boxes, land_mark, classes, normal=True, transpose=True)


def demo_for_bboxes():
    from utils import image_processing
    image_path = "test.jpg"
    src_boxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    classes = [1, 1]
    input_size = [400, 400]
    src_image = image_processing.read_image(image_path)
    src_boxes = np.asarray(src_boxes, dtype=np.float32)
    src_classes = np.asarray(classes, dtype=np.int32)
    augment = TrainTransform(input_size)
    for i in range(1000):
        # dst_image, boxes, classes, kwargs = augment(src_image, src_boxes.copy(), classes)
        dst_image, boxes, classes = augment(src_image.copy(),
                                            src_boxes.copy(),
                                            src_classes.copy())
        augment_bbox.show_image(dst_image, boxes, classes, normal=True, transpose=True)


if __name__ == "__main__":
    """
                if kwargs:
                for k in kwargs.keys():
                    scale = [self.size[0] / width, self.size[1] / height] * 5
                    kwargs[k] = kwargs[k] * scale
    """
    from utils import image_processing

    demo_for_bboxes()
    # demo_for_landmark()
