# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-07-22 19:40:49
# --------------------------------------------------------
"""

import cv2
import numpy as np
from numpy import random


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
    augmentations.Compose([
                    transforms.CenterCrop(10),
                    transforms.ToTensor(),
    ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class ConvertFromInts(object):
    def __init__(self):
        pass

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class ProtectBBoxes(object):
    def __init__(self):
        pass

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, _ = image.shape
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        return image, boxes, labels


class Normalize(object):
    """归一化数据"""

    def __init__(self, mean=0.0, std=1.0, norm=True):
        """
        :param mean:
        :param std:
        :param norm: 0-1 Normalize(True)
        """
        self.mean = mean
        self.std = std
        self.norm = norm

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        if self.norm:
            image /= 255.0
        image -= self.mean
        image /= self.std
        # check_bboxes(boxes)
        return image, boxes, labels


class ToAbsoluteCoords(object):
    """将归一化boxes坐标转换为图像坐标"""

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels


class ToPercentCoords(object):
    """将图像坐标转为归一化boxes坐标"""

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height
            boxes = np.clip(boxes, 0, 1)
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels


class Resize(object):
    """resize"""

    def __init__(self, size=[300, 300]):
        self.size = tuple(size)

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        if not boxes is None and len(boxes) > 0:
            scale = [self.size[0] / width, self.size[1] / height] * 2
            boxes = boxes * scale
        image = cv2.resize(image, (self.size[0], self.size[1]))
        return image, boxes, labels


class RandomExpand(object):
    def __init__(self, ratio=1.2, padding=0, p=0.5):
        self.ratio = ratio
        self.p = p
        self.padding = padding

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            height, width, depth = image.shape
            ratio = random.uniform(1, self.ratio)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)
            expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
            expand_image[:, :, :] = self.padding
            expand_image[int(top):int(top + height), int(left):int(left + width)] = image
            image = expand_image
            if len(boxes) > 0:
                boxes[:, :2] += (int(left), int(top))
                boxes[:, 2:] += (int(left), int(top))
        return image, boxes, labels


class RandomCrop(object):
    """ 随机裁剪"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, classes):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            if len(boxes) > 0:
                max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)], axis=-1)
            else:
                max_bbox = [int(w_img * 0.2), int(h_img * 0.2), int(w_img * 0.8), int(h_img * 0.8)]
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            h_img, w_img, _ = img.shape
            if len(boxes) > 0:
                # boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
                # boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin
                boxes = boxes - [crop_xmin, crop_ymin, crop_xmin, crop_ymin]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_img)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_img)
        return img, boxes, classes


class RandomRot90(object):
    """ 随机旋转90度"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, classes):
        '''
        :param img: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        # 顺时针旋转90度
        if random.random() < self.p:
            h, w, _ = img.shape
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)
            if len(boxes) > 0:
                # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
                # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
                # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
                boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
                boxes[:, [0, 2]] = h - boxes[:, [2, 0]]
        return img, boxes, classes


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    随机水平翻转
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, classes):
        """
        :param image:
        :param boxes: xmin,ymin,xmax,ymax
        :param classes:
        :return:
        """
        if random.random() < self.p:
            height, width, _ = image.shape
            image = image[:, ::-1, :]
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, boxes, classes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    随机上下翻转
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, classes):
        """
        :param image:
        :param boxes: xmin,ymin,xmax,ymax
        :param classes:
        :return:
        """
        if random.random() < self.p:
            height, width, _ = image.shape
            image = image[::-1, :, :]
            if len(boxes) > 0:
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        return image, boxes, classes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class RandomSaturation(object):
    def __init__(self, p=0.5, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        self.p = p
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrastBrightness(object):
    def __init__(self, p=0.5, lower=0.5, upper=1.5, delta=32):
        """
        随机调整对比度RandomContrast和亮度RandomBrightness (概率：0.5)
        :param p:
        :param lower:
        :param upper:
        """
        self.p = p
        self.random_contrast = RandomContrast(p=p, lower=lower, upper=upper)
        self.random_brightness = RandomBrightness(p=p, delta=delta)

    def __call__(self, image, boxes=None, labels=None):
        image, boxes, labels = self.random_contrast(image, boxes, labels)
        image, boxes, labels = self.random_brightness(image, boxes, labels)
        return image, boxes, labels


class RandomContrast(object):
    """随机调整对比度和亮度"""

    def __init__(self, p=0.5, lower=0.5, upper=1.5):
        self.p = p
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            dtype = image.dtype
            image = np.asarray(image, dtype=np.float32)
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image = np.clip(image, 0, 255)
            image = np.asarray(image, dtype=dtype)
        return image, boxes, labels


class RandomBrightness(object):
    """随机调整亮度"""

    def __init__(self, p=0.5, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.p = p
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            dtype = image.dtype
            image = np.asarray(image, dtype=np.float32)
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
            image = np.asarray(image, dtype=dtype)
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


def check_bboxes(boxes):
    """
    :param boxes:
    :return:
    """
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        assert xmax > xmin
        assert ymax > ymin
