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
import numbers
import PIL.Image as Image
import copy
import random
from models.transforms import augment_bbox, affine_transform


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

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        for t in self.transforms:
            image, boxes, labels, kwargs = t(image, boxes, labels, **kwargs)
        return image, boxes, labels, kwargs


class ProtectBoxes(augment_bbox.ProtectBoxes):
    def __init__(self, norm=False, pct_th=0.01, iou_th=0.65):
        """
        限制boxes边界范围，防止越界的情况，去除小框，避免出现Nan值
        :param norm: True：输入的boxes是归一化坐标，即boxes/[w,h,w,h]，范围(0,1)
                    False：输入的boxes是像素坐标，即(xmin,ymin,xmax,ymax)
        :param pct_th: 0.01表示如果长宽有一个不足图像的1%，则去除
        :param iou_th: clip会缩小boxes，因此如果clip前后的IOU小于该阈值，box会被丢弃
        """
        # 用于限制boxes边界范围，防止越界的情况
        self.clip = ClipBoxes(norm=norm, iou_th=iou_th)
        # 去除小框，避免出现Nan值
        self.min_boxes = IgnoreMinBoxes(norm=norm, pct_th=pct_th)

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        image, boxes, labels, kwargs = self.clip(image, boxes, labels, **kwargs)
        image, boxes, labels, kwargs = self.min_boxes(image, boxes, labels, **kwargs)
        return image, boxes, labels, kwargs


class ClipBoxes(augment_bbox.ClipBoxes):
    def __init__(self, norm=False, iou_th=0.65):
        """
        用于限制boxes边界范围，防止越界的情况
        :param norm: True：输入的boxes是归一化坐标，即boxes/[w,h,w,h]，范围(0,1)
                    False：输入的boxes是像素坐标，即(xmin,ymin,xmax,ymax)
        :param iou_th: clip会缩小boxes，因此如果clip前后的IOU小于该阈值，box会被丢弃
        """
        self.norm = norm
        self.iou_th = iou_th
        self.index = []

    def check_overlap(self, image, boxes, labels, cboxes, iou_th, **kwargs):
        """
        检测clip前后的boxes的IOU变化，因此如果clip前后的IOU小于该阈值iou_th，box会被丢弃
        :param image:
        :param boxes:clip前的boxes
        :param labels:
        :param cboxes:clip后的boxes
        :param iou_th:
        :return:
        """
        ious = np.zeros(shape=(len(boxes),))
        for i in range(len(boxes)):
            ious[i] = self.cal_iou(boxes[i], cboxes[i])
        # print("ious:{}".format(ious))
        index = ious > iou_th
        boxes = cboxes[index]
        labels = labels[index]
        self.index = index
        if kwargs:
            for k in kwargs.keys():
                kwargs[k] = kwargs[k][index]
        return image, boxes, labels, kwargs

    def check_landm_out_of_boxes(self, box, landm):
        """
        检测landmark是否超出boxes的范围
        :param box:
        :param landm:
        :return:
        """
        xmin, ymin, xmax, ymax = box
        r = True
        landm = landm.reshape(-1, 2)
        for point in landm:
            if point[0] < xmin or point[0] > xmax:
                r = False
                break
            if point[1] < ymin or point[1] > ymax:
                r = False
                break
        return r

    def check_landm(self, image, boxes, labels, **kwargs):
        """检测landmark是否超出boxes的范围"""
        if kwargs:
            keys = list(kwargs.keys())
            index = [self.check_landm_out_of_boxes(boxes[i], kwargs[keys[0]][i]) for i in range(len(boxes))]
            boxes = boxes[index]
            labels = labels[index]
            for k in kwargs.keys():
                kwargs[k] = kwargs[k][index]
        return image, boxes, labels, kwargs

    def clip_kwargs(self, image, boxes, labels, **kwargs):
        """限制boxes边界范围"""
        if self.norm:
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = np.clip(kwargs[k][:, 0::2], 0, 1)
                    kwargs[k][:, 1::2] = np.clip(kwargs[k][:, 1::2], 0, 1)
        else:
            height, width, _ = image.shape
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = np.clip(kwargs[k][:, 0::2], 0, width - 1)
                    kwargs[k][:, 1::2] = np.clip(kwargs[k][:, 1::2], 0, height - 1)
        return image, boxes, labels, kwargs

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if len(boxes) > 0:
            image, cboxes, labels = self.clip(image, boxes, labels)
            image, boxes, labels, kwargs = self.check_overlap(image, boxes, labels, cboxes,
                                                              iou_th=self.iou_th, **kwargs)
            image, boxes, labels, kwargs = self.check_landm(image, boxes, labels, **kwargs)
            image, boxes, labels, kwargs = self.clip_kwargs(image, boxes, labels, **kwargs)
        augment_bbox.check_bboxes(boxes)
        return image, boxes, labels, kwargs


class IgnoreMinBoxes(augment_bbox.IgnoreMinBoxes):
    def __init__(self, norm=True, pct_th=0.01):
        """
        去除小框，避免出现Nan值
        :param norm: True：输入的boxes是归一化坐标，即boxes/[w,h,w,h]，范围(0,1)
                    False：输入的boxes是像素坐标，即(xmin,ymin,xmax,ymax)
        :param pct_th: 0.01表示如果长宽有一个不足图像的1%，则去除
        """
        self.norm = norm
        self.pct_th = pct_th

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if len(boxes) > 0:
            height, width, _ = image.shape
            if not self.norm:
                boxes = boxes / [width, height, width, height]
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            index0 = w > self.pct_th
            index1 = h > self.pct_th
            index = index0 * index1
            boxes = boxes[index, :]
            labels = labels[index]
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k] = kwargs[k][index, :]
            if not self.norm:
                boxes = boxes * [width, height, width, height]
        return image, boxes, labels, kwargs


class IgnoreBadBoxes(augment_bbox.IgnoreBadBoxes):
    """去除不合法的bbox,如小于0，越界的bbox"""

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if len(boxes) > 0:
            height, width, _ = image.shape
            index0 = self.get_mask_index(boxes >= 0)
            index1 = self.get_mask_index(boxes < [width, height, width, height])
            index = index0 * index1
            boxes = boxes[index]
            labels = labels[index]
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k] = kwargs[k][index, :]
        augment_bbox.check_bboxes(boxes)
        return image, boxes, labels, kwargs


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

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        image = image.astype(np.float32)
        if self.norm:
            image /= 255.0
        image -= self.mean
        image /= self.std
        # check_bboxes(boxes)
        return image, boxes, labels, kwargs


class UnNormalizeBoxesCoords(object):
    """将归一化boxes坐标转换为图像坐标"""

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes = boxes * [width, height, width, height]
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = kwargs[k][:, 0::2] * width
                    kwargs[k][:, 1::2] = kwargs[k][:, 1::2] * height
                    kwargs[k] = np.asarray(kwargs[k], np.float32)
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels, kwargs


class NormalizeBoxesCoords(object):
    """将图像坐标转为归一化boxes坐标"""

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes = boxes / [width, height, width, height]
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = kwargs[k][:, 0::2] / width
                    kwargs[k][:, 1::2] = kwargs[k][:, 1::2] / height
                    kwargs[k] = np.asarray(kwargs[k], np.float32)
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels, kwargs


class Resize(object):
    """resize"""

    def __init__(self, size=[300, 300]):
        self.size = tuple(size)

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        height, width, _ = image.shape
        if not boxes is None and len(boxes) > 0:
            scale = [self.size[0] / width, self.size[1] / height] * 2
            boxes = boxes * scale
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = kwargs[k][:, 0::2] * scale[0]
                    kwargs[k][:, 1::2] = kwargs[k][:, 1::2] * scale[1]
        image = cv2.resize(image, (self.size[0], self.size[1]))
        return image, boxes, labels, kwargs


class ResizePadding(object):
    def __init__(self, size=[300, 300]):
        """
        等比例图像resize,保持原始图像内容比，避免失真,短边会0填充
        :param size:
        """
        self.size = tuple(size)

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        height, width, _ = image.shape
        scale = min([self.size[0] / width, self.size[1] / height])
        new_size = [int(width * scale), int(height * scale)]
        pad_w = self.size[0] - new_size[0]
        pad_h = self.size[1] - new_size[1]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        if not boxes is None and len(boxes) > 0:
            boxes = boxes * scale + [left, top, left, top]
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = kwargs[k][:, 0::2] * scale + left
                    kwargs[k][:, 1::2] = kwargs[k][:, 1::2] * scale + top
        image = cv2.resize(image, (new_size[0], new_size[1]))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        return image, boxes, labels, kwargs


class RandomResizePadding(augment_bbox.RandomResizePadding):
    def __init__(self, size, p=0.5):
        """
        随机使用ResizePadding和Resize，以提高泛化性
        :param p: 随机概率
        """
        self.p = p
        self.resize_padding = ResizePadding(size)
        self.resize = Resize(size)

    def __call__(self, img, boxes, labels, **kwargs):
        if random.random() < self.p:
            img, boxes, labels, kwargs = self.resize_padding(img, boxes, labels, **kwargs)
        else:
            img, boxes, labels, kwargs = self.resize(img, boxes, labels, **kwargs)
        return img, boxes, labels, kwargs


class RandomAffineResizePadding(augment_bbox.RandomAffineResizePadding):
    def __init__(self, degrees=15, output_size=None, p=0.5, check=True):
        """
        保持原始图像内容比，避免失真,短边会0填充，随机旋转进行仿生变换
        PS：如果训练时加入“保持原始图像内容比”的数据增强，那测试也建议加上“保持原始图像内容比”
        不然测试效果会差一点
        :param degrees:随机旋转的角度
        :param output_size: 仿生变换输出的大小
        :param p: 仿生变换的概率P:
        :param check: True False，检测使用ToPercentCoords()或ProtectBBoxes限制box越界情况
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.output_size = output_size
        # 去除因为旋转导致越界的boxes
        self.check = check
        self.p = p
        self.protect_bboxes = ProtectBoxes(norm=False)
        if self.output_size:
            self.size = Resize(self.output_size)

    def affine(self, image, boxes, labels, angle, output_size, **kwargs):
        if not output_size:
            h, w, _ = image.shape
            output_size = [w, h]
        image, boxes, center, scale, kwargs = affine_transform.AffineTransform. \
            affine_transform(image, boxes, output_size, rot=angle, **kwargs)
        return image, boxes, labels, kwargs

    def __call__(self, image, boxes, labels, **kwargs):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        if random.random() < self.p and len(boxes) > 0:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image, boxes, labels, kwargs = self.affine(image, boxes, labels, angle, self.output_size, **kwargs)
        elif self.output_size:
            image, boxes, labels, kwargs = self.size(image, boxes, labels, **kwargs)
        # 去除因为旋转导致越界的boxes
        if self.check:
            image, boxes, labels, kwargs = self.protect_bboxes(image, boxes, labels, **kwargs)
        return image, boxes, labels, kwargs


class RandomBoxNoise(augment_bbox.RandomBoxNoise):
    def __init__(self, p=0.5, noise=[3, 3]):
        """
        boxes添加随机扰动
        :param p:
        :param noise: [noise_x,noise_y]添加(x,y)方向的扰动，noise值越大，Box扰动越大
        """
        self.noise = noise
        self.p = p

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        # boxes = np.asarray(boxes)
        if random.random() < self.p:
            rx = int(random.uniform(-self.noise[0], self.noise[0]))
            ry = int(random.uniform(-self.noise[1], self.noise[1]))
            boxes = boxes + [rx, ry] * 2
            height, width, _ = image.shape
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
            if kwargs:
                for k in kwargs.keys():
                    kwargs[k][:, 0::2] = kwargs[k][:, 0::2] + rx
                    kwargs[k][:, 1::2] = kwargs[k][:, 1::2] + ry
                    kwargs[k][:, 0::2] = np.clip(kwargs[k][:, 0::2], 0, width - 1)
                    kwargs[k][:, 1::2] = np.clip(kwargs[k][:, 1::2], 0, height - 1)
        return image, boxes, labels, kwargs


class RandomCropLarge(augment_bbox.RandomCropLarge):
    def __init__(self, min_size=[320, 320], p=0.8):
        """
        RandomCropLarge与RandomCrop类似都是实现随机裁剪，
        RandomCrop会自定计算裁剪区域，为保证不会裁剪掉boxes的区域，其裁剪幅度较小
        RandomCropLarge可设定最小裁剪区域min_size，裁剪幅度较大，可能会裁剪部分box区域
        :param min_size: 最小crop的大小[W,H]
        :param p:概率P
        """
        self.p = p
        self.min_size = min_size
        self.clip = ClipBoxes(norm=False)

    def __call__(self, image, boxes, labels, **kwargs):
        if random.random() < self.p:
            return self.random_crop(image, boxes, labels, **kwargs)
        return image, boxes, labels, kwargs

    def random_crop(self, image, boxes, labels, **kwargs):
        for i in range(100):
            box = self.random_box(image, self.min_size)
            cimage = image[box[1]: box[3], box[0]: box[2]]
            cboxes = boxes.copy()
            clabels = labels.copy()
            ckwargs = copy.deepcopy(kwargs)  # fix a bug
            cboxes[:, 0::2] = cboxes[:, 0::2] - box[0]
            cboxes[:, 1::2] = cboxes[:, 1::2] - box[1]
            if ckwargs:
                for k in ckwargs.keys():
                    ckwargs[k][:, 0::2] = ckwargs[k][:, 0::2] - box[0]
                    ckwargs[k][:, 1::2] = ckwargs[k][:, 1::2] - box[1]
            cimage, cboxes, clabels, ckwargs = self.clip(cimage, cboxes, clabels, **ckwargs)
            if len(cboxes) > 0:
                return cimage, cboxes, clabels, ckwargs
        return image, boxes, labels, kwargs


class RandomCrop(object):
    """ 随机裁剪"""

    def __init__(self, p=0.8, margin_rate=0.5):
        """
        RandomCropLarge与RandomCrop类似都是实现随机裁剪，
        RandomCrop会自定计算裁剪区域，为保证不会裁剪掉boxes的区域，其裁剪幅度较小
        RandomCropLarge可设定最小裁剪区域min_size，裁剪幅度较大，可能会裁剪部分box区域
        :param p: 实现随机裁剪的概率
        :param margin_rate: 随机裁剪的幅度
        """
        self.p = p
        self.margin_rate = margin_rate

    def __call__(self, img, boxes, labels, **kwargs):
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
            # crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            # crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            # crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            # crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            crop_xmin = int(max(0, random.uniform(0, max_l_trans * self.margin_rate)))
            crop_ymin = int(max(0, random.uniform(0, max_u_trans * self.margin_rate)))
            crop_xmax = int(min(w_img, random.uniform(w_img - max_r_trans * self.margin_rate, w_img)))
            crop_ymax = int(min(h_img, random.uniform(h_img - max_d_trans * self.margin_rate, h_img)))
            img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            h_img, w_img, _ = img.shape
            if len(boxes) > 0:
                # boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_xmin
                # boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_ymin
                boxes = boxes - [crop_xmin, crop_ymin, crop_xmin, crop_ymin]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_img)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_img)
                if kwargs:
                    for k in kwargs.keys():
                        scale = [crop_xmin, crop_ymin] * int(kwargs[k].shape[1] / 2)
                        kwargs[k] = kwargs[k] - scale
                        kwargs[k][:, 0:len(scale):2] = np.clip(kwargs[k][:, 0:len(scale):2], 0, w_img)
                        kwargs[k][:, 1:len(scale):2] = np.clip(kwargs[k][:, 1:len(scale):2], 0, h_img)
        return img, boxes, labels, kwargs


class RandomRotation(augment_bbox.RandomRotation):
    """ 随机旋转"""

    def __init__(self, degrees=15, p=0.5, check=True):
        """
        随机旋转，BUG,loss出现inf,可能因为旋转导致部分box越界或者丢失
        :param degrees:
        :param p:随机旋转的概率
        :param check: True False:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.p = p
        # 去除因为旋转导致越界的boxes
        self.check = check
        self.protect_bboxes = ProtectBoxes(norm=False)

    def __call__(self, image, boxes, labels, **kwargs):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        # 顺时针旋转90度
        if random.random() < self.p and len(boxes) > 0:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            h, w, _ = image.shape
            center = (w / 2., h / 2.)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, dsize=(w, h))
            num_boxes = len(boxes)
            if num_boxes > 0:
                points, num_boxes = affine_transform.get_boxes2points(boxes)
                for i in range(num_boxes):
                    points[i, :] = affine_transform.rotate_points(points[i, :], centers=[center], angle=angle, height=h)
                boxes = affine_transform.get_points2bboxes(points)
            if kwargs:
                for k in kwargs.keys():
                    points = kwargs[k].reshape(-1, 2)
                    points = affine_transform.rotate_points(points, centers=[center], angle=angle, height=h)
                    kwargs[k] = points.reshape(num_boxes, -1)
        # 去除因为旋转导致越界的boxes
        if self.check:
            image, boxes, labels, kwargs = self.protect_bboxes(image, boxes, labels, **kwargs)
        return image, boxes, labels, kwargs


class RandomRot90(object):
    """ 随机旋转90度"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
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
        return img, boxes, labels


class RandomHorizontalFlip(augment_bbox.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.
    随机水平翻转
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, flip_index=[]):
        """
        :param p: 随机翻转的概率
        :param flip_index: 翻转后，对于的关键点也进行翻转，如果输入的关键点没有左右关系的，
                           请设置flip_index，以便翻转时，保证index的关系
                           如，对于人脸关键点：flip_index=[2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
        """
        self.p = p
        self.flip_index = flip_index

    def __call__(self, image, boxes, labels, **kwargs):
        """
        :param image:
        :param boxes: xmin,ymin,xmax,ymax
        :param labels:
        :return:
        """
        if random.random() < self.p:
            height, width, _ = image.shape
            image = image[:, ::-1, :]
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                if kwargs:
                    for k in kwargs.keys():
                        cols = kwargs[k].shape[1]
                        index1 = list(range(0, cols, 2))  # for y
                        kwargs[k][:, index1] = width - kwargs[k][:, index1]
                        # flip_index = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
                        if self.flip_index:
                            kwargs[k] = kwargs[k][:, self.flip_index]
        return image, boxes, labels, kwargs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(augment_bbox.RandomVerticalFlip):
    def __call__(self, image, boxes, labels, **kwargs):
        """
        :param image:
        :param boxes: xmin,ymin,xmax,ymax
        :param labels:
        :return:
        """
        if random.random() < self.p:
            height, width, _ = image.shape
            image = image[::-1, :, :]
            if len(boxes) > 0:
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                if kwargs:
                    for k in kwargs.keys():
                        index1 = [1, 3, 5, 7, 9]
                        kwargs[k][:, index1] = height - kwargs[k][:, index1]
                        # index2 = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
                        # kwargs[k] = kwargs[k][:, index2]
        return image, boxes, labels, kwargs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomMosaic(augment_bbox.RandomMosaic):
    def __init__(self, size=None, p=0.5, samples=[2, 3, 4], flip=True):
        pass


class SwapChannels(augment_bbox.SwapChannels):
    """交换图像颜色通道的顺序"""

    def __init__(self, swaps=[], p=1.0):
        """
        由于输入可能是RGB或者BGR格式的图像，随机交换通道顺序可以避免图像通道顺序的影响
        :param swaps:指定交换的颜色通道顺序，如[2,1,0]
                     如果swaps=[]或None，表示随机交换顺序RGB或者BGR
        :param p:概率
        """
        self.p = p
        self.swap_list = []
        if not swaps:
            self.swap_list = [[0, 1, 2], [2, 1, 0]]
        else:
            self.swap_list = [swaps]
        self.swap_index = np.arange(len(self.swap_list))

    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if random.random() < self.p:
            index = np.random.choice(self.swap_index)
            swap = self.swap_list[index]
            image = image[:, :, swap]
        return image, boxes, labels, kwargs


class RandomColorJitter(augment_bbox.RandomColorJitter):
    def __call__(self, image, boxes=None, labels=None, **kwargs):
        if random.random() < self.p:
            is_np = isinstance(image, np.ndarray)
            if is_np:
                image = Image.fromarray(image)
            # image = self.random_choice(image)
            image = self.color_transforms(image)
            if is_np:
                image = np.asarray(image)
        return image, boxes, labels, kwargs


class RandomContrastBrightness(augment_bbox.RandomContrastBrightness):
    def __call__(self, image, boxes=None, labels=None, **kwargs):
        image, boxes, labels = self.random_contrast(image, boxes, labels)
        image, boxes, labels = self.random_brightness(image, boxes, labels)
        return image, boxes, labels, kwargs


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
    print("image :{}".format(image.shape))
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
    # image_processing.show_image_boxes("train", tmp_image, bboxes)
    tmp_image = image_processing.draw_image_bboxes_text(tmp_image, bboxes, labels)
    image_processing.cv_show_image("train", tmp_image)


def demo_for_augment():
    from utils import image_processing
    image_path = "test.jpg"
    bboxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
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
    input_size = [480, 480]
    images = image_processing.read_image(image_path)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    labels = np.asarray(classes, dtype=np.int32)
    land_mark = np.asarray(land_mark).reshape(-1, 10)
    flip_index = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
    # augment = RandomCrop(p=0.5, margin_rate=0.3)
    # augment = RandomCropLarge(min_size=[320, 320], p=1.0)
    augment = Compose([
        # augment_bbox_landm.ProtectBoxes(),
        # augment_bbox.RandomRot90(),  # 随机横屏和竖屏
        # RandomRotation(degrees=15),
        ProtectBoxes(norm=False),
        # RandomHorizontalFlip(flip_index=flip_index),
        # RandomBoxesPaste(bg_dir=bg_dir),
        # RandomVerticalFlip(),
        # RandomCrop(),
        # RandomCropLarge(min_size=input_size),
        # RandomContrastBrightness(),
        RandomAffineResizePadding(degrees=15, output_size=input_size),
        # RandomAffineResizePadding(degrees=15),
        # ResizePadding(input_size),
        # RandomResizePadding(input_size, p=0.5),
        # Resize(input_size),
        # ResizePadding(input_size),
        RandomColorJitter(),
        # SwapChannels(),
    ])

    for i in range(1000):
        print("===" * 10)
        dst_image, dst_boxes, dst_labels, dst_kwargs = augment(images,
                                                               bboxes.copy(),
                                                               labels.copy(),
                                                               land_mark=land_mark.copy())
        dst_land_mark = dst_kwargs["land_mark"]
        dst_land_mark = dst_land_mark.reshape(-1, 5, 2)
        show_landmark_image(dst_image, dst_boxes, dst_land_mark, dst_labels, normal=False, transpose=False)


def demo_for_rotation():
    from utils import image_processing
    input_size = [320, 320]
    image_path = "test.jpg"
    images = image_processing.read_image(image_path)
    bboxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    labels = [1, 2]
    flip_index = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
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
    bboxes = np.asarray(bboxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    land_mark = np.asarray(land_mark).reshape(-1, 10)

    augment = RandomAffineResizePadding(degrees=15, output_size=None, p=0.5)
    augment = RandomRotation(degrees=15, p=1.0)
    # 顺时针旋转90度
    image_processing.show_image_boxes("src", images, bboxes, waitKey=10, color=(0, 255, 0))
    for angle in range(360 * 10):
        augment.set_degrees(angle)
        dst_image, dst_boxes, dst_labels, dst_kwargs = augment(images.copy(),
                                                               bboxes.copy(),
                                                               labels.copy(),
                                                               land_mark=land_mark.copy())
        print("==" * 10)
        dst_land_mark = dst_kwargs["land_mark"]
        dst_land_mark = dst_land_mark.reshape(-1, 5, 2)
        show_landmark_image(dst_image, dst_boxes, dst_land_mark, dst_labels, normal=False, transpose=False)


if __name__ == "__main__":
    # demo_for_augment()
    demo_for_rotation()
