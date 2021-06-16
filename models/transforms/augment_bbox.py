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
import random
from models.transforms import affine_transform


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


def check_bboxes(boxes):
    """
    :param boxes:
    :return:
    """
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        assert xmax > xmin
        assert ymax > ymin


class ProtectBoxes(object):
    def __init__(self, norm=False, pct_th=0.01, iou_th=0.6):
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

    def __call__(self, image, boxes=None, labels=None):
        image, boxes, labels = self.clip(image, boxes, labels)
        image, boxes, labels = self.min_boxes(image, boxes, labels)
        return image, boxes, labels


class ClipBoxes(object):
    def __init__(self, norm=False, iou_th=0.6):
        """
        用于限制boxes边界范围，防止越界的情况
        :param norm: True：输入的boxes是归一化坐标，即boxes/[w,h,w,h]，范围(0,1)
                    False：输入的boxes是像素坐标，即(xmin,ymin,xmax,ymax)
        :param iou_th: clip会缩小boxes，因此如果clip前后的IOU小于该阈值，box会被丢弃
        """
        self.norm = norm
        self.iou_th = iou_th
        self.index = []

    @staticmethod
    def cal_iou(box1, box2):
        """
        :param box1: = [xmin1, ymin1, xmax1, ymax1]
        :param box2: = [xmin2, ymin2, xmax2, ymax2]
        :return:
        """
        xmin1, ymin1, xmax1, ymax1 = box1[:4]
        xmin2, ymin2, xmax2, ymax2 = box2[:4]
        # 计算每个矩形的面积
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

        # 计算相交矩形
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        area = w * h  # C∩G的面积
        iou = area / (s1 + s2 - area)
        return iou

    def check_overlap(self, image, boxes, labels, cboxes, iou_th=0.6):
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
        return image, boxes, labels

    def get_index(self):
        return self.index

    def clip(self, image, boxes, labels):
        """限制boxes边界范围"""
        cboxes = boxes.copy()
        if self.norm:
            cboxes = np.clip(boxes, 0, 1)
        else:
            height, width, _ = image.shape
            cboxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
            cboxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        return image, cboxes, labels

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            image, cboxes, labels = self.clip(image, boxes, labels)
            image, boxes, labels = self.check_overlap(image, boxes, labels, cboxes, iou_th=self.iou_th)
        check_bboxes(boxes)
        return image, boxes, labels


class IgnoreMinBoxes(object):
    def __init__(self, norm=True, pct_th=0.01):
        """
        去除小框，避免出现Nan值
        :param norm: True：输入的boxes是归一化坐标，即boxes/[w,h,w,h]，范围(0,1)
                    False：输入的boxes是像素坐标，即(xmin,ymin,xmax,ymax)
        :param pct_th: 0.01表示如果长宽有一个不足图像的1%，则去除
        """
        self.norm = norm
        self.pct_th = pct_th

    def __call__(self, image, boxes=None, labels=None):
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
            if not self.norm:
                boxes = boxes * [width, height, width, height]
        return image, boxes, labels


class IgnoreBadBoxes(object):
    """去除不合法的bbox,如小于0，越界的bbox"""

    def __init__(self):
        pass

    def get_mask_index(self, mask):
        """
        :param mask:
        :return:
        """
        index = [sum(mask[i, :]) == len(mask[i, :]) for i in range(len(mask))]
        index = np.array(index)
        return index

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, _ = image.shape
            index0 = self.get_mask_index(boxes >= 0)
            index1 = self.get_mask_index(boxes < [width, height, width, height])
            index = index0 * index1
            boxes = boxes[index]
            labels = labels[index]
        check_bboxes(boxes)
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


class UnNormalizeBoxesCoords(object):
    """将归一化boxes坐标转换为图像坐标"""

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes = boxes * [width, height, width, height]
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels


class NormalizeBoxesCoords(object):
    """将图像坐标转为归一化boxes坐标"""

    def __call__(self, image, boxes=None, labels=None):
        if len(boxes) > 0:
            height, width, channels = image.shape
            boxes = boxes / [width, height, width, height]
        boxes = np.asarray(boxes, np.float32)
        return image, boxes, labels


class Resize(object):
    """resize"""

    def __init__(self, size=[300, 300]):
        """
        图像resize,可能会导致图像内容失真
        :param size:
        """
        self.size = tuple(size)

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        if not boxes is None and len(boxes) > 0:
            scale = [self.size[0] / width, self.size[1] / height] * 2
            boxes = boxes * scale
        image = cv2.resize(image, (self.size[0], self.size[1]))
        return image, boxes, labels


class ResizePadding(object):
    def __init__(self, size=[300, 300]):
        """
        等比例图像resize,保持原始图像内容比，避免失真,短边会0填充
        :param size:
        """
        self.size = tuple(size)

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        scale = min([self.size[0] / width, self.size[1] / height])
        new_size = [int(width * scale), int(height * scale)]
        pad_w = self.size[0] - new_size[0]
        pad_h = self.size[1] - new_size[1]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        if not boxes is None and len(boxes) > 0:
            boxes = boxes * scale
            boxes = boxes + [left, top, left, top]
        image = cv2.resize(image, (new_size[0], new_size[1]))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        return image, boxes, labels


class RandomResizePadding():
    def __init__(self, size, p=0.5):
        """
        随机使用ResizePadding和Resize，以提高泛化性
        :param p: 随机概率
        """
        self.p = p
        self.resize_padding = ResizePadding(size)
        self.resize = Resize(size)

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            if random.random() < 0.5:
                img, boxes, labels = self.resize_padding(img, boxes, labels)
            else:
                img, boxes, labels = self.resize(img, boxes, labels)
        return img, boxes, labels


class RandomAffineResizePadding(object):
    def __init__(self, output_size, degrees=15, check=False):
        """
        保持原始图像内容比，避免失真,短边会0填充，随机旋转进行仿生变换
        :param output_size: 仿生变换输出的大小
        :param degrees:随机旋转的角度
        :param check: True False:
               使用ToPercentCoords()或ProtectBBoxes限制box越界情况
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
        self.protect_bboxes = ProtectBoxes(norm=False)

    def affine(self, image, boxes, labels, angle):
        image, boxes, center, scale, kwargs = affine_transform.AffineTransform. \
            affine_transform(image, boxes, self.output_size, rot=angle)
        return image, boxes, labels

    def __call__(self, image, boxes, labels):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :return:
        '''
        if len(boxes) > 0:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image, boxes, labels = self.affine(image, boxes, labels, angle)
        # 去除因为旋转导致越界的boxes
        if self.check:
            image, boxes, labels = self.protect_bboxes(image, boxes, labels)
        return image, boxes, labels

    def test_image(self, image, boxes, labels):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        from utils import image_processing
        # 顺时针旋转90度
        image_processing.show_image_boxes("src", image, boxes, waitKey=10)
        for angle in range(360 * 10):
            # angle = 30
            trans_image, trans_boxes, trans_labels = self.affine(image, boxes, labels, angle)
            # trans_image, trans_boxes, labels = self.rotation_v2(image, boxes, labels, angle)
            trans_image, trans_boxes, trans_labels = self.protect_bboxes(trans_image, trans_boxes, trans_labels)
            print("==" * 10)
            print("angle:{}".format(angle))
            print("shape:{},bboxes     ：{}".format(image.shape, boxes))
            print("shape:{},trans_boxes：{},trans_labels:{}".format(trans_image.shape, trans_boxes, trans_labels))
            trans_image = image_processing.draw_image_bboxes_text(trans_image, trans_boxes, trans_labels)
            cv2.imshow("image", trans_image)
            cv2.waitKey(0)
        return image, boxes, labels


class RandomBoxNoise(object):
    def __init__(self, p=0.5, noise=[3, 3]):
        """
        boxes添加随机扰动
        :param p:
        :param noise: [noise_x,noise_y]添加(x,y)方向的扰动，noise值越大，Box扰动越大
        """
        self.noise = noise
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        # boxes = np.asarray(boxes)
        if random.random() < self.p:
            rx = int(random.uniform(-self.noise[0], self.noise[0]))
            ry = int(random.uniform(-self.noise[1], self.noise[1]))
            boxes = boxes + [rx, ry] * 2
            height, width, _ = image.shape
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
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


class RandomCropLarge(object):
    def __init__(self, min_size=[320, 320], p=0.8, iou_th=0.6):
        """
        RandomCropLarge与RandomCrop类似都是实现随机裁剪，
        RandomCrop会自定计算裁剪区域，为保证不会裁剪掉boxes的区域，其裁剪幅度较小
        RandomCropLarge可设定最小裁剪区域min_size，裁剪幅度较大，可能会裁剪部分box区域
        :param min_size: 最小crop的大小[W,H]
        :param p:概率P
        :param iou_th: clip会缩小boxes，因此如果clip前后的IOU小于该阈值，box会被丢弃
        """
        self.p = p
        self.min_size = min_size
        self.clip = ClipBoxes(norm=False, iou_th=iou_th)

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return self.random_crop(image, boxes, labels)
        return image, boxes, labels

    def random_box(self, image, min_size):
        """
        Get parameters for ``crop`` for a random crop.
        :param image:
        :param min_size: 最小crop的大小
        :return:
        """
        h, w, d = image.shape
        tw, th = int(random.uniform(min_size[0], w)), int(random.uniform(min_size[1], h))
        y = int(random.uniform(0, h - th + 1))
        x = int(random.uniform(0, w - tw + 1))
        box = [x, y, x + tw, y + th]
        return box

    def random_crop(self, image, boxes, labels):
        for i in range(100):
            box = self.random_box(image, self.min_size)
            cimage = image[box[1]: box[3], box[0]: box[2]]
            cboxes = boxes.copy()
            clabels = labels.copy()
            cboxes[:, 0::2] = cboxes[:, 0::2] - box[0]
            cboxes[:, 1::2] = cboxes[:, 1::2] - box[1]
            cimage, cboxes, clabels = self.clip(cimage, cboxes, clabels)
            # 为了避免因裁剪导致boxes为空，当len(boxes)=0时继续循环，最多循环100次
            if len(cboxes) > 0:
                return cimage, cboxes, clabels
        return image, boxes, labels


class RandomCrop(object):
    """ 随机裁剪"""

    def __init__(self, p=0.5, margin_rate=0.5):
        """
        RandomCropLarge与RandomCrop类似都是实现随机裁剪，
        RandomCrop会自定计算裁剪区域，为保证不会裁剪掉boxes的区域，其裁剪幅度较小
        RandomCropLarge可设定最小裁剪区域min_size，裁剪幅度较大，可能会裁剪部分box区域
        :param p: 实现随机裁剪的概率
        :param margin_rate: 随机裁剪的幅度
        """
        self.p = p
        self.margin_rate = margin_rate

    def __call__(self, img, boxes, labels):
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
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_img - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_img - 1)
        return img, boxes, labels


class RandomAffine(object):
    """ 随机Padding"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0), ], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))
            boxes[:, [0, 2]] = boxes[:, [0, 2]] + tx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] + ty
        return img, boxes, labels

    def test_image(self, image, input_boxes, classes):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        from utils import image_processing
        # 顺时针旋转90度
        for angle in range(480):
            # angle = random.uniform(self.degrees[0], self.degrees[1])
            num_boxes = len(input_boxes)
            if num_boxes > 0:
                dst_image, boxes, labels = self.__call__(image.copy(), input_boxes.copy(), classes)
                dst_image = image_processing.draw_image_bboxes_text(dst_image, boxes, labels)
                # dst_image = image_processing.draw_points_text(dst_image, [center], texts=["center"], drawType="simple")
                # dst_image = image_processing.draw_landmark(dst_image, points, point_color=(255, 0, 0), vis_id=True)
                cv2.imshow("image", dst_image)
                cv2.waitKey(0)
        return image, boxes, labels


class RandomRotation(object):
    def __init__(self, degrees=15, p=0.5, check=False):
        """
        随机旋转，BUG,loss出现inf,可能因为旋转导致部分box越界或者丢失
        :param degrees:
        :param p:随机旋转的概率
        :param check: True False:
               使用ToPercentCoords()或ProtectBBoxes限制box越界情况
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

    def rotation_v1(self, image, boxes, labels, angle):
        """
        :param image:
        :param boxes:
        :param labels:
        :param angle: 旋转角度
        :return:
        """
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
        return image, boxes, labels

    def rotation_v2(self, image, boxes, labels, angle, output_size=None):
        """
        :param image:
        :param boxes:
        :param labels:
        :param angle: 旋转角度
        :param output_size: 输出大小
        :return:
        """
        h, w, _ = image.shape
        if not output_size:
            output_size = [w, h]
        image, boxes = affine_transform.affine_transform_for_boxes(image,
                                                                   boxes,
                                                                   output_size=output_size,
                                                                   rot=angle)
        return image, boxes, labels

    def __call__(self, image, boxes, labels):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        if random.random() < self.p and len(boxes) > 0:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image, boxes, labels = self.rotation_v1(image, boxes, labels, angle)
            # 去除因为旋转导致越界的boxes
            if self.check:
                image, boxes, labels = self.protect_bboxes(image, boxes, labels)
        return image, boxes, labels

    def test_image(self, image, boxes, labels):
        '''
        :param image: nparray img
        :param boxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
        :param p: 随机比例
        :return:
        '''
        from utils import image_processing
        # 顺时针旋转90度
        image_processing.show_image_boxes("src", image, boxes, waitKey=10)
        for angle in range(360 * 10):
            # angle = 30
            trans_image, trans_boxes, trans_labels = self.rotation_v1(image, boxes, labels, angle)
            # trans_image, trans_boxes, labels = self.rotation_v2(image, boxes, labels, angle)
            trans_image, trans_boxes, trans_labels = self.protect_bboxes(trans_image, trans_boxes, trans_labels)
            print("==" * 10)
            print("angle:{}".format(angle))
            print("shape:{},bboxes     ：{}".format(image.shape, boxes))
            print("shape:{},trans_boxes：{},trans_labels:{}".format(trans_image.shape, trans_boxes, trans_labels))
            trans_image = image_processing.draw_image_bboxes_text(trans_image, trans_boxes, trans_labels)
            cv2.imshow("image", trans_image)
            cv2.waitKey(0)
        return image, boxes, labels


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


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    随机水平翻转
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
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
        return image, boxes, labels

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
    """交换图像颜色通道的顺序"""

    def __init__(self, swaps=[], p=1.0):
        """
        由于输入可能是RGB或者BGR格式的图像，随机交换通道顺序可以避免图像通道顺序的影响
        :param swaps:指定交换的颜色通道顺序，如[2,1,0]
                     如果swaps=[]或None，表示随机交换顺序
        :param p:概率
        """
        self.p = p
        self.swap_list = []
        if not swaps:
            self.swap_list = [[0, 1, 2], [2, 1, 0]]
        else:
            self.swap_list = [swaps]
        self.swap_index = np.arange(len(self.swap_list))

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            index = np.random.choice(self.swap_index)
            swap = self.swap_list[index]
            image = image[:, :, swap]
        return image, boxes, labels


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


class RandomColorJitter(object):
    def __init__(self, p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
        """
        :param p:
        :param brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        :param contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        :param saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        :param hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.(色调建议设置0.1，避免颜色变化过大)
        """
        from torchvision import transforms
        self.p = p
        self.random_choice = transforms.RandomChoice([
            transforms.ColorJitter(brightness=brightness),
            transforms.ColorJitter(contrast=contrast),
            transforms.ColorJitter(saturation=saturation),
            transforms.ColorJitter(hue=hue),
        ])
        self.color_transforms = transforms.ColorJitter(brightness=brightness,
                                                       contrast=contrast,
                                                       saturation=saturation,
                                                       hue=hue)

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            is_pil = isinstance(image, np.ndarray)
            if is_pil:
                image = Image.fromarray(image)
            # image = self.random_choice(image)
            image = self.color_transforms(image)
            if is_pil:
                image = np.asarray(image)
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


class PortraitMode():
    def __init__(self):
        """
        竖屏模式
        """
        self.rot = RandomRot90(p=1.0)

    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        if h <= w:
            # 如果h < w，是横屏图片,需要旋转为竖屏
            img, boxes, labels = self.rot(img, boxes, labels)
        return img, boxes, labels


class LandscapeMode():
    def __init__(self):
        """
        横屏模式
        """
        self.rot = RandomRot90(p=1.0)

    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        if h >= w:
            # 如果h > w，是竖屏图片,需要旋转为横屏
            img, boxes, labels = self.rot(img, boxes, labels)
        return img, boxes, labels


class OrientationModel():
    def __init__(self, width, height):
        """
        根据长宽比，自适应旋转90°，获得对应的横屏或者竖屏的图像
        当width>height时，输出始终为获得横屏图像
        当width<height时，输出始终为获得竖屏图像
        :param width:
        :param height:
        """
        self.aspect_ratio = height / width
        self.portrait = PortraitMode()  # 竖屏模式
        self.landscape = LandscapeMode()  # 横屏模式

    def __call__(self, img, boxes, labels):
        if self.aspect_ratio >= 1.0:
            img, boxes, labels = self.portrait(img, boxes, labels)  # 竖屏模式
        else:
            img, boxes, labels = self.landscape(img, boxes, labels)  # 横屏模式
        return img, boxes, labels


class RandomOrientationModel():
    def __init__(self, p=0.5):
        """
        随机获得横屏或者竖屏的图像，相当于对输入图像随机旋转90°
        :param p: 随机获得横屏或者竖屏的图像的概率,值为0时，表示输出原始图像
        """
        self.p = p
        self.portrait = PortraitMode()  # 竖屏模式
        self.landscape = LandscapeMode()  # 横屏模式

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            if random.random() < 0.5:
                img, boxes, labels = self.portrait(img, boxes, labels)  # 竖屏模式
            else:
                img, boxes, labels = self.landscape(img, boxes, labels)  # 横屏模式
        return img, boxes, labels


class RandomBoxesPaste(object):
    """ 实现随机背景贴图"""

    def __init__(self, p=0.5, bg_dir="bg_image/"):
        """
        实现随机背景贴图，适用于检测对象之间没有overlap关系的目标检测数据增强
        使用条件：
        (1)检测对象之间没有overlap关系,如：face和person之间boxes经常存在overlap关系，
           随机贴图容易出现face的box缺漏的问题
        (2)背景图不能含有目标检测的对象，避免背景的干扰
        (3)对于单目标检测，都可以使用该方法，实现数据增强
        (4)支持最多2个boxes的随机贴图，如果输入boxes多余2个，则自动丢弃多余的boxes
        :param p: 实现随机背景贴图的概率
        :param bg_dir: 背景图库，PS：背景图不能含有目标检测的对象，避免背景的干扰
        """
        from utils import file_processing
        self.p = p
        self.max_scale = 2 / 5
        self.scale = [1.2, 1.2]
        self.bg_image_list = file_processing.get_files_lists(bg_dir, subname="")
        self.bg_nums = len(self.bg_image_list)
        self.is_rgb = True

    def random_read_bg_image(self, is_rgb=False, crop_rate=0.5):
        index = int(np.random.uniform(0, self.bg_nums))
        image_path = self.bg_image_list[index]
        # image_path = self.bg_image_list[0]
        image = cv2.imread(image_path)
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape
        xmin, ymin, xmax, ymax = self.extend_bboxes([0, 0, w_img, h_img], [0.8, 0.8])
        crop_xmin = int(random.uniform(0, xmin * crop_rate))
        crop_ymin = int(random.uniform(0, ymin * crop_rate))
        crop_xmax = int(min(w_img, random.uniform(w_img - xmax * crop_rate, w_img)))
        crop_ymax = int(min(h_img, random.uniform(h_img - ymax * crop_rate, h_img)))
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        return image

    @staticmethod
    def extend_bboxes(box, scale=[1.0, 1.0]):
        """
        :param box: [xmin, ymin, xmax, ymax]
        :param scale: [sx,sy]==>(W,H)
        :return:
        """
        sx = scale[0]
        sy = scale[1]
        xmin, ymin, xmax, ymax = box[:4]
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        ex_w = (xmax - xmin) * sx
        ex_h = (ymax - ymin) * sy
        ex_xmin = cx - 0.5 * ex_w
        ex_ymin = cy - 0.5 * ex_h
        ex_xmax = ex_xmin + ex_w
        ex_ymax = ex_ymin + ex_h
        ex_box = [ex_xmin, ex_ymin, ex_xmax, ex_ymax]
        return ex_box

    @staticmethod
    def cv_paste_image(im, mask, boxes, start_point=(0, 0)):
        """
        :param im:
        :param start_point:
        :param mask:
        :return:
        """
        xim, ymin = start_point
        shape = mask.shape  # h, w, d
        im[ymin:(ymin + shape[0]), xim:(xim + shape[1])] = mask
        boxes = boxes + [start_point[0], start_point[1], start_point[0], start_point[1]]
        return im, boxes

    def center2box(self, center):
        cx, cy, w, h = center
        box = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
        return box

    @staticmethod
    def pil_paste_image(im, mask, boxes, start_point=(0, 0)):
        """
        :param im:
        :param mask:
        :param start_point: tupe
        :return:
        """
        assert isinstance(start_point, tuple)
        out = Image.fromarray(im)
        mask = Image.fromarray(mask)
        out.paste(mask, start_point)
        boxes = boxes + [start_point[0], start_point[1], start_point[0], start_point[1]]
        return np.asarray(out), boxes

    @staticmethod
    def get_random_point(w, h, edge_wh=[0, 0]):
        """
        获得随机点
        :param w:
        :param h:
        :param edge_wh: 右小角的边缘宽度
        :return:
        """
        dst_w = int(np.random.uniform(0, w - edge_wh[0]))
        dst_h = int(np.random.uniform(0, h - edge_wh[1]))
        return [dst_w, dst_h]

    def get_crop_roi(self, img, box, edge_wh_scale=[1.0, 1.0]):
        """
        获得裁剪区域ROI
        :param img:
        :param box:
        :param edge_wh_scale: 裁剪区域ROI外扩大小
        :return:
        """
        h_img, w_img, _ = img.shape
        xmin, ymin, xmax, ymax = self.extend_bboxes(box, edge_wh_scale)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w_img, xmax)
        ymax = min(h_img, ymax)
        roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        box = box - [xmin, ymin, xmin, ymin]
        return roi, box

    def check_distance(self, point0, point1, dis_th):
        """
        检测点与点之间的距离关系
        :param point0:
        :param point1:
        :param dis_th: 距离阈值
        :return: True:存在overlap，False:不存在overlap
        """
        r = abs(point0[0] - point1[0]) < dis_th and abs(point0[1] - point1[1]) < dis_th
        return r

    def check_point(self, point, max_wh, roi_wh):
        rw = point[0] + roi_wh[0] < max_wh[0] and point[0] >= 0
        rh = point[1] + roi_wh[1] < max_wh[1] and point[1] >= 0
        return rw * rh

    def random_paste(self, img, boxes, labels, bg_image):
        h_img, w_img, _ = img.shape
        bg_image = cv2.resize(bg_image, dsize=(w_img, h_img))
        h_bg, w_bg, _ = bg_image.shape
        start_point = [0, 0]
        bg_boxes = []
        bg_labels = []
        max_wh = 0
        # 目前支持最多2个boxes的随机贴图
        for i in range(min(len(boxes), 2)):
            box = boxes[i, :]
            label = labels[i]
            roi, box = self.get_crop_roi(img, box)
            roi_h, roi_w, _ = roi.shape
            # dis_th：贴图之间的间距，避免overlap
            dis_th = max(roi_w, roi_h, max_wh)
            for k in range(10):
                point = self.get_random_point(w_bg, h_bg, edge_wh=[roi_w, roi_h])
                # print("(i,k)=({},{}),point0：{}，point1：{}".format(i, k, point, start_point))
                if not self.check_point(point, max_wh=(w_bg, h_bg), roi_wh=[roi_w, roi_h]):
                    continue
                elif not self.check_distance(point, start_point, dis_th=dis_th * 0.9):
                    start_point = point
                    bg_image, box = self.pil_paste_image(bg_image, roi, box, start_point=tuple(start_point))
                    bg_boxes.append(box)
                    bg_labels.append(label)
                    break
            max_wh = dis_th
        if len(bg_boxes) == 0:
            bg_boxes = boxes
            bg_labels = labels
            bg_image = img
        bg_boxes = np.asarray(bg_boxes)
        bg_labels = np.asarray(bg_labels)
        return bg_image, bg_boxes, bg_labels

    def __call__(self, img, boxes, labels):
        """
        Args:
            img (numpy Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            bg_image = self.random_read_bg_image(is_rgb=self.is_rgb)
            img, boxes, labels = self.random_paste(img, boxes, labels, bg_image)
        return img, boxes, labels


def show_image(image, bboxes, labels, normal=False, transpose=False):
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
    labels = np.asarray(labels)
    print("bboxes:{}".format(bboxes))
    print("labels:{}".format(labels))
    if transpose:
        image = image_processing.untranspose(image)
    h, w, _ = image.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        bboxes = bboxes * bboxes_scale
    # tmp_image = image_processing.untranspose(tmp_image)
    tmp_image = image_processing.convert_color_space(image, colorSpace="BGR")
    image_processing.show_image_boxes("train", tmp_image, bboxes)


def demo_for_bboxes():
    from utils import image_processing
    input_size = [320, 320]
    image_path = "test.jpg"
    src_boxes = [[200, 222, 439, 500], [98, 42, 160, 100], [244, 260, 297, 332]]
    # src_boxes = [[-1, -1, 300, 300], [300, 300, 400, 700]]
    classes = [1, 2, 3]
    # augment = RandomCrop(p=1.0,margin_rate=0.5)
    augment = SwapChannels()
    # augment = RandomCropLarge(p=1.0, min_size=input_size)
    # augment = ResizePadding(size=input_size)
    # augment = IgnoreBadBBoxes()
    # augment = RandomBoxesPaste(p=1.0)
    # augment = RandomOrientationModel()
    # augment = RandomBoxNoise()
    # augment = RandomColorJitter()
    src_image = image_processing.read_image(image_path)
    src_boxes = np.asarray(src_boxes)
    src_classes = np.asarray(classes)
    for i in range(1000):
        print("===" * 10)
        dst_image, dst_boxes, classes = augment(src_image.copy(), src_boxes.copy(), src_classes.copy())
        show_image(dst_image, dst_boxes, classes, normal=False, transpose=False)


def demo_RandomRotation():
    from utils import image_processing

    input_size = [320, 320]
    image_path = "test.jpg"
    # src_boxes = [[8.20251, 1, 242.2412, 699.2236],
    #              [201.14865, 204.18265, 468.605, 696.36163], [100, 100, 150, 150]]
    src_boxes = [[98, 42, 160, 100], [98 + 50, 42 + 50, 160 + 50, 100 + 50], [244, 260, 297, 332], [124, 126, 125, 135]]

    classes = np.asarray([1, 2, 3, 4])
    # src_boxes=[]
    # classes=[]
    # augment = RandomRotation(degrees=15)
    augment = RandomAffineResizePadding(input_size, degrees=15)
    # augment = RandomAffine()
    # src_image = image_processing.read_image(image_path, resize_height=800, resize_width=800)
    src_image = image_processing.read_image(image_path)
    src_boxes = np.asarray(src_boxes)
    dst_image, boxes, classes = augment.test_image(src_image.copy(), src_boxes.copy(), classes)


if __name__ == "__main__":
    # demo_for_bboxes()
    demo_RandomRotation()
