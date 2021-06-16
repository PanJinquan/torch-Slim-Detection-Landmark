# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-07-30 17:03:36
# --------------------------------------------------------
"""
import cv2
import numpy as np
from numpy import random


class MosaicAugment():
    """
    url: https://www.cnblogs.com/xsliu/p/13482381.html
    """

    def __init__(self, dataset, out_size=None, p=0.5, samples=3, from_torch=True):
        """
        dataset是一个类似于torch.utils.data.Dataset的Class类,返回img和bboxes_label
        ```
        from torch.utils.data import DataLoader, ConcatDataset
        class Dataset(object):
            def __init__(self, **kwargs):
                pass
            def __getitem__(self, index):
                img = ...
                bboxes_label = ...
                return img, bboxes_label
        ```
        :param dataset: dataset是一个类似于torch.utils.data.Dataset的Class类,返回img和bboxes_label,
                        输入的图像必须是resize同一大小(图片的长宽可以任意)
        :param out_size: 返回图片的大小[width , height]
        :param p: 随机概率
        :param samples: Mosaic需要的图片数据量{3,4}
        :param from_torch: 输入dataset是否是Pytorch数据格式
        """
        super(MosaicAugment, self).__init__()
        self._dataset = dataset
        self.out_size = out_size
        self.samples = samples
        self.from_torch = from_torch
        self.p = p

    def __len__(self):
        return len(self._dataset)

    @staticmethod
    def mosaic_for4samples(data_list, out_size=None):
        """
        需要四张图像进行拼接
        :param data_list:[data]--> data = {"img": img, "boxes": bboxes, "label": label}
        :param out_size: None or [resize-height,resize-width]
        :return:
        """
        assert len(data_list) == 4
        # 垂直方向拼接
        img_01 = np.vstack((data_list[0]["img"].copy(), data_list[1]["img"].copy()))
        img_23 = np.vstack((data_list[2]["img"].copy(), data_list[3]["img"].copy()))
        # 水平方向拼接
        img_new = np.hstack((img_01, img_23))

        h0, w0, d0 = data_list[0]["img"].shape
        dst_bboxes = data_list[0]["boxes"]
        if len(dst_bboxes) > 0:
            dst_bboxes = dst_bboxes * [w0, h0, w0, h0]
        dst_labels = data_list[0]["label"]

        h1, w1, d1 = data_list[1]["img"].shape
        label1 = data_list[1]["label"]
        boxes1 = data_list[1]["boxes"]
        if len(boxes1) > 0:
            boxes1 = boxes1 * [w1, h1, w1, h1] + [0, h0, 0, h0]

        h2, w2, d2 = data_list[2]["img"].shape
        label2 = data_list[2]["label"]
        boxes2 = data_list[2]["boxes"]
        if len(boxes2) > 0:
            boxes2 = boxes2 * [w2, h2, w2, h2] + [w0, 0, w0, 0]

        h3, w3, d3 = data_list[3]["img"].shape
        label3 = data_list[3]["label"]
        boxes3 = data_list[3]["boxes"]
        if len(boxes3) > 0:
            boxes3 = boxes3 * [w3, h3, w3, h3] + [w0, h0, w0, h0]

        dst_bboxes = [d for d in [dst_bboxes, boxes1, boxes2, boxes3] if len(d) > 0]
        dst_labels = [d for d in [dst_labels, label1, label2, label3] if len(d) > 0]
        if len(dst_bboxes) > 0 and len(dst_labels) > 0:
            dst_bboxes = np.vstack(dst_bboxes)
            dst_labels = np.hstack(dst_labels)
            h, w, d = img_new.shape
            dst_bboxes = dst_bboxes / [w, h, w, h]
        else:
            dst_bboxes = np.asarray([])
            dst_labels = np.asarray([])
        assert len(dst_bboxes) == len(dst_labels)
        if out_size:
            img_new = cv2.resize(img_new, dsize=tuple(out_size))
        data = {"img": img_new, "boxes": dst_bboxes, "label": dst_labels}
        return data

    @staticmethod
    def mosaic_for3samples(data_list, out_size=None):
        """
        需要三张图像进行拼接
        :param data_list:[data]--> data = {"img": img, "boxes": bboxes, "label": label}
        :param out_size: None or [resize-height,resize-width]
        :return:
        """
        assert len(data_list) == 3
        # 垂直方向拼接
        img_0 = data_list[0]["img"].copy()
        img_12 = np.vstack((data_list[1]["img"].copy(), data_list[2]["img"].copy()))
        # 水平方向拼接
        h12, w12, d12 = img_12.shape
        img_0 = cv2.resize(img_0, dsize=tuple([h12, h12]))
        img_new = np.hstack((img_0, img_12))

        h0, w0, d0 = img_0.shape
        dst_labels = data_list[0]["label"]
        dst_bboxes = data_list[0]["boxes"]
        if len(dst_bboxes) > 0:
            dst_bboxes = dst_bboxes * [w0, h0, w0, h0]
        # 右上角
        h1, w1, d1 = data_list[1]["img"].shape
        label1 = data_list[1]["label"]
        boxes1 = data_list[1]["boxes"]
        if len(boxes1) > 0:
            boxes1 = boxes1 * [w1, h1, w1, h1] + [w0, 0, w0, 0]
        # 右下角
        h2, w2, d2 = data_list[2]["img"].shape
        label2 = data_list[2]["label"]
        boxes2 = data_list[2]["boxes"]
        if len(boxes2) > 0:
            boxes2 = boxes2 * [w2, h2, w2, h2] + [w0, int(h0 / 2), w0, int(h0 / 2)]

        dst_bboxes = [d for d in [dst_bboxes, boxes1, boxes2] if len(d) > 0]
        dst_labels = [d for d in [dst_labels, label1, label2] if len(d) > 0]
        if len(dst_bboxes) > 0 and len(dst_labels) > 0:
            dst_bboxes = np.vstack(dst_bboxes)
            dst_labels = np.hstack(dst_labels)
            # 随机水平翻转图片
            img_new, dst_bboxes, dst_labels = MosaicAugment.random_horizontal_flip(img_new, dst_bboxes, dst_labels)
            h, w, d = img_new.shape
            dst_bboxes = dst_bboxes / [w, h, w, h]
        else:
            dst_bboxes = np.asarray([])
            dst_labels = np.asarray([])

        assert len(dst_bboxes) == len(dst_labels)
        if out_size:
            img_new = cv2.resize(img_new, dsize=tuple(out_size))
        data = {"img": img_new, "boxes": dst_bboxes, "label": dst_labels}
        # print(dst_bboxes)
        return data

    @staticmethod
    def random_horizontal_flip(image, boxes, classes, p=0.5):
        """
        随机水平翻转图片
        :param image:
        :param boxes:
        :param classes:
        :param p:
        :return:
        """
        if random.random() < p:
            height, width, _ = image.shape
            image = image[:, ::-1, :]
            if len(boxes) > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, boxes, classes

    def __get_data(self, index):
        item = self._dataset[index]
        if len(item) == 3:
            img, bboxes, label = item
        elif len(item) == 2:
            img, bboxes_label = item
            bboxes = bboxes_label[:, :4]
            label = bboxes_label[:, -1:]
        else:
            raise Exception("Error{}".format(item))
        if self.from_torch:
            img = self.torch_to_numpy(img)
        data = {"img": img, "boxes": bboxes, "label": label}
        return data

    def torch_to_numpy(self, img):
        img = img.permute(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        img = np.asarray(img)
        return img

    def numpy_to_torch(self, img):
        import torch
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        # first image
        data0 = self.__get_data(index)
        if random.random() > self.p:
            img, bboxes, label = data0["img"], data0["boxes"], data0["label"]
            if len(bboxes) > 0:
                bboxes_labels = np.hstack((bboxes, label.reshape(-1, 1)))
            # return img, bboxes_labels
            if self.from_torch:
                img = self.numpy_to_torch(img)
            return img, bboxes, label

        # get other data
        if self.samples == 4:
            idx1 = np.random.choice(np.delete(np.arange(len(self)), index))
            idx2 = np.random.choice(np.delete(np.arange(len(self)), index))
            idx3 = np.random.choice(np.delete(np.arange(len(self)), index))
            data1 = self.__get_data(idx1)
            data2 = self.__get_data(idx2)
            data3 = self.__get_data(idx3)
            # append data
            data_list = [data0, data1, data2, data3]
            # mixup two images
            data = self.mosaic_for4samples(data_list, out_size=self.out_size)
        elif self.samples == 3:
            idx1 = np.random.choice(np.delete(np.arange(len(self)), index))
            idx2 = np.random.choice(np.delete(np.arange(len(self)), index))
            data1 = self.__get_data(idx1)
            data2 = self.__get_data(idx2)
            # append data
            data_list = [data0, data1, data2]
            # mixup two images
            data = self.mosaic_for3samples(data_list, out_size=self.out_size)
        else:
            raise Exception("Error:samples:{}".format(self.samples))
        img, bboxes, label = data["img"], data["boxes"], data["label"]
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if self.from_torch:
            img = self.numpy_to_torch(img)
        if len(bboxes) > 0:
            bboxes_labels = np.hstack((bboxes, label.reshape(-1, 1)))
        # return img, bboxes_labels
        return img, bboxes, label


def show_image(image_t, bboxes, labels, transpose=False, normal=False):
    """
    :param image_t:
    :param targets_t:
                bboxes = targets[idx][:, :4].data
                landms = targets[idx][:, 4:14].data
                labels = targets[idx][:, -1].data
    :return:
    """
    import numpy as np
    from utils import image_processing

    if not isinstance(image_t, np.ndarray):
        image_t = np.asarray(image_t, dtype=np.uint8)
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.asarray(bboxes)
        labels = np.asarray(labels)

    nums = len(bboxes)
    if transpose:
        image_t = image_processing.untranspose(image_t)
    h, w, _ = image_t.shape
    landms_scale = np.asarray([w, h] * 5)
    bboxes_scale = np.asarray([w, h] * 2)
    if normal:
        bboxes = bboxes * bboxes_scale
    # tmp_image = image_processing.untranspose(tmp_image)
    tmp_image = image_processing.convert_color_space(image_t, colorSpace="BGR")
    # tmp_image = image_processing.convert_color_space(image_t, colorSpace="BGR")
    image_processing.show_image_boxes("train", tmp_image, bboxes)


if __name__ == "__main__":
    from utils import image_processing, file_processing
    # from models.dataloader import data_transforms
    from models.transforms import data_transforms
    from models.dataloader import voc_parser

    # from modules.image_transforms import voc_parser

    print_info = True
    isshow = True
    # filename1 = '/home/dm/panjinquan3/dataset/finger/finger_v1/test.txt'
    # filename2 = '/home/dm/panjinquan3/dataset/finger/finger_v1/test.txt'
    filename1 = '/home/dm/panjinquan3/dataset/Character/gimage_v1/test.txt'
    filename2 = '/home/dm/panjinquan3/dataset/Character/gimage_v1/test.txt'
    filename1 = '/home/dm/panjinquan3/dataset/wider_face_add_lm_10_10/trainval.txt'
    # filename1 = "/home/dm/panjinquan3/dataset/MPII/test.txt"

    input_size = [640, 360]
    batch_size = 1
    shuffle = False
    # class_names = ["fingernail"]
    class_names = ["face", "person"]
    # anno_dir = data_root + '/Annotations'
    # class_names = ["face"]
    # class_names = None
    # anno_list = file_processing.get_files_list(anno_dir, postfix=["*.xml"])
    # image_id_list = file_processing.get_files_id(anno_list)
    target_transform = None
    train_transform = data_transforms.TrainTransform(size=input_size, mean=0, std=1.0)
    # train_transform = data_transforms.TestTransform(size=input_size, mean=0, std=1.0)
    voc = voc_parser.VOCDataset(filename1,
                                data_root=None,
                                class_names=class_names,
                                transform=train_transform,
                                target_transform=target_transform,
                                check=False)

    voc = MosaicAugment(voc, p=0.5, out_size=input_size, samples=3)
    print("have num:{}".format(len(voc)))
    for i in range(len(voc)):
        image, boxes, labels = voc.__getitem__(index=i)
        # boxes = boxes_labels[:, :4]
        # labels = boxes_labels[:, -1:].astype(np.int)
        height, width, depth = image.shape
        print("nums:{}".format(len(boxes)))
        # print(i, boxes)
        if isshow:
            show_image(image, boxes, labels, transpose=True, normal=True)
