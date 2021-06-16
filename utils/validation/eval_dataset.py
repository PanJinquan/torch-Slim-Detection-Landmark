import os
import numpy as np
import xmltodict
import cv2
import glob
from tqdm import tqdm


class VOCDataset:
    """
    Dataset for VOC data.
    """

    def __init__(self, data_root, test_file, class_names=None, colorSpace="BGR", keep_difficult=False, check=True):
        """
        :param data_root:
        :param test_file:
        :param class_names:
        :param colorSpace:
        :param keep_difficult:
        :param check:
        """
        self.class_names = class_names
        if not class_names:
            self.class_names = ['BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            # self.class_names = ('BACKGROUND', "body", "face")
            # self.class_names = ['BACKGROUND', "person"]

        if isinstance(self.class_names, list):
            self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        elif isinstance(self.class_names, dict):
            self.class_dict = self.class_names
        else:
            raise Exception("class_names:{}".format(self.class_names))

        self.data_root = data_root
        self.anno_dir = os.path.join(self.data_root, "Annotations")
        self.image_dir = os.path.join(self.data_root, "JPEGImages")

        self.colorSpace = colorSpace
        if not test_file:
            test_file = os.path.join(self.data_root, "ImageSets/Main/test.txt")
            # test_file = os.path.join(self.root, "test.txt")
        self.keep_difficult = keep_difficult
        self.ids = VOCDataset._read_image_ids(test_file)
        self.postfix = self.get_image_postfix(self.image_dir, self.ids)
        if check:
            self.ids = self.check_image_idx(self.ids)

    def get_image_postfix(self, image_dir, image_id):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." in image_id[0]:
            postfix = ""
        else:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            postfix = os.path.basename(image_list[0]).split(".")[1]
        return postfix

    def get_image_anno_file(self, image_dir, anno_dir, image_id: str, img_postfix=None):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :param img_postfix:
        :return:
        """
        if not img_postfix and "." in image_id:
            image_id, img_postfix = image_id.split(".")
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.xml".format(image_id))
        return image_file, annotation_file

    def check_image_idx(self, image_idx):
        """
        check image idx
        :param image_idx:
        :return:
        """
        dst_image_idx = []
        # image_idx = image_idx[:20]
        print("checking images,please wait a minute")
        for image_id in tqdm(image_idx):
            image_file, annotation_file = self.get_image_anno_file(self.image_dir, self.anno_dir, image_id,
                                                                   self.postfix)
            if (not os.path.exists(image_file)) or (not os.path.exists(annotation_file)):
                continue
            boxes, labels, landms, is_difficult = self._get_annotation(annotation_file, class_dict=self.class_dict)
            if len(boxes) == 0:
                continue
            dst_image_idx.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_idx), len(dst_image_idx)))
        return dst_image_idx

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_file, annotation_file = self.get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        boxes, labels, landms, is_difficult = self._get_annotation(annotation_file, class_dict=self.class_dict)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_file)
        # # 测试读取的图片
        # print(" boxes.shape:{},boxes:{}".format(boxes.shape,boxes))
        # print("labels.shape:{},labels:{}".format(labels.shape,labels))
        # image_processing.show_image_boxes("image",image,boxes)
        return image, boxes, labels

    def get_image(self, index):
        """
        get image
        :param index:
        :return:
        """
        image_id = self.ids[index]
        image_file, annotation_file = self.get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        image = self._read_image(image_file)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        image_file, annotation_file = self.get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        boxes, labels, landms, is_difficult = self._get_annotation(annotation_file, class_dict=self.class_dict)
        return image_id, (boxes, labels, is_difficult)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def read_xml2json(self, xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def _get_annotation(self, xml_file, class_dict=None):
        """
        :param xml_file:
        :param class_dict: class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        :return:
        """
        content = self.read_xml2json(xml_file)
        try:
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])
            filename = annotation["filename"]
            objects = annotation["object"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            exit(0)

        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            class_name = object["name"]
            if class_name in self.class_dict:
                difficult = int(object["difficult"])
                xmin = float(object["bndbox"]["xmin"])
                xmax = float(object["bndbox"]["xmax"])
                ymin = float(object["bndbox"]["ymin"])
                ymax = float(object["bndbox"]["ymax"])
                # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                bbox = [xmin, ymin, xmax, ymax]
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                if w <= 0 or h <= 0:
                    print("illegal annotation:{}".format(xml_file))
                    break
                # get person keypoints ,if exist
                if 'lm' in object:
                    lm = object["lm"]
                    landms = [lm["x1"], lm["y1"], lm["x2"], lm["y2"], lm["x3"],
                              lm["y3"], lm["x4"], lm["y4"], lm["x5"], lm["y5"]]
                else:
                    landms = [-1.0] * 5 * 2
                kp_bbox = {}
                kp_bbox["landms"] = landms
                kp_bbox["bbox"] = bbox
                kp_bbox["difficult"] = difficult
                if class_dict:
                    kp_bbox["class_name"] = class_dict[class_name]
                else:
                    kp_bbox["class_name"] = class_name
                objects_list.append(kp_bbox)
        # annotation_dict = {}
        # annotation_dict["image"] = filename
        # annotation_dict["object"] = objects_list
        boxes, labels, landms, is_difficult = self.get_object(objects_list)
        return boxes, labels, landms, is_difficult

    def get_object(self, objects_list):
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
            labels.append(item['class_name'])
            landms.append(item['landms'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        landms = np.array(landms, dtype=np.float32)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, landms, is_difficult

    def _read_image(self, image_file):
        image = cv2.imread(str(image_file))  # BGR
        if self.colorSpace == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
