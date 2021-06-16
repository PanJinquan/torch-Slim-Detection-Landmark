import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import debug
import torch
import xmltodict

logging = debug.get_logger("train")


class VOCLandmarkDataset:

    def __init__(self,
                 filename,
                 data_root=None,
                 class_names=None,
                 transform=None,
                 target_transform=None,
                 keep_difficult=False,
                 check=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        if not data_root:
            data_root = os.path.dirname(filename)
        self.data_root = data_root
        self.transform = transform
        self.keep_difficult = keep_difficult
        logging.info("read filename   from :{}".format(filename))
        self.class_names, self.class_dict = self.get_classes(class_names)
        self.num_classes = max(list(self.class_dict.values())) + 1
        self.ids = VOCLandmarkDataset._read_image_ids(filename)
        if check:
            self.ids = self.check_ids(self.ids)

    def get_classes(self, class_names):
        """
        read class(labels) sets
        :param label_file:
        :return:
        """
        # if the labels file exists, read in the class names
        if not class_names:
            class_names = ('BACKGROUND', 'face')
            # class_names = ('BACKGROUND', 'face', "person")
            logging.info("using default classes:{}".format(class_names))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        logging.info("class_dict: {}".format(class_dict))
        return class_names, class_dict

    def check_ids(self, ids):
        dst_ids = []
        # ids = ids[:100]
        logging.info("checking images,please wait a minute")
        for image_id in tqdm(ids):
            annotation_file = os.path.join(self.data_root, f"Annotations/{image_id}.xml")
            image_file = os.path.join(self.data_root, f"JPEGImages/{image_id}.jpg")
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            # boxes, labels, is_difficult = self._get_annotation(image_id)
            boxes, labels, landms, is_difficult = self._get_annotation(annotation_file, class_dict=self.class_dict)
            if not self.keep_difficult:
                boxes = boxes[is_difficult == 0]
                labels = labels[is_difficult == 0]
            if len(boxes) == 0 or len(labels) == 0:
                continue
            dst_ids.append(image_id)
        logging.info("have nums image:{},legal image:{}".format(len(ids), len(dst_ids)))
        return dst_ids

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
        target=[nums,15]
            bboxes = targets[idx][:, :4]
            landms = targets[idx][:, 4:14]
            labels = targets[idx][:, -1]
        if landms=-1,transform will set landms=0, labels=-1
        :param index:
        :return:
        """
        image_id = self.ids[index]
        annotation_file = os.path.join(self.data_root, f"Annotations/{image_id}.xml")
        boxes, labels, landms, is_difficult = self._get_annotation(annotation_file, class_dict=self.class_dict)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
            landms = landms[is_difficult == 0]
        img = self._read_image(image_id)
        target = self.convert_target(boxes, labels, landms)
        if self.transform:
            # self.show_image(img, target,normal=False, transpose=False)
            img, target = self.transform(img, target)  # <class 'tuple'>: (3, 360, 480),<class 'tuple'>: (1, 15)
            # show_targets_image(img, target, normal=True, transpose=True)
        return torch.from_numpy(img), target

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(filename):
        ids = []
        with open(filename) as f:
            for line in f:
                line = line.rstrip().split(" ")[0]
                ids.append(line.rstrip())
        return ids

    def read_xml2json(self, xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file) as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def _get_annotation(self, xml_file, class_dict=None):
        """
        :param xml_file:
        :param class_dict: class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        :return:
        """
        content = self.read_xml2json(xml_file)
        annotation = content["annotation"]
        # get image shape
        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])
        depth = int(annotation["size"]["depth"])
        filename = annotation["filename"]
        objects_list = []
        objects = annotation["object"]
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

    def _read_image(self, image_id):
        """

        :param image_id:
        :return: BGR Image
        """
        image_file = os.path.join(self.data_root, f"JPEGImages/{image_id}.jpg")
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



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
    # from models.transforms.data_transforms import TrainLandmsTransform, TestTransform
    from models.dataloader import WiderFaceDetection, detection_collate, preproc, val_preproc
    # from models.dataloader import voc_parser, collate_fun
    import torch.utils.data as data

    image_mean = np.array([0, 0, 0]),
    image_std = 255.0
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    input_size = [400, 400]
    target_transform = None
    check = False
    rgb_mean = [0, 0, 0]
    rgb_std = [255, 255, 255]
    batch_size = 1
    shuffle = False
    # filename = "/home/dm/data3/dataset/face_person/FDDB/trainval.txt"
    filename = "/home/dm/data3/dataset/face_person/wider_face_add_lm_10_10/test.txt"
    class_names = {'BACKGROUND': 0, 'face': 1}
    transform = preproc(input_size, rgb_mean, rgb_std)
    transform = val_preproc(input_size, rgb_mean, rgb_std)
    # transform = TrainLandmsTransform(input_size, rgb_mean, rgb_std)
    # transform = TestLandmsTransform(input_size, rgb_mean, rgb_std)
    train_dataset = VOCLandmarkDataset(filename,
                                       data_root=None,
                                       class_names=class_names,
                                       transform=transform,
                                       keep_difficult=False,
                                       check=False)
    # train_dataset = ConcatDataset([train_dataset, train_dataset])
    train_loader = data.DataLoader(train_dataset,
                                   batch_size,
                                   num_workers=0,
                                   shuffle=False,
                                   collate_fn=detection_collate)
    for i, inputs in enumerate(train_loader):
        img, target = inputs
        show_targets_image(img[0], target[0], normal=True, transpose=True)
        # show_landmark_image(img, target, normal=True, transpose=True)