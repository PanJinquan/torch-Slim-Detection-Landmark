# -*-coding: utf-8 -*-
"""
    @Project: Pytorch-SSD
    @File   : ssd_detector.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-11 14:55:55
"""
import sys
import os

sys.path.append(os.getcwd())
import cv2
import numpy as np
import demo
import demo_for_landm
from tqdm import tqdm
from utils import file_processing, image_processing
from utils.validation import eval_dataset, measurements
from models.dataloader.parser_voc import VOCDataset


class Validation(demo.Detector):
    # class Validation(demo_for_landm.Detector):
    def __init__(self,
                 model_path,
                 net_type,
                 input_size,
                 class_names,
                 priors_type,
                 prob_threshold=0.01,
                 iou_threshold=0.5,
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
                                         self.priors_type,
                                         self.input_size,
                                         prob_threshold=self.prob_threshold,
                                         iou_threshold=self.iou_threshold,
                                         device=self.device)

    def get_ground_truth(self, filename, label_dir=None):
        """
        :param filename:
        :param label_dir:
        :return:
        """
        if label_dir:
            filename = os.path.join(label_dir, filename)
        label_bbox_list = file_processing.read_data(filename, convertNum=True)
        label_list, rect_list = file_processing.split_list(label_bbox_list, split_index=1)
        label_list = [l[0] for l in label_list]
        return label_list, rect_list

    def batch_detect_image(self,
                           image_list,
                           save_dir=None,
                           label_dir=None,
                           show=False):
        """

        :param image_list:
        :param prob_threshold:
        :param save_dir:
        :param label_dir:
        :param show:
        :return:
        """
        print("have image:{}".format(len(image_list)))
        for image_path in tqdm(image_list):
            filename = os.path.basename(image_path)[:-len(".jpg")] + ".txt"
            gt_path = os.path.join(label_dir, filename)
            if not os.path.exists(image_path):
                print("no path:{}".format(image_path))
                continue
            if not os.path.exists(gt_path):
                print("no path:{}".format(gt_path))
                continue
            orig_image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = self.predict(rgb_image, isshow=False)
            label_names = file_processing.decode_label(labels, name_table=self.class_names)
            info = image_processing.combile_label_prob(label_names, probs)
            dt_filename = file_processing.create_dir(save_dir, "dt_result", filename=filename)
            # rects = image_processing.bboxes2rects(boxes)
            self.save_labels_probs(dt_filename, boxes, labels, probs)
            if label_dir:
                true_label_list, true_rect_list = self.get_ground_truth(gt_path, label_dir=None)
                true_bbox_list = image_processing.rects2bboxes(true_rect_list)
                gt_filename = file_processing.create_dir(save_dir, "gt_result", filename=filename)
                self.save_labels_probs(gt_filename, true_bbox_list, true_label_list, probs=None)
                if show:
                    rgb_image = image_processing.show_image_bboxes_text("image",
                                                                        rgb_image,
                                                                        true_bbox_list,
                                                                        true_label_list,
                                                                        color=(0, 0, 255),
                                                                        waitKey=0)

            if show:
                image_processing.show_image_detection_bboxes("image", rgb_image, boxes, probs, label_names,
                                                             color=(255, 0, 0))

    def save_labels_probs(self, filename, boxes, labels, probs=None):
        """
        :param filename:
        :param boxes:
        :param labels:
        :param probs:
        :return:
        """
        if probs is None:
            probs = [""] * len(labels)
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        content_list = []
        for boxe, label, prob in zip(boxes, labels, probs):
            if label == 2:
                label = 1
            if prob == "" or prob is None:
                data = [label] + list(boxe)
            else:
                data = [label] + [prob] + list(boxe)
            content_list.append(data)
        if filename:
            file_processing.write_data(filename, content_list)

    def eval_image(self, gt_label_dir, image_dir, dt_label_dir, isshow=False):
        """
        min_score_thresh = 0.01
        :param gt_label_dir:
        :param image_dir:
        :param dt_label_dir:
        :return:
        """
        if os.path.exists(dt_label_dir):
            file_processing.remove_dir(dt_label_dir)
        file_processing.create_dir(dt_label_dir)
        gt_label_list = file_processing.get_files_list(gt_label_dir, postfix=["*.txt"])
        for file_path in gt_label_list:
            filename = os.path.basename(file_path)
            image_path = os.path.join(image_dir, filename[:-len("txt")] + "jpg")
            print(image_path)
            image = image_processing.read_image(image_path)
            bboxes, classes, scores = self.predict(image, isshow=isshow)
            dt_path = os.path.join(dt_label_dir, filename)
            self.save_detect_result(image, dt_path, bboxes, scores, classes, isshow=isshow)

    def save_detect_result(self, image, filename, bboxes, scores, classes, isshow=False):
        """
        :param image:
        :param filename:
        :param bboxes:
        :param scores:
        :param classes:
        :param isshow:
        :return:
        """
        # classes[classes == 0] = 1
        rects = image_processing.bboxes2rects(bboxes)
        if isshow:
            image = image_processing.show_image_detection_rects("Det", image, rects, scores, classes)
        content_list = []
        for c, s, r in zip(classes, scores, rects):
            # image = image_processing.show_image_detection_rects("Det", image, [r], [s], [c])
            r = [str(i) for i in r]
            line = "{} {} ".format(str(c), str(s)) + " ".join(r)
            content_list.append(line)
        file_processing.write_list_data(filename, content_list)

    def save_dt_gt_result(self, dataset, save_dir=None, show=True):
        """
        :param gt_results:
        :param dt_results:  (num_bboxes, 7)=[indexes,labels,probs,boxes]
        :return:
        """
        from utils.validation import bboxes_match
        if not save_dir:
            save_dir = os.path.join(dataset.data_root, "gt_dt_result")
            file_processing.create_dir(save_dir)

        for i in tqdm(range(len(dataset))):
            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation
            rgb_image = dataset.get_image(i)
            dt_boxes, labels, probs = self.predict(rgb_image, isshow=False)
            pred_bboxes, true_bboxes = bboxes_match.MatchingBBoxes.bboxes_matching(dt_boxes, gt_boxes)
            # 如果IOU匹配后，bboxes的个数不一致，则说明出现漏检或者误检测
            if show and len(pred_bboxes) < len(dt_boxes):
                print("image_id：{}\nboxes:{}\nlabels:{}\nprobs:{}".format(image_id, dt_boxes, labels, probs))
                image, gt_boxes = image_processing.resize_image_bboxes(rgb_image, resize_width=500, bboxes=gt_boxes)
                image, dt_boxes = image_processing.resize_image_bboxes(rgb_image, resize_width=500, bboxes=dt_boxes)
                image = image_processing.show_image_boxes("gt_boxes", image, gt_boxes, color=(0, 255, 0), waitKey=3)
                # image_processing.show_image_boxes("image", image, boxes, color=(255, 0, 0), waitKey=0)
                image = image_processing.show_image_detection_bboxes("image", image, dt_boxes, probs, labels,
                                                                     color=(255, 0, 0), waitKey=30)
                image_file = os.path.join(save_dir, image_id)
                image_processing.save_image(image_file, image)

    def metrics_for_text(self, image_dir, filename, save_dir, label_dir=None, show=False):
        """
        :param image_dir: JPEGImages directory
        :param filename: val.txt or None
        :param prob_threshold:
        :param save_dir: save dets result directory
        :param label_dir: path to label directory
        :param show:
        :return:
        """
        if filename:
            image_id = file_processing.read_data(filename, convertNum=False)
            image_list = [os.path.join(image_dir, str(id[0]) + ".jpg") for id in image_id]
        else:
            image_list = file_processing.get_images_list(image_dir, postfix=["*.jpg"])
        self.batch_detect_image(image_list, save_dir, label_dir=label_dir, show=show)

    def metrics_for_voc(self, dataroot, test_file):
        """
        :param dataroot: VOC Dataset Root
        :return:
        """
        # test_file = os.path.join(dataroot, "test.txt")
        eval_results = os.path.join(dataroot, "eval_results")
        dataset = eval_dataset.VOCDataset(dataroot,
                                          test_file=test_file,
                                          colorSpace="RGB",
                                          class_names=self.class_names)
        assert self.class_names == dataset.class_names
        # self.num_true_cases, self.all_gb_boxes, self.all_difficult_cases = self.get_gt_result(dataset)
        # self.save_dt_gt_result(dataset)

        if isinstance(self.class_names, dict):
            class_names = list(set(self.class_names.values()))
        else:
            class_names = self.class_names
        gt_results = self.get_gt_result(dataset)
        dt_results = self.get_dt_result(dataset)
        if not os.path.exists(eval_results):
            os.makedirs(eval_results)
        self.save_dt_result(dt_results, dataset, class_names, eval_results)
        self.calculate_map(gt_results, class_names, eval_results)

    def get_dt_result(self, dataset):
        """
        :param dataset:
        :return:
        """
        results = []
        for i in tqdm(range(len(dataset))):
            image = dataset.get_image(i)
            dets, labels = self.predict(image, isshow=False)
            boxes = dets[:, 0:4]
            probs = dets[:, 4:5]
            indexes = np.ones(labels.shape[0], dtype=np.float32) * i
            results.append(np.concatenate([indexes.reshape(-1, 1),
                                           labels.reshape(-1, 1),
                                           probs.reshape(-1, 1),
                                           boxes  # matlab's indexes start from 1
                                           ], axis=1))
        results = np.concatenate(results)
        return results

    def save_dt_result(self, dt_results, dataset, class_names, eval_results="eval_results"):
        """
        :param dt_results:
        :param dataset:
        :param class_names:
        :param eval_results:
        :return:
        """
        for class_index, class_name in enumerate(class_names):
            if class_index == 0: continue  # ignore background
            prediction_path = os.path.join(eval_results, f"det_test_{class_name}.txt")
            with open(prediction_path, "w") as f:
                sub = dt_results[dt_results[:, 1] == class_index, :]
                for i in range(sub.shape[0]):
                    prob_box = sub[i, 2:]
                    image_id = dataset.ids[int(sub[i, 0])]
                    print(image_id + " " + " ".join([str(v) for v in prob_box]), file=f)

    def calculate_map(self, gt_results, class_names, eval_results="eval_results"):
        if not os.path.exists(eval_results):
            os.makedirs(eval_results)
        aps = []
        ap_logs = eval_results.split(os.sep)[-2] + ": "
        print("Average Precision Per-class:")
        results = {}
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue
            if not class_index in gt_results["num_true_cases"]:
                print("gt_results no class:{}".format(class_name))
                continue
            prediction_path = os.path.join(eval_results, f"det_test_{class_name}.txt")
            result = measurements.compute_average_precision_per_class(
                gt_results["num_true_cases"][class_index],
                gt_results["gt_boxes"][class_index],
                gt_results["difficult_cases"][class_index],
                prediction_path,
                self.iou_threshold,
                use_2007_metric=False,
                name=class_name
            )
            results[class_name] = result
            ap = result["ap"]
            aps.append(ap)
            ap_log = "{}: {:3.5f}".format(class_name, ap)
            ap_logs += ap_log
            print(ap_log)
        print("\nAverage Precision Across All Classes: mAP:{:3.5f}".format(sum(aps) / len(aps)))
        ap_logs += " mAP:{:3.5f}".format(sum(aps) / len(aps))
        filename = os.path.join(os.path.dirname(self.model_path), ap_logs + ".txt")
        writer = file_processing.WriterTXT(filename=filename)
        writer.write_line_str(ap_logs)
        print(ap_logs)
        print("==" * 20)
        measurements.plot_map_recall_precision(results)

    def get_gt_result(self, dataset):
        num_true_cases = {}
        all_gt_boxes = {}
        all_difficult_cases = {}
        for i in range(len(dataset)):
            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation
            # gt_boxes = torch.from_numpy(gt_boxes)
            for i, difficult in enumerate(is_difficult):
                class_index = int(classes[i])
                gt_box = gt_boxes[i]
                if not difficult:
                    num_true_cases[class_index] = num_true_cases.get(class_index, 0) + 1

                if class_index not in all_gt_boxes:
                    all_gt_boxes[class_index] = {}
                if image_id not in all_gt_boxes[class_index]:
                    all_gt_boxes[class_index][image_id] = []
                all_gt_boxes[class_index][image_id].append(gt_box)
                if class_index not in all_difficult_cases:
                    all_difficult_cases[class_index] = {}
                if image_id not in all_difficult_cases[class_index]:
                    all_difficult_cases[class_index][image_id] = []
                all_difficult_cases[class_index][image_id].append(difficult)

        for class_index in all_gt_boxes:
            for image_id in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = np.stack(all_gt_boxes[class_index][image_id])
        for class_index in all_difficult_cases:
            for image_id in all_difficult_cases[class_index]:
                all_gt_boxes[class_index][image_id] = np.asarray(all_gt_boxes[class_index][image_id])
        gt_result = {}
        gt_result["num_true_cases"] = num_true_cases
        gt_result["gt_boxes"] = all_gt_boxes
        gt_result["difficult_cases"] = all_difficult_cases
        return gt_result


def validation_report():
    '''
    Average Precision Across All Classes:0.9093616438835752
    best:model_path="pretrained/RFB_person_640_360_MPII_VOC2012_VOC2007_VOC_20200624105257/model/RFB-Epoch-197-Loss-1.8028894911951094.pth"
    :return:
    '''
    prob_threshold = 0.05
    iou_threshold = 0.5
    model_path = "/home/dm/data3/FaceDetector/Face-Detector-1MB-with-landmark/work_space/RFB_face_person/RFB1.0_face_person_320_320_MPII_v2_20210615210557/model/best_model_RFB_150_loss2.8792.pth"
    # priors_type = "face"
    priors_type = "face_person"
    net_type = "RFB"
    # net_type = "RFB_landms"
    # net_type = "mbv2"
    class_names = ["BACKGROUND", "face", "person"]
    # class_names = ["BACKGROUND", "face"]
    input_size = [320, 320]
    dataroot = "/home/dm/data3/dataset/face_person/MPII"
    # dataroot = "/data3/panjinquan/dataset/face_person/MPII"
    image_dir = os.path.join(dataroot, "JPEGImages")
    filename = os.path.join(dataroot, "test.txt")
    save_dir = os.path.join(dataroot, "eval")
    label_dir = os.path.join(dataroot, "labels")
    det = Validation(model_path,
                     net_type=net_type,
                     input_size=input_size,
                     class_names=class_names,
                     priors_type=priors_type,
                     iou_threshold=iou_threshold,
                     prob_threshold=prob_threshold,
                     device=device)
    det.metrics_for_voc(dataroot, filename)
    # det.metrics_for_text(image_dir,
    #                      filename,
    #                      save_dir=save_dir,
    #                      label_dir=label_dir,
    #                      show=False)
    print(model_path)
    print(input_size)
    print(dataroot)


if __name__ == "__main__":
    args = demo.get_parser()
    device = args.device
    validation_report()
