# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-05-25 09:31:17
"""
import cv2
import numpy as np


def affine_transform_point(point, trans):
    """
    输入原坐标点，进行仿射变换，获得变换后的坐标
    :param point: 输入坐标点 (x,y)
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return: 变换后的新坐标
    """
    new_point = np.array([point[0], point[1], 1.]).T
    new_point = np.dot(trans, new_point)  # 矩阵相乘
    return new_point[:2]


def affine_transform_points(points, trans):
    """
    输入原坐标点，进行仿射变换，获得变换后的坐标
    :param points: 输入坐标点集合，shape= (num_points,2)
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return: 变换后的新坐标
    """
    points = np.asarray(points)
    new_point = np.ones(shape=(len(points), 3))
    new_point[:, 0:2] = points[:, 0:2]
    new_point = np.dot(trans, new_point.T).T  # 矩阵相乘
    return new_point


def affine_transform_image(image, dsize, trans):
    """
    输入原始图像，进行仿射变换，获得变换后的图像
    :param image: 输入图像
    :param dsize: 输入目标图像大小
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return:
    """
    out_image = cv2.warpAffine(image, M=trans, dsize=tuple(dsize))
    return out_image


def get_kpts_affine_transform(kpts, kpts_ref, trans_type="estimate"):
    """
    估计最优的仿射变换矩阵
    :param kps: 实际关键点
    :param kpts_ref: 参考关键点
    :param trans_type:变换类型
    :return: 仿射变换矩阵
    """
    kpts = np.float32(kpts)
    kpts_ref = np.float32(kpts_ref)
    if trans_type == "estimate":
        # estimateAffine2D()可以用来估计最优的仿射变换矩阵
        trans, _ = cv2.estimateAffine2D(kpts, kpts_ref)
    elif trans_type == "affine":
        # 通过3点对应关系获仿射变换矩阵
        trans = cv2.getAffineTransform(kpts[0:3], kpts_ref[0:3])
    else:
        raise Exception("Error:{}".format(trans_type))
    return trans


def get_affine_transform(output_size,
                         center,
                         scale=[1.0, 1.0],
                         rot=0,
                         shift=[0, 0],
                         inv=False):
    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    center = np.asarray(center)
    scale = np.asarray(scale)
    output_size = np.asarray(output_size)
    shift = np.array(shift, dtype=np.float32)
    # scale_tmp = scale * 200.0
    scale_tmp = scale * output_size
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_reference_facial_points(square=True, isshow=False):
    """
    获得人脸参考关键点,目前支持两种输入的参考关键点,即[96, 112]和[112, 112]
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    ==================
    face_size_ref = [112, 112]
    kpts_ref = [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]

    ==================
    square = True, crop_size = (112, 112)
    square = False,crop_size = (96, 112),
    :param square: True is [112, 112] or False is [96, 112]
    :param isshow: True or False,是否显示
    :return:
    """
    # face size[96_112] reference facial points
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    kpts_ref = np.asarray(kpts_ref)  # kpts_ref_96_112
    # for output_size=[112, 112]
    if square:
        face_size_ref = np.array(face_size_ref)
        size_diff = max(face_size_ref) - face_size_ref
        kpts_ref += size_diff / 2
        face_size_ref += size_diff

    if isshow:
        from utils import image_processing
        tmp = np.zeros(shape=(face_size_ref[1], face_size_ref[0], 3), dtype=np.uint8)
        tmp = image_processing.draw_landmark(tmp, [kpts_ref], vis_id=True)
        cv2.imshow("kpts_ref", tmp)
        cv2.waitKey(0)
    return kpts_ref


def affine_transform_for_landmarks(image, landmarks, output_size=None):
    """
    对图像和landmarks关键点进行仿生变换
    :param image:输入RGB/BGR图像
    :param landmarks:人脸关键点landmarks(5个点)
    :param output_size:输出大小
    :return:
    """
    if not output_size:
        h, w, _ = image.shape
        output_size = [w, h]
    kpts_ref = get_reference_facial_points(square=True, isshow=False)
    alig_faces = []
    warped_landmarks = []
    for landmark in landmarks:
        trans = get_kpts_affine_transform(kpts=landmark, kpts_ref=kpts_ref, trans_type="estimate")
        trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
        alig_faces.append(trans_image)
        landmark = affine_transform_points(landmark, trans)
        warped_landmarks.append(landmark)
    return alig_faces, warped_landmarks


def rotate_points(points, centers, angle, height):
    """
    eg.:
    height, weight, d = image.shape
    point1 = [[300, 200],[50, 200]]
    point1 = np.asarray(point1)
    center = [[200, 200]]
    point3 = rotate_points(point1, center, angle=30, height=height)
    :param points:
    :param centers:
    :param angle:
    :param height:
    :return:
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if not isinstance(centers, np.ndarray):
        centers = np.asarray(centers)
    dst_points = points.copy()
    # 将图像坐标转换到平面坐标
    dst_points[:, 1] = height - dst_points[:, 1]
    centers[:, 1] = height - centers[:, 1]
    x = (dst_points[:, 0] - centers[:, 0]) * np.cos(np.pi / 180.0 * angle) - (
            dst_points[:, 1] - centers[:, 1]) * np.sin(np.pi / 180.0 * angle) + centers[:, 0]
    y = (dst_points[:, 0] - centers[:, 0]) * np.sin(np.pi / 180.0 * angle) + (
            dst_points[:, 1] - centers[:, 1]) * np.cos(np.pi / 180.0 * angle) + centers[:, 1]
    # 将平面坐标转换到图像坐标
    y = height - y
    dst_points[:, 0] = x
    dst_points[:, 1] = y
    return dst_points


def get_boxes2points(boxes):
    """
    :param boxes:
    :return:
    """
    # (num_boxes,4)=(num_boxes,xmin,ymin,xmax)
    num_boxes = len(boxes)
    xmin = boxes[:, 0:1]
    ymin = boxes[:, 1:2]
    xmax = boxes[:, 2:3]
    ymax = boxes[:, 3:4]
    t1 = np.hstack([xmin, ymin])
    t2 = np.hstack([xmin, ymax])
    t3 = np.hstack([xmax, ymin])
    t4 = np.hstack([xmax, ymax])
    # (num_boxes,8)=(num_boxes,xmin,ymin,xmax,ymax,xmin,ymax,xmax,ymin)
    points = np.hstack([t1, t4, t2, t3])
    # dst_boxes = dst_boxes[:, 0:4]
    points = points.reshape(num_boxes, -1, 2)  # (num_boxes,box_point(4),2)
    return points, num_boxes


def get_points2bboxes(points):
    """
    :param boxes:
    :return:
    """
    xmin = np.min(points[:, :, 0:1], axis=1)
    ymin = np.min(points[:, :, 1:2], axis=1)
    xmax = np.max(points[:, :, 0:1], axis=1)
    ymax = np.max(points[:, :, 1:2], axis=1)
    t1 = np.hstack([xmin, ymin])
    t2 = np.hstack([xmax, ymax])
    boxes = np.hstack([t1, t2])
    return boxes


def affine_transform_for_boxes(image, boxes, output_size=None, rot=0, inv=False):
    """
    对图像和boxes进行仿生变换
    :param image:输入RGB/BGR图像
    :param boxes:检测框
    :param output_size:输出大小
    :param rot:旋转角度，PS：旋转时，由于boxes只包含左上角和右下角的点，
               所以旋转时box的矩形框会变得比较扁
    :return:
    """
    boxes = np.asarray(boxes)
    h, w, _ = image.shape
    center = (int(w / 2), int(h / 2))
    scale = [1.0, 1.0]
    if not output_size:
        output_size = [w, h]
    trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
    trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
    points, num_boxes = get_boxes2points(boxes)
    for i in range(num_boxes):
        points[i, :] = affine_transform_points(points[i, :], trans)
    boxes = get_points2bboxes(points)
    return trans_image, boxes


class AffineTransform(object):
    @staticmethod
    def affine_transform_for_boxes(boxes, output_size, center, scale, rot=0, inv=False, **kwargs):
        """
        根据center, scale对bbox，或者kwargs进行仿生变换
        :param boxes: shape(num_boxes,(xmin,ymin,xmax,ymax))
        :param output_size:
        :param center: 旋转中心点
        :param scale: 缩放因子
        :param rot: 旋转角度
        :param inv: True: 仿生变换,False:反变换
        :param kwargs: {"key":shape(num_boxes,x1,y1,x2,y2,...,xn,yn)}
        :return:
        """
        trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
        points, nums = get_boxes2points(boxes)
        for i in range(nums):
            points[i, :] = affine_transform_points(points[i, :], trans)
        boxes = get_points2bboxes(points)
        if kwargs:
            for k in kwargs.keys():
                points = np.asarray(kwargs[k]).reshape(-1, 2)
                points = affine_transform_points(points, trans)
                kwargs[k] = points.reshape(nums, -1)
        return boxes, trans, kwargs

    @staticmethod
    def affine_transform_for_points(points, output_size, center, scale, rot=0, inv=False):
        trans = get_affine_transform(output_size, center, scale, rot=rot, shift=[0, 0], inv=inv)
        nums = len(points)
        for i in range(nums):
            points[i, :] = affine_transform_points(points[i, :], trans)
        return points, trans

    @staticmethod
    def affine_transform(image, boxes, output_size, rot=0, **kwargs):
        """
        对图像和boxes进行仿生变换
        :param image:
        :param boxes:
        :param output_size:
        :param rot: 旋转角度
        :return:
        """
        boxes = np.asarray(boxes)
        h, w, _ = image.shape
        center = (int(w / 2), int(h / 2))
        long_side = max([w / output_size[0], h / output_size[1]])
        scale = [long_side, long_side]
        boxes, trans, kwargs = AffineTransform.affine_transform_for_boxes(boxes,
                                                                          output_size,
                                                                          center,
                                                                          scale,
                                                                          rot=rot,
                                                                          inv=False,
                                                                          **kwargs)
        trans_image = affine_transform_image(image, dsize=output_size, trans=trans)
        return trans_image, boxes, center, scale, kwargs

    @staticmethod
    def inverse_affine_transform(boxes, output_size, center, scale, rot=0, **kwargs):
        """对图像和boxes进行反变换"""
        boxes, trans, kwargs = AffineTransform.affine_transform_for_boxes(boxes,
                                                                          output_size,
                                                                          center,
                                                                          scale,
                                                                          rot=rot,
                                                                          inv=True,
                                                                          **kwargs)
        return boxes, kwargs


def demo_for_landmarks():
    image_path = "face.jpg"
    image = image_processing.read_image(image_path)
    # face detection from MTCNN
    bbox_score = np.asarray([[69.48486808, 58.12609892, 173.92575279, 201.95947894, 0.99979943]])
    landmarks = np.asarray([[[103.97721, 119.6718],
                             [152.35837, 113.06249],
                             [136.67535, 142.62952],
                             [112.62607, 171.1305],
                             [154.60092, 165.12515]]])
    bboxes = bbox_score[:, :4]
    scores = bbox_score[:, 4:]
    image = image_processing.draw_landmark(image, landmarks)
    image_processing.cv_show_image("image", image, type="bgr")
    alig_faces, warped_landmarks = affine_transform_for_landmarks(image, landmarks, output_size=[256, 256])
    for i in range(len(alig_faces)):
        alig_face = image_processing.draw_landmark(alig_faces[i], [warped_landmarks[i]], color=(0, 255, 0))
        image_processing.cv_show_image("image", alig_face, type="bgr")


def demo_for_image_boxes():
    image_path = "test.jpg"
    bboxes = [[98, 42, 160, 100], [244, 260, 297, 332]]
    output_size = [320, 320]
    image = image_processing.read_image(image_path)
    image_processing.show_image_boxes("src", image, bboxes, waitKey=10)
    for i in range(360):
        trans_image, trans_boxes = affine_transform_for_boxes(image, bboxes, output_size=output_size, rot=i)
        print("shape:{},bboxes     ：{}".format(image.shape, bboxes))
        print("shape:{},trans_boxes：{}".format(trans_image.shape, trans_boxes))
        image_processing.show_image_boxes("trans", trans_image, trans_boxes, color=(0, 255, 0))


def demo_for_image_affine_transform():
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
    land_mark = np.asarray(land_mark).reshape(-1, 10)
    output_size = [256, 480]
    image = image_processing.read_image(image_path)
    image_processing.show_image_boxes("src", image, bboxes, waitKey=10)
    at = AffineTransform()
    for i in range(360):
        trans_image, trans_boxes, center, scale, kwargs = at.affine_transform(image,
                                                                              bboxes,
                                                                              output_size=output_size,
                                                                              rot=i,
                                                                              land_mark=land_mark)
        img = image.copy()
        image_boxes, kwargs = at.inverse_affine_transform(trans_boxes, output_size, center, scale, rot=i, **kwargs)
        points = kwargs["land_mark"].reshape(len(trans_boxes), -1, 2)
        print("shape:{},bboxes     ：{}".format(image.shape, bboxes))
        print("shape:{},trans_boxes：{}".format(trans_image.shape, trans_boxes))
        image_processing.show_image_boxes("trans", trans_image, trans_boxes, color=(0, 255, 0), waitKey=1)
        img = image_processing.draw_landmark(img, points, color=(0, 255, 0))
        image_processing.show_image_boxes("img", img, image_boxes, color=(0, 255, 0))


if __name__ == "__main__":
    from utils import image_processing

    # demo_for_landmarks()
    # demo_for_image_boxes()
    demo_for_image_affine_transform()
