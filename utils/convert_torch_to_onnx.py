"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys
import os
import torch.onnx
import demo

# import demo_for_landms
# class_names = ["BACKGROUND", "person"]
# class_names = ["BACKGROUND", "face", "person"]
class_names = ["BACKGROUND", "face"]


def convert_for_det_landms(freeze_header=True):
    """
    scores, boxes, landms
    :return:
    """
    det = demo.Detector(model_path,
                        net_type=net_type,
                        input_size=input_size,
                        priors_type=priors_type,
                        iou_threshold=iou_threshold,
                        prob_threshold=prob_threshold,
                        device=device)
    net = det.net
    flag = [net_type.lower() + str(width_mult), priors_type.lower(), input_size[0], input_size[1]]
    model_name = [str(f) for f in flag if f]
    if not freeze_header:
        onnx_path = os.path.join(os.path.dirname(model_path), "_".join(model_name) + ".onnx")
    else:
        onnx_path = os.path.join(os.path.dirname(model_path), "_".join(model_name) + "_freeze.onnx")
    # dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0]).to("cuda")
    torch.onnx.export(net,
                      dummy_input,
                      onnx_path,
                      verbose=False,
                      input_names=['input'],
                      output_names=['boxes', 'scores', 'ldmks'])
    print(onnx_path)


if __name__ == "__main__":
    args = demo.get_parser()
    print(args)
    # net_type = "RFB_landms"
    net_type = "RFB"
    # net_type = "mbv2"
    # priors_type = "person"
    # priors_type = "face_person"
    priors_type = "face"
    # priors_type = "person"
    # input_size = [480, 360]  # [W,H]
    input_size = [320, 320]
    # input_size = [960, 540]
    width_mult = 1.0
    device = "cuda:0"
    model_path="/home/dm/data3/FaceDetector/Face-Detector-1MB-with-landmark/work_space/RFB_landms/rfb1.0_face_320_320_wider_face_add_lm_10_10_preproc_20210608155931/model/best_model_rfb_199_loss6.4545.pth"

    # model_path = "/home/dm/panjinquan3/modes/face/RFB1.0_face_320_320_MPII_VOC2012_VOC2007_COCO_wider_face_add_lm_10_10_20200807140707/model/best_model_RFB_159_loss2.6988.pth"
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    convert_for_det_landms(freeze_header=False)
    # convert_for_det_landms(freeze_header=True)
