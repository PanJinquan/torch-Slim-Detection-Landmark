import torch
import torch.nn as nn
import torch.nn.functional as F

from models.anchor_utils import box_utils


class MultiBoxLossLandm(nn.Module):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLossLandm, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, predicted_landms, gt_labels, gt_locations, gt_landms):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            gt_labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        # print(gt_labels[:,0])
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        class_loss = F.cross_entropy(confidence.reshape(-1, num_classes), gt_labels[mask], reduction='sum')

        pos_mask = gt_labels > 0
        # cal locations loss
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        loc_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')  # smooth_l1_loss
        # cal landmark loss
        predicted_landms = predicted_landms[pos_mask, :].reshape(-1, 10)
        gt_landms = gt_landms[pos_mask, :].reshape(-1, 10)
        landms_loss = F.smooth_l1_loss(predicted_landms, gt_landms, reduction='sum')  # smooth_l1_loss

        # smooth_l1_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')  #l2 loss
        num_pos = gt_locations.size(0)
        return loc_loss / num_pos, class_loss / num_pos, landms_loss / num_pos
