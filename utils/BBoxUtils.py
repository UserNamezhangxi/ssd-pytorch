import numpy as np
import torch
from torch import nn
from torchvision.ops import nms

class BBoxUtils(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes


    def _decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        #  先验框d 网络输出 l
        # 真实框 cx = lcx * dw + dcx
        pred_cx = mbox_loc[:, 0] * anchor_width * variances[0] + anchor_center_x
        # 真实框 cy = lcy * dh + dcy
        pred_cy = mbox_loc[:, 1] * anchor_height * variances[0] + anchor_center_y
        # 真实框 w = dw * exp(lw)
        pred_w = anchor_width * torch.exp(mbox_loc[:, 2] * variances[1])
        # 真实框 h = dh * exp(lh)
        pred_h = anchor_height * torch.exp(mbox_loc[:, 3] * variances[1])

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = pred_cx - pred_w * 0.5
        decode_bbox_ymin = pred_cy - pred_h * 0.5
        decode_bbox_xmax = pred_cx + pred_w * 0.5
        decode_bbox_ymax = pred_cy + pred_h * 0.5

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, anchors, input_shape, image_shape, letterbox_image, variances=[0.1, 0.2],
                   nms_iou=0.3, confidence=0.5):
        # 坐标位置回归数据
        mbox_loc = predictions[0]

        # 种类置信度
        mbox_conf = nn.Softmax(-1)(predictions[1])

        results = []

        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self._decode_boxes(mbox_loc[0], anchors, variances)
            for c in range(1, self.num_classes):
                c_confs = mbox_conf[i, :, c] # 8732 个框框，在每一个分类上的概率
                c_confs_m = c_confs > confidence # 大于置信度的 标记为有效的分类

                if len(c_confs[c_confs_m]) > 0:
                    # 再根据符合分类置信度的mask 取出 对应的 回归框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )

                    # -----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    # -----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c-1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c-1) * torch.ones((len(keep), 1))

                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()

                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) * 0.5, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results








