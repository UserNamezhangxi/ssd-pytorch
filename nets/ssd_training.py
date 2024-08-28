import torch
from torch import nn


class MutilBoxLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 1.0
        self.negatives_for_hard = torch.FloatTensor([100])[0]

    def forward(self, y_true, y_pred):
        # y_true [batch_size,8732, 4 + number_class + 1] # number_class = 21 包含背景，1 表示先验框是否有对应的物体
        # y_pred [batch_size,8732, 4 + number_class]

        y_pred = torch.cat([y_pred[0], nn.Softmax(dim=-1)(y_pred[1])], dim=-1)

        # 分类的loss
        classes_loss = self._class_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])

        # 边界框loss
        loc_loss = self._location_loss(y_true[:, :, :4], y_pred[:, :, :4])

        # 所有正样本的损失
        pos_classes_loss = torch.sum(classes_loss * y_true[:, :, -1], 1)
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1], 1)

        # 每一个图的正样本个数
        num_pos = torch.sum(y_true[:, :, -1], -1)

        # 根据正样本计算负样本个数 3倍于正样本个数
        num_neg = torch.min(3 * num_pos, y_true.size()[1] - num_pos)

        # 如果说这个batch里一个正样本都没有的话，就默认给100个负样本个数
        has_min_nag = torch.sum(num_neg > 0)
        num_neg = torch.sum(num_neg) if has_min_nag > 0 else self.negatives_for_hard

        # 5 是从第一个分类不是背景开始，到最后一个分类 判断除过正样本之外的样本（负样本）的概率比较大（接近正样本概率的样本）
        every_pred_box_is_hard_to_classes_P = torch.sum(y_pred[:, :, 5:25], dim=2)
        every_pred_box_is_hard_to_classes_P = (every_pred_box_is_hard_to_classes_P * (1 - y_true[:, :, -1])).view(-1)

        _, indices = torch.topk(every_pred_box_is_hard_to_classes_P, k=int(num_neg.cpu().numpy().tolist()))
        neg_conf_loss = torch.gather(classes_loss.view([-1]), 0, indices)

        # 归一化
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(pos_classes_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss = total_loss / torch.sum(num_pos)
        if total_loss == torch.nan:
            print(" =part1 {},part2 {}, part3 {}".format(pos_classes_loss, neg_conf_loss, self.alpha * pos_loc_loss))
            print(" part1 {},part2 {}, part3 {}".format(torch.sum(pos_classes_loss), torch.sum(neg_conf_loss), torch.sum(self.alpha * pos_loc_loss)))
        return total_loss

    # 分类损失
    def _class_loss(self, y_true, y_pred):
        y_true = torch.clamp(y_true, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred), -1)
        return softmax_loss

    # 邊界框回归损失 SmoothL1 loss
    def _location_loss(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        small = 0.5 * diff ** 2
        large = diff - 0.5
        l1_loss = torch.where(diff < 1, small, large)
        return torch.sum(l1_loss, -1)
