import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.utils import cvtColor, preprocess_input,get_classes
from utils.anchors import get_anchors

class SSDDataset(Dataset):
    def __init__(self, annotations_lines, input_shape, num_classes, anchors, random, overlap_threshold=0.5):
        super(SSDDataset, self).__init__()
        self.annotations_lines = annotations_lines
        self.input_shape = input_shape
        self.random = random
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.iou_threshold = overlap_threshold
        self.anchors = anchors


    def __len__(self):
        return len(self.annotations_lines)

    def __getitem__(self, index):
        image, box = self.get_item_data(self.annotations_lines[index], self.input_shape, random=self.random)
        image_data = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
           boxes = np.array(box[:, :4], dtype=np.float32)
           # 进行归一化，调整到0-1之间
           boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
           boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
           # 对真实框的种类进行one hot处理
           one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
           # 拼接成为 [x1,y1,x2,y2,(one hot class)]
           box = np.concatenate([boxes, one_hot_label], axis=-1)
        box = self.assign_boxes(box)

        return np.array(image_data, np.float32), np.array(box, np.float32)

    def get_item_data(self, annotations_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=False):
        data = annotations_line.split()
        image_dir = data[0]
        boxes_label = data[1:]
        # print("imgdir=", line[0])
        # print("boxes_label=", line[1:])
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(image_dir)
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in boxes_label])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                # np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def assign_boxes(self, boxes):
        # 4 代表 x1,x2,y1,y2 + 分类onehot + 置信度
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        # 将one_hot编码的第一位设置为默认背景
        assignment[:, 4] = 1.0

        if len(boxes) == 0:
            print("ERR!!, len(boxes) == 0")
            return assignment

        # 针对每一个真实框和 所有的8732个先验框计算iou
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        # 这里4+True 是因为在上一步计算encode_box 的过程中 除 Center_x1_diff,Center_x2_diff,diffW,diffY 额外返回了iou 值
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 4 + True)

        # 站在先验框的角度（8732个先验框） 与 每一个真实框的iou [4,8732]
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_index = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_index = best_iou_index[best_iou_mask]

        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num = len(best_iou_index)
        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # TODO 这里得画图理解
        # 前四位 赋值编码后的 中心点x y偏差，wh 偏差值
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_index, np.arange(assign_num), :4]
        # 4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        assignment[:, 4][best_iou_mask] = 0
        # 将预测框与真实框iou最大的index 对应 分类 one_hot编码赋值过去
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_index, 4:]
        # -1表示先验框是否有对应的物体
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def encode_box(self, boxes, return_iou = True, variances=[0.1, 0.1, 0.2, 0.2]): # TODO  variances ？？
        # 计算预测框 和 当前GT 之间的iou
        iou = self.iou(boxes)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        assign_mask = iou > self.iou_threshold

        # 如果没有任何预测框与真实框iou>iou_threshold，那么取iou最大的作为正样本
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        if return_iou:
            # 给 assign_mask 最后一位赋值 iou
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 那些iou 是 作为正样本，找到对应先验框
        assign_anchors = self.anchors[assign_mask]

        # ---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        # (x1,y1 +  x2,y2) / 2
        box_center = 0.5 * (boxes[:2] + boxes[2:])
        box_wh = boxes[2:] - boxes[:2]

        # 计算符合正样本的预测框之间的中心点、宽、高
        assign_box_center = (assign_anchors[:, 0:2] + assign_anchors[:, 2:4]) * 0.5
        assign_box_wh = (assign_anchors[:, 2:4] - assign_anchors[:, 0:2])

        encoded_box[:, 0:2][assign_mask] = box_center - assign_box_center
        encoded_box[:, 0:2][assign_mask] /= assign_box_wh
        encoded_box[:, 0:2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assign_box_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encoded_box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def iou(self, box):
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_rightbottom = np.minimum(self.anchors[:, 2:4], box[2:4])

        inter_wh = inter_rightbottom - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)

        # 相交部分面积
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # 真实框面积
        gt_box_wh = box[2:4] - box[0:2]
        true_area = gt_box_wh[0] * gt_box_wh[1]

        # 先验框面积
        anchors_box_wh = self.anchors[:, 2:4] - self.anchors[:, :2]
        anchors_box_area = anchors_box_wh[:,0] * anchors_box_wh[:, 1]

        iou = inter_area / (anchors_box_area + true_area - inter_area)

        return iou

if __name__ == "__main__":
    annotations_lines = []
    with open('../2007_train.txt', 'r') as f:
        annotations_lines = f.readlines()

    backbone        = "vgg"
    anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    class_names, num_classes = get_classes('../model_data/voc_classes.txt')
    num_classes += 1
    anchors = get_anchors([300, 300], anchors_size, backbone)

    dataloader = SSDDataset(annotations_lines, [300, 300], num_classes, anchors, False)
    image_data, boxes = dataloader[0]
    print("OK")
