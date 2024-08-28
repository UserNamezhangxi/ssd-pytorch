import colorsys

import numpy as np
import torch
from PIL import ImageFont, ImageDraw

from nets.ssd import SSD300
from utils.BBoxUtils import BBoxUtils
from utils.anchors import get_anchors
from utils.utils import get_classes,cvtColor,resize_image,preprocess_input


class SSD(object):
    _defaults = {
        "model_path" : "model_data/ep200_loss3.058_val-loss4.327.pth",
        "class_path" : "model_data/voc_classes.txt",
        "input_shape": [300, 300],
        "backbone": 'vgg',
        "anchors_size": [30, 60, 111, 162, 213, 264, 315],
        "device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "letterbox_image": False,
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.45,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else :
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes = get_classes(self.class_path)
        self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone))
        self.anchors = self.anchors.to(self.device)
        self.num_classes = self.num_classes + 1

        hsv_tuples = [(x/self.num_classes,1.0,1.0) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_utils = BBoxUtils(self.num_classes)
        self.generate()

    def generate(self):
        self.net = SSD300(self.num_classes, self.backbone, pretrained=False)
        model_dict = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(model_dict)
        self.net = self.net.eval()
        self.net.to(device=self.device)
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    def detect_image(self, image, crop=False, count=False):
        image_shape = np.shape(image)[0:2]
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            images = images.to(self.device)
            outputs = self.net(images)
            results = self.bbox_utils.decode_box(outputs, self.anchors, self.input_shape, image_shape, self.letterbox_image,
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

            if len(results[0]) <= 0:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            print("predicted_class " + str(predicted_class) + ", " + str(int(c)))
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image







