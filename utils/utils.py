import math

import numpy as np
import torch
from functools import partial

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(img):
    # img = img / 255.0
    # return img
    MEANS = (104, 117, 123)
    return img - MEANS   # TODO 这里为何不进行图像归一化处理呢？ 用这个 MEANS 是什么含义


# def get_classes(classes_path):
#     with open(classes_path, encoding='utf-8') as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names, len(class_names)


def down_load_weight(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    download_urls = {
        'vgg' : 'https://download.pytorch.org/models/vgg16-397923af.pth'
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    load_state_dict_from_url(download_urls[backbone], model_dir)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ------------------------------#
#   学习率调整策略：预热+余弦退火
# ------------------------------#
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        # 前3轮进行warmup预测
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        # 训练收官阶段，模型参数需要稳定，所以最后的15轮以最小的学习率进行训练
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr

        # ------------------------------#
        # 中间轮数使用cos余弦退火策略
        # cos余弦退火：cos(当前训练轮数/总训练轮数)
        # ------------------------------#
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        # ------------------------------#
        #   预热轮数不超过3轮  1~3
        # ------------------------------#
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)

        # ------------------------------#
        #   最小学习率轮数不少于15轮
        # ------------------------------#
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

# 设置学习率
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("epoch {} , set_optimizer_lr {} ".format(epoch, lr))

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def ssd_collect_fn(batch):
    # 方法1
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    boxes = torch.from_numpy(np.array(bboxes)).type(torch.FloatTensor)
    return images, boxes