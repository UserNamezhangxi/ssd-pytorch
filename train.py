import os

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.utils import down_load_weight, get_classes, weights_init,get_lr_scheduler,set_optimizer_lr, ssd_collect_fn
from utils.anchors import get_anchors
from nets.ssd import SSD300
from nets.ssd_training import MutilBoxLoss
from utils.dataloader import SSDDataset
from utils.utils_fit import fit_one_epoch

backbone = 'vgg'
class_path = './model_data/voc_classes.txt'
input_shape = [300, 300]
anchors_size = [30, 60, 111, 162, 213, 264, 315]
pretrained = False
model_path = "./model_data/ssd_weights.pth"

# 设置打印全部数据
# torch.set_printoptions(profile="full")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

down_load_weight(backbone)

class_names, num_classes = get_classes(class_path)
num_classes += 1
anchors = get_anchors(input_shape, anchors_size, backbone)
model = SSD300(num_classes, backbone, pretrained)
model.to(device)
print('net', model)
# 权重初始化
# TODO 为什么要进行权重初始化？
#  如果权重是随机的，非常容易出现梯度爆炸或者梯度消失问题
if not pretrained:
    weights_init(model)

# 模型加载已经训练好的权重文件
if model_path != '':
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)

    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\034[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\034[0m")

# 获取损失函数
loss_fn = MutilBoxLoss()

# 记录loss
writer = SummaryWriter(log_dir='./tensorboard_logs')

save_period = 10
save_dir = './weight_logs/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取优化器
UnFreeze_flag = False
Freeze_Train = True

Freeze_Epoch = 50
Freeze_batch_size = 16
Init_Epoch = 0

UnFreeze_Epoch = 200
Unfreeze_batch_size = 8

optimizer_type = 'sgd'
momentum = 0.937
weight_decay = 5e-4
# ------------------------------------------------------------------#
#   Init_lr         模型的最大学习率
#                   当使用Adam优化器时建议设置  Init_lr=6e-4
#                   当使用SGD优化器时建议设置   Init_lr=2e-3
#   Min_lr          模型的最小学习率，默认为最大学习率的0.01
# ------------------------------------------------------------------#
Init_lr = 2e-3
Min_lr = Init_lr * 0.01


if Freeze_Train:
    if backbone == "vgg":
        for params in model.vgg[:28].parameters(): # TODO 为什么到28？
            params.requires_grad = False

# 如果不冻结训练，直接设置batch_size 为解冻后的batch_size
batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

nbs = 64
lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-5
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# ---------------------------------------#
#   根据optimizer_type选择优化器
# ---------------------------------------#
optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
}[optimizer_type]

lr_rate_func = get_lr_scheduler('cos', Init_lr_fit, Min_lr_fit, total_iters=UnFreeze_Epoch)

train_data = open('./2007_train.txt', 'r')
valid_data = open('./2007_val.txt', 'r')

train_lines = train_data.readlines()
valid_lines = valid_data.readlines()

num_train = len(train_lines)
num_valid = len(valid_lines)

epoch_train_step = num_train // batch_size
epoch_valid_step = num_valid // batch_size

if epoch_train_step == 0 or epoch_valid_step == 0:
    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

train_dataset = SSDDataset(train_lines, input_shape, num_classes, anchors, True)  # TODO random 在训练的时候可以改为True,做对应的数据增强处理
valid_dataset = SSDDataset(valid_lines, input_shape, num_classes, anchors, False)  # TODO random 在训练的时候可以改为True,做对应的数据增强处理

train_dataloader = DataLoader(train_dataset, batch_size, True, drop_last=True, collate_fn=ssd_collect_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size, True, drop_last=True, collate_fn=ssd_collect_fn)


for epoch in range(Init_Epoch, UnFreeze_Epoch):

    if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
        # 如果不冻结训练，直接设置batch_size 为解冻后的batch_size
        batch_size = Unfreeze_batch_size
        train_dataloader = DataLoader(train_dataset, batch_size, True, drop_last=True, collate_fn=ssd_collect_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size, True, drop_last=True, collate_fn=ssd_collect_fn)

        # batch_size 变化后动态调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-5
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        lr_rate_func = get_lr_scheduler('cos', Init_lr_fit, Min_lr_fit, total_iters=UnFreeze_Epoch)

        UnFreeze_flag = True
        if backbone == 'vgg':
            for params in model.vgg[:28].parameters():
                params.requires_grad = True

    # 动态更新学习率
    set_optimizer_lr(optimizer, lr_rate_func, epoch)
    fit_one_epoch(model, loss_fn, train_dataloader, valid_dataloader, epoch_train_step, epoch_valid_step ,optimizer, epoch, UnFreeze_Epoch, writer, device, save_period, save_dir)
