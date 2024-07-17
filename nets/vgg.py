import torch.nn as nn
from torch.hub import load_state_dict_from_url

# 300,300,3 -->
# 300,300,64 --> 300,300,64 --> 150,150,64 -->
# 150,150,128 --> 150,150,128 --> 75,75,128 -->
# 75,75,256 --> 75,75,256 --> 75,75,256 --> 38,38,256 -->
# 38,38,512 --> 38,38,512 --> 38,38,512 --> 19,19,512 -->
# 19,19,512 --> 19,19,512 --> 19,19,512 --> VGG 结束

# pook5 19,19,512 --> 19,19,512
# conv6 19,19,512 --> 19,19,1024
# conv7 19,19,1024 --> 19,19,1024

# 38,38,512 在第22 层
# 19,19,1024 在第34 层

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


def vgg(pretrained=False):
    layers = []
    in_channels = 3

    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers+=[nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            relu = nn.ReLU(inplace=True)
            layers += [conv, relu]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    model = nn.ModuleList(layers)

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",model_dir="./model_data")
        state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    net = vgg()
    for i, layer in enumerate(net):
        print(i, layer)