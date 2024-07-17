import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from nets.vgg import vgg

def add_extra(in_channel, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # 19*19*1024 -> 19*19*256 -> 10*10*512
        layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        # 10*10*512 -> 10*10*128 -> 5*5*256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        #  5*5*256 -> 5*5*128 -> 3*3*256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        #  3*3*256 -> 3*3*128 -> 1*1*256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class SSD300(nn.Module):
    def __init__(self, number_classes, backbone_name, pretrained=False):
        super(SSD300, self).__init__()
        self.number_classes = number_classes

        loc_layers = []
        conf_layers = []
        if backbone_name == 'vgg':
            self.vgg = vgg(pretrained)
            self.extras = add_extra(1024, backbone_name)
            self.weight = nn.Parameter(torch.Tensor(512))
            self.L2Norm = L2Norm(512, 20)
            mbox = [4, 6, 6, 6, 4, 4]

            backbone_source = [21, -2]
            #---------------------------------------------------#
            #   在vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            #---------------------------------------------------#
            for k, v in enumerate(backbone_source, 0):
                loc_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * number_classes, kernel_size=3, padding=1)]

            # 在add_extras获得的特征层里
            # 第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            # shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            for k, v in enumerate(self.extras[1::2], 2): # 参数2 代表mbox 中 index = 2
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * number_classes, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.backbone_name = backbone_name


    # def normalize(self, x):
    #     x = F.normalize(x)
    #     # unsqueeze 參數是几代表在第几维扩展
    #     return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

    def forward(self, x):

        source = list()
        location = list()
        confidence = list()

        # 将输入进行特征提取，提取到23层
        if self.backbone_name == 'vgg':
            for k in range(23):
                x = self.vgg[k](x)
        # 23 层的内容需要进行l2 标准化
        s = self.L2Norm(x)
        source.append(s)

        # vgg 最后一层也属于特征输出层
        if self.backbone_name == 'vgg':
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)

        source.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == 'vgg':
                if k % 2 == 1:
                    source.append(x)

        for x, l, c in zip(source, self.loc, self.conf):
            # 将source对应特征层数据 送入对应的卷积网络进行运算后，得到位置坐标和置信度
            location.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidence.append(c(x).permute(0, 2, 3, 1).contiguous())


        # 进行resize,将每一个特征层的位置坐标取出在 1 轴上进行堆叠
        # 进行resize,将每一个特征层的置信度取出在 1 轴上进行堆叠
        loc = torch.cat([loc.view(loc.size(0), -1) for loc in location], 1)
        conf = torch.cat([conf.view(conf.size(0), -1) for conf in confidence], 1)

        outputs = (
            loc.view(loc.size(0), -1, 4), # shape(2,8732,4)
            conf.view(conf.size(0), -1, self.number_classes) # (2,8732,21) TODO 这里的2是什么意思
        )
        return outputs
