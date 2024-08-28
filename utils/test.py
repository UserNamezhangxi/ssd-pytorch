from os import path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import init

random_seed = 200
torch.manual_seed(random_seed)
input = torch.randint(0, 100, (2, 3, 4))
print("input:")
print(input)

index = torch.randint(0, 2, (2, 1, 2))
print("index:")
print(index)

output = torch.gather(input, 0, index)
print("output:")
print(output)


"""
torch 类型的 Bool 类型也可以进行if 判断
"""
b = torch.tensor([True], dtype=torch.bool)
res = 1 if b else 0
print(",res ,", res)
"""
np.apply_along_axis 函数演示
参数1：要执行的方法
参数2：在给定数据的哪个轴上进行操作，
参数3：任意给定input 的数据
综合来说：在给定input（参数3）的数据 的 轴上（参数2），执行（参数1）方法
"""
def function(a):
    return (a[0] + a[-1]) * 3
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
res = np.apply_along_axis(function, 0, b)
print("res = " , res)




"""
np.eye 函数演示
适用于生成one-hot编码格式的数据
"""
arr1 = np.array([0,1,2,3,0,1])
print(arr1)

# 适用于生成one-hot编码格式的数据
res1 =np.eye(7)[arr1]
print(res1)

import torch.nn.functional as F
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

a = torch.tensor([[20.2,30.3],[40.4,50.5]])
res = F.normalize(a, p=2, dim=1,eps=1e-10)
print("res is ",res)

# res = L2Norm(2,3)[a]
# print("res is ",res)

for k in range(23):
    print("k",k)

#                       人        猴子      类人猿     鱼        乌龟       草    ，     人        猴子      类人猿     鱼    乌龟       草
pred = torch.Tensor([[15.2270,  5.0079,  5.3355,  -4.3531,  -4.3316,  -2.1105],[  9.3171,  -2.8099,  -3.3955, 7.4970,  8.1539,  -0.7890]])
trues_mask = torch.where(pred > 0.5, 1, 0)
softmax0 = nn.Softmax(1)
res0 = softmax0(pred)
res = torch.sum(pred * (1-trues_mask), 1)
print("softmax0", res0)
print("hard class", res)
# b = torch.LongTensor([0,2])
# res = torch.gather(a,0,b)
# print(res)

import os
os.system('shutdown /s /t 10')
