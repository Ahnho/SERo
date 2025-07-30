import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x 
 
    @staticmethod
    def backward(ctx, grad):
        return grad


class Static_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)

        # print(( x * mask).mean())
        # print(x.size(),mask.size())
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad*mask, None
    
    
class Static_scale_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        
        return x.abs() * grad * mask  , None
    
class scale_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        x, mask = ctx.saved_tensors
        
        return (x ** 2) * grad   , None
    
class Dynamic_Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad , None

# class MaskerScaling(torch.autograd.Function):
#     # mask에 따라서 다르게 주기
#     @staticmethod
#     def forward(ctx, x, mask, beta, threshold):
#         ctx.save_for_backward(mask)
#         ctx.beta = beta
#         ctx.th = torch.abs(torch.abs(x) - threshold)
#         ctx.th2 = torch.abs(x)

#         return x * mask

#     @staticmethod
#     def backward(ctx, grad):
#         (mask,) = ctx.saved_tensors

#         # TODO : 양쪽 beta 튜닝
#         # 살아있는애는 0으로, 죽은애는 threshold로
#         return (
#             (grad * (1 - mask) * ctx.th * ctx.beta)
#             + (grad * mask * ctx.th2 * ctx.beta),
#             None,
#             None,
#             None,
#         )


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',scale_it=None):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.type_value = 1

    def forward(self, input):
        if self.type_value == 0:
            masked_weight = Static_Masker.apply(self.weight, self.mask)

        elif self.type_value == 1:
            masked_weight = Dynamic_Masker.apply(self.weight, self.mask)       

        return super(MaskConv2d, self)._conv_forward(input, masked_weight,self.bias)

    
class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.type_value = 0
        # self.beta = 1.0
        # self.threshold = Parameter(torch.zeros(1), requires_grad=False)

    
    def forward(self, input):

        if self.type_value == 0:
            masked_weight = Static_Masker.apply(self.weight, self.mask)

        elif self.type_value == 1:
            masked_weight = Dynamic_Masker.apply(self.weight, self.mask)
        elif self.type_value == 2:
            masked_weight = dense.apply(self.weight)
        if self.type_value == 3:
            masked_weight = Static_scale_Masker.apply(self.weight, self.mask)
        if self.type_value == 4:
            masked_weight = scale_Masker.apply(self.weight, self.mask)

        return F.linear(input, masked_weight,self.bias)
        # return F.linear(input, masked_weight
        
    def set_type_value(self, type_value):
        self.type_value = type_value