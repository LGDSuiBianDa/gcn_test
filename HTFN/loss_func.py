import torch
import torch.nn as nn

class loss_L12(nn.Module):
    def __init__(self,fact):
        super(loss_L12, self).__init__()
        self.fact=fact
    def forward(self,output,target):
        batch_size_ = output.size()[0] # 获得batch_size
        gap=output-target
        mape=torch.abs(gap)+torch.pow(gap,2)/2*self.fact
        return torch.sum(mape)/batch_size_

# 改进后的损失函数
class H_Loss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        batch_size_ = y_pred.size()[0]
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, self.delta*torch.ones_like(abs_error))
        linear = abs_error - quadratic
        h_Loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.sum(h_Loss)/batch_size_
