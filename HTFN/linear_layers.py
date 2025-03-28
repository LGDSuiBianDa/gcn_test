import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_,uniform_

class MLP(torch.nn.Module):
    def __init__(self, num_features,batch=True,relu=True,Dropout=0.05):
        super(MLP, self).__init__()
        modules=[]
        for i in range(len(num_features)-1):
            modules.append(nn.Linear(num_features[i],num_features[i+1],bias=True))
            if batch:
                modules.append(nn.BatchNorm1d(num_features[i+1]))
            if relu:
                modules.append(torch.nn.ReLU())
            if Dropout>0:
                modules.append(nn.Dropout(p=Dropout))
        self.mlp = nn.Sequential(*modules)
    def forward(self, x):
        return self.mlp(x)
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer,nn.Linear):
                xavier_uniform_(layer.weight)

    

class gated_MLP(torch.nn.Module):
    def __init__(self,stacks,num_feature,hidden_num_feature,out_num_feature,d=0.05):
        super(gated_MLP, self).__init__()
        self.left = nn.ModuleList([MLP([num_feature,hidden_num_feature],Dropout=d,batch=False)])
        self.right = nn.ModuleList([MLP([num_feature,hidden_num_feature],Dropout=d,batch=False)])
        for _ in range(stacks):
            self.left.append(MLP([hidden_num_feature,hidden_num_feature],Dropout=d,batch=False))
            self.right.append(MLP([hidden_num_feature,hidden_num_feature],Dropout=d,batch=False))
        self.final = nn.Linear(hidden_num_feature,out_num_feature,bias=True)
    def reset_parameters(self):
        xavier_uniform_(self.final.weight)
        # zeros_(self.final.bias)
        for i in range(len(self.left)):
            self.left[i].reset_parameters()
            self.right[i].reset_parameters()

    def forward(self, x):
        outs = [x]
        for i in range(len(self.left)):
            l=self.left[i](outs[-1])
            r=F.softmax(self.right[i](outs[-1]),dim=1)
            outs.append(torch.mul(l,r))
        x = self.final(sum(outs[1:]))
        return x
