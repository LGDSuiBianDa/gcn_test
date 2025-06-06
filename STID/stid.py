import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import torch.nn.utils as utils
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import argparse
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import torch.nn.init as init
import math
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, ProfilerActivity
from torchinfo import summary
import sys
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 自定义布尔值转换函数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='STID')
    # parser.add_argument('--gnn_struc', type=str, default='concat')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=300)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--penalty', type=float, default=1e-3)
    parser.add_argument('--droupout_rate', type=float, default=0.1)
    parser.add_argument('--epoch_train', type=int, default=50)
    parser.add_argument('--input_feat_dim', type=int, default=3)
    parser.add_argument('--node_emb_dim', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=16)#>0
    parser.add_argument('--num_layer', type=int, default=2)#>0
    parser.add_argument('--num_heads', type=int, default=4)#>0
    parser.add_argument('--temp_dim_tid',type = int, default=10)
    parser.add_argument('--temp_dim_diw',type = int, default=10)
    parser.add_argument('--clip_value',type = int, default=1)
    parser.add_argument('--event_dim',type = int, default=2)
    parser.add_argument('--hidden_dim',type = int, default=8)
    parser.add_argument('--month_of_year_size',type = int, default=12)
    parser.add_argument('--day_of_week_size',type = int, default=7)
    parser.add_argument('--if_T_i_M',type = str2bool, default=True)
    parser.add_argument('--if_D_i_W',type = str2bool, default=True)
    parser.add_argument('--if_IS_WORK',type = str2bool, default=True)
    parser.add_argument('--if_Transformer',type = str2bool, default=True)
    parser.add_argument('--if_Fusion_encoder',type = str2bool, default=False)
    parser.add_argument('--if_Pos_encoder',type = str2bool, default=True)
    parser.add_argument('--if_Cross_attn',type = str2bool, default=True)
    parser.add_argument('--if_output_attn',type = str2bool, default=True)
    parser.add_argument('--method', type=str, default='add')
    parser.add_argument('--device', type=str, default='cuda')
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args 

def setup_seed(seed_n):
    # 固定随机种子等操作
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    #torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)
    
def freeze_module(model, module_name):
        for name, param in model.named_parameters():
            if module_name in name:
                param.requires_grad = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        
class WMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_true - y_pred).sum()
        denominator = torch.abs(y_true).sum() + self.epsilon
        return numerator / denominator

class HybridDynamicLoss(nn.Module):
    def __init__(self, base_weight=0.1,alpha=0.8):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.base_weight = base_weight
        self.alpha = alpha
        self.wmape = WMAPELoss()
        
    def forward(self, y_pred, y_true):
        # 计算当前 batch 缺失比例
        zeros = (y_true == 0).float().sum()
        total = torch.numel(y_true)
        zero_ratio = zeros / (total + 1e-8)  # 加epsilon防止为0
        # no_goods_weight = self.base_weight * (1 - zero_ratio)  # 比例越高权重越小
        no_goods_weight = self.base_weight * torch.exp(-3 * zero_ratio)   # 随着比例增大, 权重更小
        # 动态加权
        weights = torch.where(y_true > 0, 1.0, no_goods_weight)
        huber_loss = (self.smooth_l1(y_pred, y_true) * weights).mean()
        wmape_loss = self.wmape(y_pred, y_true)
        # 按alpha加权混合
        total_loss = self.alpha * huber_loss + (1 - self.alpha) * wmape_loss
        # Loss加权求均值
        return total_loss
    
class DaynamicWeightHuberLoss(nn.Module):
    def __init__(self, base_weight=0.1):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.base_weight = base_weight
        
    def forward(self, y_pred, y_true):
        # 计算当前 batch 缺失比例
        zeros = (y_true == 0).float().sum()
        total = torch.numel(y_true)
        zero_ratio = zeros / (total + 1e-8)  # 加epsilon防止为0
        # no_goods_weight = self.base_weight * (1 - zero_ratio)  # 比例越高权重越小
        no_goods_weight = self.base_weight * torch.exp(-3 * zero_ratio)   # 随着比例增大, 权重更小
        # 动态加权
        weights = torch.where(y_true > 0, 1.0, no_goods_weight)
        # 求加权平均损失
        huber_loss = (self.smooth_l1(y_pred, y_true) * weights).mean()
        return huber_loss
        
# 定义loss
class HuberWMAPELoss(nn.Module):
    def __init__(self, alpha=0.8, delta=1.0, eps=1e-8):
        """
        混合Huber Loss与WMAPE Loss
        
        参数:
            alpha: Huber Loss的权重(0-1之间)
            delta: Huber Loss的阈值参数
            eps: 防止除零错误的小常数
        """
        super(HuberWMAPELoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        """
        计算混合损失
        
        参数:
            y_pred: 模型预测值 (batch_size, ...)
            y_true: 真实值 (batch_size, ...)
            
        返回:
            loss: 混合损失值
        """
        # 1. 计算Huber Loss (SmoothL1Loss)
        huber_loss = nn.SmoothL1Loss(beta=self.delta)(y_pred, y_true)
        
        # 2. 计算WMAPE Loss
        # 处理可能的除零问题: 确保分母不为零
        absolute_error = torch.abs(y_pred - y_true)
        weighted_error = absolute_error / (torch.abs(y_true) + self.eps)
        wmape_loss = 10 * torch.mean(weighted_error)
        
        # 3. 加权混合
        total_loss = self.alpha * huber_loss + (1 - self.alpha) * wmape_loss
        
        return total_loss

# 计算 MAPE 的函数
def MAPE(y, real):
    # 将输入转换为numpy数组以确保矢量化操作
    y = np.asarray(y)
    real = np.asarray(real)
    
    # 检查数据中是否存在非数值类型或无穷大
    if np.isnan(y).any() or np.isnan(real).any():
        print("输入数据中存在 NaN 值，请检查数据。")
    if np.isinf(y).any() or np.isinf(real).any():
        print("输入数据中存在无穷大值，请检查数据。")
    
    # 计算绝对误差
    error = np.abs(y - real)
    
    # 处理真实值为0的特殊情况
    mask_real_zero = (real == 0)
    
    # 初始化比率数组
    ratio = np.where(
        mask_real_zero,
        # 当真实值为0时：预测也为0则误差率0，否则1
        np.where(y == 0, 0.0, 1.0),
        # 当真实值非0时：取误差率（不超过1）
        # np.minimum(error / real, 1.0)  # 等价于 np.where(error > real, 1.0, error/real)
        error / real
    )
    # print(f'y={np.max(y)}')
    # print(f'real={np.max(real)}')
    # print(f'mapeee={ratio.mean()}')
    return ratio.mean()

def WMAPE(y, real):
    # 将输入转换为 numpy 数组以确保矢量化操作
    y = np.asarray(y)
    real = np.asarray(real)

    # 检查数据中是否存在非数值类型或无穷大
    if np.isnan(y).any() or np.isnan(real).any():
        print("输入数据中存在 NaN 值，请检查数据。")
    if np.isinf(y).any() or np.isinf(real).any():
        print("输入数据中存在无穷大值，请检查数据。")

    # 计算绝对误差
    error = np.abs(y - real)

    # 计算分子：绝对误差的总和
    numerator = np.sum(error)

    # 计算分母：真实值绝对值的总和
    denominator = np.sum(np.abs(real))

    # 处理分母为 0 的情况
    if denominator == 0:
        if numerator == 0:
            return 0.0
        else:
            print("分母为 0，无法计算 WMAPE。")
            return np.nan

    # 计算 WMAPE
    wmape = numerator / denominator
    return wmape

# 计算 MAE 的函数
def MAE(y, real):
    mae = np.mean(np.abs(y-real))
    return mae

# --------------data process--------------
def preprocess_data(data,date_split,date_end):
    # 数据预处理 - 时间序列补全

    data['node_date'] = pd.to_datetime(data['node_date'])
    data['plan_start_tm'] = pd.to_datetime(data['plan_start_tm'])
    
    # 定义一个函数来计算日期差
    def date_difference(group):
        return (group.max() - group.min()).days

    # 使用 transform 方法计算每个分组的日期差
    data['date_len'] = data.groupby('node_code')['node_date'].transform(date_difference)

    data['nums'] = data.groupby('node_code')['node_date'].transform('count')
    data['active_frequency'] = data['nums']/data['date_len']
    data['active_frequency1'] = data['nums']/365
    print(f'sparse batch nums = {len(data.query("active_frequency < 0.9")["node_code"].unique())} ,sparse batch rate = {len(data.query("active_frequency < 0.9")["node_code"].unique())/len(data["node_code"].unique())}')
    filled_df = data.query("active_frequency >= 0.9").copy(deep=True)
    # data = data.query("active_frequency >= 0.9 & active_frequency1 >= 0.9 ").copy()
    
    # 将缺失的 数值 填充为 0
    filled_df['top_area_l1_sum'] = filled_df['top_area_l1_sum'].fillna(0)
    filled_df['top_area_l4_sum'] = filled_df['top_area_l4_sum'].fillna(0)
    filled_df['target_label'] = filled_df['target_label'].fillna(0)
    filled_df['target_label2'] = filled_df['target_label2'].fillna(0)
    
    # 特征工程 - 编码及标准化
    # todo 这里需要实验，将目标值为0的直接过滤掉/+1取log 的效果差异
    filled_df['votes'] = np.log(filled_df['target_label'] + 1)
    shift = 0
    scale = 15
    filled_df['votes_scale'] = (filled_df['votes']-shift)/scale
    filled_df['weight_scale'] = (np.log(filled_df['target_label2'] + 1) - shift)/scale
    
    print(f'before fillna filled_df describe() :{filled_df[["target_label","target_label2","top_area_l1_sum","top_area_l4_sum","votes_scale","weight_scale"]].describe()}')
    filled_df = filled_df.fillna(0)
    print(f'ori filled_df describe() :{filled_df[["target_label","target_label2","top_area_l1_sum","top_area_l4_sum","votes_scale","weight_scale"]].describe()}')
    filled_df = filled_df[filled_df['votes_scale'] > 0]
    print(f'dropna filled_df describe() :{filled_df[["target_label","target_label2","top_area_l1_sum","top_area_l4_sum","votes_scale","weight_scale"]].describe()}')
    
    # 添加时间特征
    filled_df['TiD'] = filled_df['plan_start_tm'].dt.hour * 12 + filled_df['plan_start_tm'].dt.minute // 5
    filled_df['DiW'] = filled_df['node_date'].dt.dayofweek
    filled_df['month'] = filled_df['node_date'].dt.month
    filled_df['quarter'] = filled_df['node_date'].dt.quarter

    filled_df['Iswork_encoded'] = np.where(filled_df['is_work'] == 'Y', 1, 0)
    filled_df['date_type_encoded'] = np.where(filled_df['date_type_label'] == -1, 0, 
                                    np.where(filled_df['date_type_label'] == 0, 1, 
                                            np.where(filled_df['date_type_label'] == 1, 2, filled_df['date_type_label'])))
    filled_df['node_encode'] = filled_df['node_code'].astype('category').cat.codes
    filled_df['trans_code'] = filled_df['node_code'].str[:-4]
    filled_df['trans_encode'] = filled_df['trans_code'].astype('category').cat.codes

    # 按时间排序
    # data.sort_values(by='timestamp', inplace=True)

    # 确定划分时间点
    split_date = filled_df['node_date'].max() - pd.DateOffset(months=1)  # 最后一个月作为验证集
    split_date = pd.to_datetime(date_split)
    end_date = pd.to_datetime(date_end)
    print(f'split_date = {split_date}')

    # 划分训练集和验证集
    train_data = filled_df[filled_df['node_date'] <= split_date].copy()
    vali_data = filled_df[(filled_df['node_date'] > split_date) & (filled_df['node_date'] < end_date)].copy()

    # 定义需要标准化的列
    columns_to_scale = ['top_area_l1_sum', 'top_area_l4_sum']
    # 数据归一化
    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])

    # 在验证集上进行 transform
    vali_data[columns_to_scale] = scaler.transform(vali_data[columns_to_scale])
    
    # month_len = len(data['month'].unique())
    # DiW_len = len(data['DiW'].unique())
    # Iswork_encoded_len = len(data['Iswork_encoded'].unique())
    node_emb_dim = len(filled_df['node_encode'].unique()) #场地嵌入
    print(f'node_encode : {node_emb_dim}')
    
    return train_data,vali_data,node_emb_dim

class LogisticsDataset(Dataset):
    def __init__(self, ts_data, hist_feat, time_feat, node_feat ,event_feat, future_event_feat,node_date,node_code, seq_len, pred_len):
        """
        ts_data: 包含所有特征的 DataFrame
        hist_feat: 历史时序特征列名，如 ['votes_scale']
        time_feat: 时间特征列名，如 ['TiD','DiW','month','quarter']
        node_feat: 节点编码
        event_feat: 事件特征列名，如 ['Iswork_encoded','date_type_encoded']
        future_event_feat: 未来事件特征列名，如 ['top_area_l1_sum','top_area_l4_sum']
        node_date:节点日期
        node_code:节点编码
        seq_len: 历史窗口长度
        pred_len: 预测步长
        """
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 按节点 ID 聚合数据
        self.data = []
        for node_id, group in ts_data.groupby(node_code):
            group = group.sort_values(node_date)  # 确保按日期排序
            hist_data = group[hist_feat].values                   # (T, 1)
            time_data = group[time_feat].values                   # (T, len(time_feat))
            node_data = group[node_feat].values                   # (T, len(node_feat))
            event_data = group[event_feat].values                 # (T, len(event_feat))
            future_event_data = group[future_event_feat].values   # (T, len(future_event_feat))
            dates_data = group[node_date].values
            node_codes_data = group[node_code].values
            labels = group[hist_feat].values                      # (T, 1)
            
            for i in range(len(group) - seq_len - pred_len + 1):
                self.data.append({
                    # 'features': np.stack([hist_data[i:i + seq_len], time_data[i:i + seq_len], event_data[i:i + seq_len]], axis=1),
                    'hist_vol':hist_data[i:i + seq_len],
                    'times':time_data[i:i + seq_len],
                    'nodes':node_data[i:i + seq_len],
                    'events':event_data[i:i + seq_len],
                    'future_events':future_event_data[i + seq_len:i + seq_len + pred_len],
                    'dates': [str(date)[2:12] for date in dates_data[i+seq_len:i+seq_len+pred_len]],
                    'node_codes': [str(node_code) for node_code in node_codes_data[i+seq_len:i+seq_len+pred_len]],
                    # 'dates': dates_data[i + seq_len:i + seq_len + pred_len],
                    # 'node_codes': node_codes_data[i + seq_len:i + seq_len + pred_len].tolist(),
                    'labels': labels[i + seq_len:i + seq_len + pred_len]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return (
            # torch.FloatTensor(data['features']),  # 特征
            torch.FloatTensor(data['hist_vol']),    # 历史目标值
            torch.FloatTensor(data['times']),       # 时间特征
            torch.FloatTensor(data['nodes']),       # 节点特征
            torch.FloatTensor(data['events']),      # 事件特征
            torch.FloatTensor(data['future_events']),   # 未来事件
            data['dates'],                          # 预测日期
            data['node_codes'],                    # 节点 ID
            torch.FloatTensor(data['labels']),      # 标签
        )

def split_dataset_by_date(dataset, split_date):
    train_indices = []
    val_indices = []
    for idx in range(len(dataset)):
        # 获取当前样本的日期
        dates = dataset[idx][5]
        # 假设日期是按升序排列的，取第一个日期作为判断依据
        first_date = dates[0]
        if first_date <= split_date:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
        
def my_data_loader(train_data,vali_data, hist_feat, time_feat, node_feat, event_feat, future_event_feat,node_date,node_code,seq_len,pred_len,batch_size):
    """_summary_

    Args:
        train_data (_type_): _description_
        vali_data (_type_): _description_
        hist_feat=['votes_scale']
        time_feat= ['TiD','DiW','month','quarter','Iswork_encoded']
        node_feat = ['trans_code_encode']
        event_feat= ['date_type_encoded']
        future_event_feat= ['top_area_l1_sum','top_area_l4_sum']
        seq_len (_type_): _description_
        pred_len (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 合并数据
    df = pd.concat([train_data, vali_data], axis=0, ignore_index=True)
    # 实例化 TimeSeriesDataset
    dataset = LogisticsDataset(df, hist_feat, time_feat, node_feat,event_feat, future_event_feat, node_date, node_code, seq_len,pred_len)

    # 划分日期
    split_date = '2025-01-20'
    # 创建数据集
    train_dataset, val_dataset = split_dataset_by_date(dataset, split_date)
    
    # 查看数据集长度
    print("train 数据集长度:", len(train_dataset))
    print("val 数据集长度:", len(val_dataset))
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True,persistent_workers=True)
    
    return train_loader,val_loader

# --------------model arch--------------
# 模型架构
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=400, dropout=0.01):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 全局位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('global_pe', pe)
        
        # 周周期位置编码（假设每周7天）
        week_pe = torch.zeros(max_len, hidden_dim)
        week_position = torch.arange(max_len) % 7
        week_div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(100.0) / hidden_dim))
        week_pe[:, 0::2] = torch.sin(week_position.unsqueeze(1) * week_div_term)
        week_pe[:, 1::2] = torch.cos(week_position.unsqueeze(1) * week_div_term)
        self.register_buffer('week_pe', week_pe)
        
        # 月周期位置编码（假设每月30天）
        month_pe = torch.zeros(max_len, hidden_dim)
        month_position = torch.arange(max_len) % 30
        month_div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(100.0) / hidden_dim))
        month_pe[:, 0::2] = torch.sin(month_position.unsqueeze(1) * month_div_term)
        month_pe[:, 1::2] = torch.cos(month_position.unsqueeze(1) * month_div_term)
        self.register_buffer('month_pe', month_pe)
        
        # 融合权重
        self.alpha_g = nn.Parameter(torch.ones(1))
        self.alpha_w = nn.Parameter(torch.ones(1))
        self.alpha_m = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x: [B, seq_len, hidden_dim]
        seq_len = x.size(1)
        pe = (self.alpha_g * self.global_pe[:seq_len] + 
              self.alpha_w * self.week_pe[:seq_len] + 
              self.alpha_m * self.month_pe[:seq_len])
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, hidden_dim,if_Pos_encoder,method):
        super().__init__()
        self.if_pos_encoder = if_Pos_encoder
        self.method = method
        self.hidden_dim = 16 if hidden_dim>16 else hidden_dim
        self.date_emb_dim = 2 if self.hidden_dim//8 <= 1 else self.hidden_dim//8
        self.month_emb = nn.Embedding(12, self.hidden_dim//4)
        self.week_emb = nn.Embedding(7, self.hidden_dim//4)
        self.holiday_emb = nn.Embedding(2, self.date_emb_dim)
        self.tid_emb = nn.Embedding(288,self.hidden_dim//4)
        self.date_type_emb = nn.Embedding(3,self.date_emb_dim)
        if self.method == 'add':
            self.month_linear = nn.Linear(self.hidden_dim//4, self.hidden_dim)
            self.week_linear = nn.Linear(self.hidden_dim//4, self.hidden_dim)
            self.tid_linear = nn.Linear(self.hidden_dim//4, self.hidden_dim)
            self.holiday_linear = nn.Linear(self.date_emb_dim, self.hidden_dim)
            self.date_type_linear = nn.Linear(self.date_emb_dim, self.hidden_dim)
            self.fuse_dim_adjust = nn.Linear(self.hidden_dim,hidden_dim)
        else:
            self.fuse_dim_adjust = nn.Linear(self.hidden_dim//2+self.hidden_dim//4+2*self.date_emb_dim, hidden_dim)
        
    def forward(self, time_features):
        # time_features: [batch, seq_len, 5] (['month','TiD','DiW','Iswork_encoded','date_type_encoded'])
        month_emb = self.month_emb(time_features[...,0].long()-1)
        tid_emb = self.tid_emb(time_features[...,1].long())
        week_emb = self.week_emb(time_features[...,2].long())
        holiday_emb = self.holiday_emb(time_features[...,3].long())
        date_type_emb = self.date_type_emb(time_features[...,4].long())
        if self.method == 'add':
            month_emb = self.month_linear(month_emb)
            week_emb = self.week_linear(week_emb)
            tid_emb = self.tid_linear(tid_emb)
            holiday_emb = self.holiday_linear(holiday_emb)
            date_type_emb = self.date_type_linear(date_type_emb)
            combined_emb = month_emb + tid_emb + week_emb + holiday_emb + date_type_emb #[B,L,D]
            combined_emb = self.fuse_dim_adjust(combined_emb)
        else:
            combined_emb = torch.cat([month_emb , tid_emb , week_emb , holiday_emb , date_type_emb], dim=-1) #[B,L,self.hidden_dim//2+self.hidden_dim+self.hidden_dim//4+2*self.date_emb_dim]
            combined_emb = self.fuse_dim_adjust(combined_emb) #[B,L,D]

        return combined_emb
    
class TCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding,dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(1,2)  # [B, D, L]
        out = self.conv(x)    # [B, out_dim, L]
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out.transpose(1,2)
        return out

class MultiScaleTCN(nn.Module):
    def __init__(self, hidden_dim,num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.tcn1 = TCNBlock(self.hidden_dim,self.hidden_dim,3,1)
        self.tcn2 = TCNBlock(self.hidden_dim,self.hidden_dim,5,2)
        self.tcn3 = TCNBlock(self.hidden_dim,self.hidden_dim,7,3)
        self.global_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
    def forward(self, x):
        x1 = self.tcn1(x)
        x2 = self.tcn2(x)
        x3 = self.tcn3(x)
        x_tcn = x1 + x2 + x3     # [B, L, D]
        x_attn, _ = self.global_attn(x_tcn, x_tcn, x_tcn)
        return x_attn

class FeatureResidualConv(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.conv = nn.Sequential(
        #     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2)
        # )
        
        self.conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        identity = x
        out = self.conv(x)
        out += identity
        return out

class FeedForward(nn.Module):
    def __init__(self, hidden_dim,ff_dim,dropout=0.1):
        super(FeedForward,self).__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4*self.hidden_dim),  
            nn.ReLU(),
            nn.Linear(4*self.hidden_dim, self.ff_dim),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.ff_dim)
        )
    
    def forward(self,x):
        out = self.ffn(x)
        return out

class MultiLayerFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_layers=3, dropout=0.1):
        """
        d_model: 输入和最终输出维度
        d_ff: 隐藏层宽度
        num_layers: FFN内部全连接层数，>=2
        dropout: 每层后的dropout概率
        """
        super().__init__()
        self.residual_proj = nn.Linear(d_model, d_ff)
        layers = []
        # 第一层: d_model -> 4*d_model
        layers.append(nn.Linear(d_model, 2*d_model))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        # 中间层: 4*d_model -> 4*d_model
        for _ in range(num_layers-2):
            layers.append(nn.Linear(2*d_model, 2*d_model))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        # 最后一层: 4*d_model -> d_ff（保持残差输入输出一致）
        layers.append(nn.Linear(2*d_model, d_ff))
        layers.append(nn.Dropout(dropout))
        self.ffn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_ff)

    def forward(self, x):
        # x: [B, L, D]
        out = self.ffn(x)
        res = self.residual_proj(x)
        out = self.norm(out + res) # 残差+LayerNorm
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim,num_heads,dropout,num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=self.num_heads, 
            dim_feedforward= 4*self.hidden_dim,
            dropout=self.dropout, 
            batch_first=True)
        self.encoder  = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
    
    def forward(self,x):
        out = self.encoder(x)
        return out
    
class OutputAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(OutputAttention,self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self,x):
        # 计算注意力分数
        attention_scores = self.attention_weights(x).squeeze(-1) #[B,T_hist + T_pred]
        # 对注意力分数进行 softmax 归一化
        attention_probs = torch.softmax(attention_scores, dim=1)
        # 计算加权输出
        weighted_output = torch.sum(x * attention_probs.unsqueeze(-1), dim=1)
        return weighted_output

class MultiHeadOutputAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_weights = nn.Linear(hidden_dim, num_heads)
        
    def forward(self, x):
        # x: [B, T, H]
        attn_scores = self.attention_weights(x)    # [B, T, num_heads]
        attn_probs = torch.softmax(attn_scores, dim=1)  # softmax over time
        weighted_outputs = []
        for h in range(self.num_heads):
            attn = attn_probs[:,:,h]             # [B,T]
            weighted = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B,H]
            weighted_outputs.append(weighted)
        out = torch.cat(weighted_outputs, dim=-1) # [B,H*num_heads]
        return out   # 作为表征特征输出, 可用于后续任务头

class StepwiseMLPHead(nn.Module):
    def __init__(self, input_dim, pred_len, day_emb_dim=8, mlp_hidden=128, mlp_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.day_embed = nn.Embedding(pred_len, day_emb_dim)
        self.heads = nn.ModuleList([
            self._make_mlp(input_dim + day_emb_dim, mlp_hidden, mlp_layers, dropout)
            for _ in range(pred_len)
        ])
        
    def _make_mlp(self, in_dim, hidden_dim, num_layers, dropout):
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, H]
        B = x.size(0)
        outs = []
        for step, head in enumerate(self.heads):
            step_id = torch.full((B,), step, dtype=torch.long, device=x.device)
            step_emb = self.day_embed(step_id)   # [B, day_emb_dim]
            f = torch.cat([x, step_emb], dim=-1) # [B, H+day_emb_dim]
            pred = head(f)    # [B,1]
            outs.append(pred)
        outs = torch.cat(outs, dim=1)  # [B, pred_len]
        return outs

class MyPredictor(nn.Module):
    def __init__(self, in_dim, num_heads, pred_len, mlp_hidden=128, mlp_layers=2, dropout=0.1, day_emb_dim=8):
        super().__init__()
        self.attn = MultiHeadOutputAttention(in_dim, num_heads)
        self.ffn = MultiLayerFFN(num_heads*in_dim, num_heads*in_dim, num_layers=2, dropout=dropout)
        self.stepwise_head = StepwiseMLPHead(num_heads*in_dim, pred_len, day_emb_dim, mlp_hidden, mlp_layers, dropout)
        self.pred_len = pred_len
        
    def forward(self, x):
        # x: [B, T, D]
        feat = self.attn(x)        # [B, H] = [B, num_heads*D]
        feat_ffn = self.ffn(feat)  # [B, H]
        out = self.stepwise_head(feat_ffn)  # [B, pred_len]
        return out


class HTFN(nn.Module):
    def __init__(self, seq_len, pred_len, num_layer, event_dim, hidden_dim, temp_dim_tid, temp_dim_diw, device, dropout, num_heads,
                 month_of_year_size,day_of_week_size,if_T_i_M,if_D_i_W,if_IS_WORK,
                 if_Transformer,if_Fusion_encoder,if_Pos_encoder,if_Cross_attn,if_output_attn,method,d_model,num_nodes):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layer = num_layer
        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.device = device
        self.dropout = dropout
        self.num_heads = num_heads
        
        self.month_of_year_size = month_of_year_size
        self.day_of_week_size = day_of_week_size
        
        self.if_time_in_month = if_T_i_M
        self.if_day_in_week = if_D_i_W
        self.if_is_work = if_IS_WORK
        
        self.if_fusion_encoder = if_Fusion_encoder
        self.if_transformer = if_Transformer
        self.if_pos_encoder = if_Pos_encoder
        self.if_cross_attn = if_Cross_attn
        self.if_output_attn = if_output_attn
        self.method = method
        self.num_blocks = 30
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        self.node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.d_model)
        
         # 时序特征处理
        self.temp_embed = TemporalEmbedding(self.hidden_dim,self.if_pos_encoder,self.method)
        
        # 时间特征处理
        # self.ffn = FeedForward(self.event_dim,self.hidden_dim)
        self.ffn = MultiLayerFFN(self.event_dim,self.hidden_dim,2*self.num_layer,self.dropout)
        self.event_ffn = MultiLayerFFN(self.event_dim,3 * self.hidden_dim + self.d_model,2*self.num_layer,self.dropout)
        self.vol_embed = MultiLayerFFN(1,self.hidden_dim,2*self.num_layer,self.dropout) # 将单维目标label映射到隐藏空间
                   
        if self.if_fusion_encoder:
            # 时间-目标值融合编码器
            self.fusion_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim + 1, self.hidden_dim*2),  # 直接使用原始目标值
                nn.GELU(),
                nn.Linear(self.hidden_dim*2, self.hidden_dim)
            )
            
        self.cnn = FeatureResidualConv(3 * self.hidden_dim + self.d_model)
        # self.cnn = MultiScaleTCN(self.hidden_dim,self.num_heads)
        
        # 事件特征添加位置编码
        self.pos_encoder = PositionalEncoding(3 * self.hidden_dim + self.d_model)
        
        # transformer
        if self.if_transformer:
            # self.encoder  = BlockWiseTransformer(self.hidden_dim,self.num_blocks,self.num_heads,self.dropout,self.num_layer)
            # self.event_encoder = BlockWiseTransformer(self.hidden_dim,self.num_blocks,self.num_heads,self.dropout,self.num_layer)
            self.lstm = nn.LSTM(3 * self.hidden_dim + self.d_model, 3 * self.hidden_dim + self.d_model, num_layers=self.num_layer,bidirectional=True, batch_first=True)
            # self.lstm = nn.LSTM(3 * self.hidden_dim + self.d_model, 3 * self.hidden_dim + self.d_model, num_layers=self.num_layer, batch_first=True)
            self.encoder  = TransformerEncoder(2*(3 * self.hidden_dim + self.d_model),self.num_heads,self.dropout,self.num_layer)
            
        else: 
            self.encoder = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layer,batch_first=True)
            self.event_encoder = nn.LSTM(self.hidden_dim, self.hidden_dim//2,num_layers=self.num_layer, bidirectional=True, batch_first=True)
        
        # 融合
        if self.if_cross_attn:
            # 事件特征添加位置编码
            self.pos_event_encoder = PositionalEncoding(self.hidden_dim)
            
            # 跨时段注意力
            self.cross_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
        
        if self.if_output_attn:
            # self.predictor = nn.Sequential(
            #     OutputAttention(self.hidden_dim),
            #     # nn.Linear(self.hidden_dim, self.pred_len),
            #     MultiLayerFFN(self.hidden_dim,self.pred_len,self.num_layer,self.dropout)
            # )
            # self.predictor = nn.Sequential(
            # MultiHeadOutputAttention(2*(3 * self.hidden_dim + self.d_model),self.num_heads),
            # MultiLayerFFN(self.num_heads*2*(3 * self.hidden_dim + self.d_model),self.pred_len,self.num_layer,self.dropout)
            # )
            self.predictor = MyPredictor(2*(3 * self.hidden_dim + self.d_model), self.num_heads, self.pred_len, mlp_hidden=128, mlp_layers=2, dropout=0.1, day_emb_dim=8)
        else:
            # 多尺度预测
            self.predictor = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(start_dim=1),  # 展平操作，为全连接层做准备
                nn.Linear(self.hidden_dim, self.pred_len)
            )

    def forward(self, hist_vol, time_feat, event_feat, future_event,x_nodes):
        # # feature_cols = ['votes_scale','TiD','DiW','Iswork_encoded','date_type_encoded','top_area_l1_sum','top_area_l4_sum','trans_encode']
        # batch_size, seq_len, feat_dim = src.shape # [B, T_hist, D=8]
        
        # hist_vol, time_feat, event_feat, future_event = src[:, :, :1], src[:, :, 1:5], src[:, :, 5:7],tgt[:, :, 5:7]
        # node_emb = self.node_emb(x_nodes.squeeze(-1).long()-1) #[B, T_hist,d_model]
        node_emb = self.node_emb(x_nodes[...,0].long()) #[B, T_hist,d_model] squeeze(-1)和[...,0]等价，都是将[B,T,1] -> [B,T]
        
        # 时序特征嵌入
        t_emb = self.temp_embed(time_feat)  # [B, T_hist, D]
        
        # 时间特征
        event_emb = self.ffn(event_feat)  # [B, T_hist, D]
        future_event_emb = self.event_ffn(future_event)  # [B, T_pred, 3D+d_model]
                
        # 融合时间和原始货量特征
        if self.if_fusion_encoder:
            # 直接融合
            fused = torch.cat([t_emb, hist_vol], dim=-1)  # [B, T, D+1]
            fused = self.fusion_encoder(fused)  # [B, T, D]
        else:
            v_emb = self.vol_embed(hist_vol)  # [B, T_hist, 1] -> [B, T_hist, D]
            # 线性变换为hidden_dim
            fused_emb = torch.cat([t_emb, v_emb, node_emb,event_emb], dim=-1)  # [B, T_hist, 3D+d_model]
            # fused = self.fuse_dim_adjust(fused_emb)  # [B, T_hist, D]
            fused = torch.cat([fused_emb, future_event_emb], dim=1) # [B, T_hist + T_pred, 3D+d_model]
        
        # 一维卷积提取不同层次的特征
        cnn_feat = self.cnn(fused.transpose(1,2)).transpose(1,2) # [B, T_hist + T_pred, 3D+d_model]
        # cnn_feat = self.cnn(fused) # [B, T, D]
        
        # 4. 添加时间位置编码
        cnn_feat = self.pos_encoder(cnn_feat)
        
        # 历史货量和事件的融合特征encoder (transformer / LSTM)
        if self.if_transformer:
            lstm_out, _ = self.lstm(cnn_feat) #[B, T_hist + T_pred, 3D+d_model]
            former_out = self.encoder(lstm_out) # [B, T_hist + T_pred, 2*(3D+d_model)]
        else:
            former_out,_ = self.encoder(cnn_feat)  
        
            
        if self.if_output_attn:
            pred = self.predictor(former_out) # [B, pred_len]
        else:
            # 多尺度预测 取最后一个时间步
            pred = self.predictor(former_out[:, -1:, :].transpose(1,2).contiguous()) # [B, pred_len]
            
        return pred.unsqueeze(2) # [B, pred_len, 1]

def add_days_to_date(date_str, days):
    # 将字符串转换为datetime对象
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 计算新的日期
    new_date_obj = date_obj + timedelta(days=days)
    
    # 将新的日期转换回字符串
    new_date_str = new_date_obj.strftime('%Y-%m-%d')
    
    return new_date_str

if __name__ == "__main__":
    
    
    args = parse_args()
    print(args)

    setup_seed(args.seed)
    # def main():
    
    seq_len = args.seq_len  # 使用过去7天的数据
    pred_len = args.pred_len  # 预测未来2天的目标量
    batch_size=args.batch_size
    device = args.device
    lr_rate = args.lr
    wd=args.penalty
    input_feat_dim=args.input_feat_dim
    node_emb_dim=args.node_emb_dim
    # embed_dim = 64
    # num_layer=4
    embed_dim = args.embed_dim
    num_layer=args.num_layer
    temp_dim_tid=args.temp_dim_tid
    temp_dim_diw=args.temp_dim_diw
    clip_value=args.clip_value
    num_epochs = args.epoch_train
    dropout=args.droupout_rate
    num_heads = args.num_heads
    
    event_dim=args.event_dim
    hidden_dim=args.hidden_dim

    month_of_year_size = args.month_of_year_size
    day_of_week_size= args.day_of_week_size

    if_T_i_M= args.if_T_i_M
    if_D_i_W= args.if_D_i_W
    if_IS_WORK= args.if_IS_WORK
    
    if_Transformer= args.if_Transformer
    if_Fusion_encoder= args.if_Fusion_encoder
    if_Pos_encoder= args.if_Pos_encoder
    if_Cross_attn= args.if_Cross_attn
    if_output_attn = args.if_output_attn
    method = args.method
    num_workers = 4
    
    d_model = 4 #节点的嵌入维度
    
    print('load data')
    data_path = '/xxx/stid_dataset_latest.csv'
    data = pd.read_csv(data_path)

    # target_id =['xxx1','xxx2']

    # # 过滤出 City 列中值在指定列表中的行
    # data = data[data['node_code'].isin(target_id)].copy()

    print(data.head(5))

    split_date = '2025-01-20'
    end_date = '2025-05-31'
    train_data,vali_data,node_emb_dim = preprocess_data(data,split_date,end_date)
    print(f'train_data len:{len(train_data)}')
    print(f'vali_data len:{len(vali_data)}')

    hist_feat=['votes_scale']
    time_feat= ['month','TiD','DiW','Iswork_encoded','date_type_encoded']
    node_feat = ['node_encode']
    event_feat= ['top_area_l1_sum','top_area_l4_sum']
    future_event_feat= ['top_area_l1_sum','top_area_l4_sum']
    node_date=['node_date']
    node_code = ['node_code']
    
    # DataLoader
    train_loader,val_loader = my_data_loader(train_data,vali_data, hist_feat, time_feat, node_feat, event_feat, future_event_feat,node_date,node_code,seq_len,pred_len,batch_size)
    
    model = HTFN(seq_len, pred_len, num_layer, event_dim, hidden_dim, temp_dim_tid, temp_dim_diw, device, dropout, num_heads,
                    month_of_year_size,day_of_week_size,if_T_i_M,if_D_i_W,if_IS_WORK,if_Transformer,if_Fusion_encoder,if_Pos_encoder,if_Cross_attn,if_output_attn,method,d_model,node_emb_dim)
    # 应用参数初始化
    model.apply(init_weights)
    
    model.to(torch.device(device))
    criterion = nn.SmoothL1Loss()
    # criterion = HuberWMAPELoss(alpha=0.9) # 80% Huber + 20% WMAPE
    # criterion = HybridDynamicLoss(base_weight=0.1,alpha=0.8)
    # criterion = DaynamicWeightHuberLoss(base_weight=0.1)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=wd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_rate/100, last_epoch=-1, verbose=False)

    train_losses = []
    vali_losses = []
    # 训练模型

    train_metric={}
    vali_metric={}

    # metric  = [['MAPE'],['MAE'],['RMSE']]
    metric  = [['MAPE','WMAPE','MAE'],[MAPE,WMAPE,MAE]]

    print("Begin Training.")

    shift = 0
    scale = 15

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss_all = 0
        item_count=0
        predict_value_list = []
        real_value_list=[]
        predict_alpha_list=[]
        real_alpha_list=[]
        alpha_value_max={'node_cnt':0}
        for batch in tqdm(train_loader,mininterval=60):
            x_his, x_ts,x_nodes,x_event, x_future_event,node_date,node_code, y_true  = batch
            item_count+=1
            x_his, x_ts, x_nodes,x_event, x_future_event,y_true = x_his.to(device), x_ts.to(device), x_nodes.to(device),x_event.to(device), x_future_event.to(device), y_true.to(device)  
                
            output = model(x_his, x_ts, x_event, x_future_event,x_nodes) #前向传播 
            
            train_loss = criterion(output*scale + shift, y_true*scale + shift) #计算损失
            
            optimizer.zero_grad() #梯度清零
            
            # with autocast():
            #     output = model(x_his, x_ts, x_event, x_future_event) #前向传播
            #     train_loss = criterion(output*scale + shift, y_true*scale + shift) #计算损失
            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
                
            train_loss.backward() #反向传播 计算梯度
            # # 梯度裁剪
            # # utils.clip_grad_norm_(model.parameters(), clip_value)
            # # # 打印梯度
            # # total_norm = 0
            # # for p in model.parameters():
            # #     if p.grad is not None:
            # #         param_norm = p.grad.data.norm(2)
            # #         total_norm += param_norm.item() ** 2
            # # total_norm = total_norm ** 0.5
            # # print(f"Gradient Norm: {total_norm}")
            
            optimizer.step() #更新参数
            
            train_loss_all += train_loss.item()
            
            # train_losses.append(train_loss.item())
            
            real_alpha_node = y_true.cpu().detach().numpy()  
            output_alpha_node = output.cpu().detach().numpy()
            
            predict_alpha_list.append(output_alpha_node)
            real_alpha_list.append(real_alpha_node)
            
            # scale1 = np.array([self.scale['edge_cnt'],self.scale['edge_wt']])
            # shift1 = np.array([self.shift['edge_cnt'],self.shift['edge_wt']])
            
            predict_value_list.append(np.exp(output_alpha_node*scale + shift))
            real_value_list.append(np.exp(real_alpha_node*scale + shift))
            
            alpha_value_max['node_cnt'] = max(alpha_value_max['node_cnt'],abs(output_alpha_node[:,0]).max())
            
        predict_value=np.concatenate(predict_value_list)
        real_value=np.concatenate(real_value_list)
        
        avg_train_loss=train_loss_all / item_count
        train_losses.append(avg_train_loss)
        
            
        for i,metric_key in enumerate(metric[0]):
            train_metric[metric_key+'_node_cnt']=metric[1][i](predict_value,real_value)
            
        print(f'alpha_value_max:{alpha_value_max}')
                        
        model.eval()  # 设置为评估模式

        # # 使用 profiler 分析模型性能
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with torch.profiler.record_function("model_inference"):
        #         for x_his, x_ts, x_event, x_future_event, batch_date,batch_code, y_true  in val_loader:
        #             x_his, x_ts, x_event, x_future_event,y_true = x_his.to(device), x_ts.to(device), x_event.to(device), x_future_event.to(device), y_true.to(device) 
        #             model(x_his, x_ts, x_event, x_future_event) # 模型推理

        # # 打印性能分析结果
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        test_loss_all=0
        with torch.no_grad():
            val_predict_value_list = []
            val_real_value_list=[]
            val_predict_alpha_list=[]
            val_real_alpha_list=[]
            for x_his, x_ts,x_nodes, x_event, x_future_event, node_date,node_code, y_true  in val_loader:
                x_his, x_ts, x_nodes,x_event, x_future_event,y_true = x_his.to(device), x_ts.to(device), x_nodes.to(device),x_event.to(device), x_future_event.to(device), y_true.to(device) 
                output = model(x_his, x_ts, x_event, x_future_event,x_nodes)
                vali_loss = criterion(output*scale + shift,y_true*scale + shift)
                
                test_loss_all += vali_loss.item()
                
                real_alpha_node = y_true.cpu().detach().numpy()  
                output_alpha_node=output.cpu().detach().numpy()
                
                val_predict_alpha_list.append(output_alpha_node)
                val_real_alpha_list.append(real_alpha_node)
                
                val_predict_value_list.append(np.exp(output_alpha_node*scale + shift))
                val_real_value_list.append(np.exp(real_alpha_node*scale + shift))
                
                
            val_predict_value=np.concatenate(val_predict_value_list)
            val_real_value=np.concatenate(val_real_value_list)
        
            avg_test_loss=test_loss_all / item_count
            vali_losses.append(avg_test_loss)
            
            for i,metric_key in enumerate(metric[0]):
                vali_metric[metric_key+'_node_cnt']=metric[1][i](val_predict_value,val_real_value)
        scheduler.step()
            
        print(f'epoch [{epoch}], / ,train_losses: {train_losses[-1]:.4f},train_metric: {train_metric}')
        print(f'/ , val_losses: {vali_losses[-1]:.4f}, val_metric: {vali_metric}')
            
    print("Training completed.")
    
    # # 测试模型
    # # 绘制学习曲线
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, num_epochs + 1), vali_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    
    
    # 评估模型
    model.eval()
    val_loss = 0
    predictions = []
    true_values = []
    dates = []
    node_codes = []
    pred_period=[]
    with torch.no_grad():
        for x_his, x_ts,x_nodes, x_event, x_future_event, node_date,node_code, y_true in val_loader:
            x_his, x_ts, x_nodes,x_event, x_future_event,y_true = x_his.to(device), x_ts.to(device), x_nodes.to(device),x_event.to(device), x_future_event.to(device), y_true.squeeze(-1).to(device) 
            # print(f'x_his shape:{x_his.shape}')
            # print(f'x_ts shape:{x_ts.shape}')
            # print(f'x_event shape:{x_event.shape}')
            # print(f'x_future_event shape:{x_future_event.shape}')
            output = model(x_his, x_ts, x_event, x_future_event,x_nodes).squeeze(-1)
            # loss = criterion(output, y_true)
            # val_loss += loss.item()
            pred = output.cpu().numpy()
            label = y_true.cpu().numpy()
            
            pred = np.exp(pred*scale + shift)-1
            label= np.exp(label*scale + shift)-1
            node_dates_np = np.array(node_date).transpose()
            node_codes_np = np.array(node_code).transpose()
            
            # print(f'node_dates_np shape = {node_dates_np.shape}')
            # print(f'node_codes_np shape = {node_codes_np.shape}')
            # print(f'pred shape = {pred.shape}')

            # 展开数据到每个时间步
            batch_size = pred.shape[0]
            for i in range(batch_size):
                if pred.shape[1] > 1:
                    for j in range(pred.shape[1]):
                        predictions.append(pred[i][j])
                        true_values.append(label[i][j])
                        dates.append(node_dates_np[i][j])
                        node_codes.append(node_codes_np[i][j])
                        pred_period.append(f"{j + 1}D")
                else:
                    predictions.append(label[i][0])
                    true_values.append(label[i][0])
                    dates.append(node_dates_np[i][0])
                    node_codes.append(node_codes_np[i][0])
                    pred_period.append('1D')

    # 创建结果DataFrame并保存到CSV
    results = {
        '预测值': predictions,
        '真实值': true_values,
        '预测日期': dates,
        '节点ID': node_codes,
        'period':pred_period
    }
    pre_data = pd.DataFrame(results)
    print(f'results_df:{pre_data.head(10)}')
    
    # 获取当前脚本文件（main.py）的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 结果为：root/model

    # 目标目录：root/results（从script_dir向上一级，再进入results）
    results_dir = os.path.join(os.path.dirname(script_dir), "results")

    # 确保目录存在（避免因目录不存在报错）
    os.makedirs(results_dir, exist_ok=True)

    file_name = f'{results_dir}/{args.seed}_{hidden_dim}_{num_heads}_{num_layer}_{lr_rate}_{seq_len}_{pred_len}_{method}_{if_Transformer}_{if_Cross_attn}_{if_output_attn}_formermixlstm_resv4.csv'
    print(f'file name:{file_name}')
    # 保存到CSV文件
    pre_data.to_csv(file_name, index=False)
    
    
    try:
        # 生成 summary（必须指定 input_size 和 device）
        # summary(model,depth=10, col_width=30,row_settings=['var_names'],device=device)
        summary(model,depth=2)
    except Exception as e:
        print(f"Error during summary generation: {e}", file=sys.stderr)
    
    print(f'model: {model}')
    
    
