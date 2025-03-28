import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

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

    
def freeze_module(model, module_name):
        for name, param in model.named_parameters():
            if module_name in name:
                param.requires_grad = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class TimeSeriesDataset(Dataset):
    def __init__(self, ts_data, hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code, seq_len, pred_len):
        """
        ts_data: 包含所有特征的 DataFrame
        hist_feat: 历史时序特征列名，如 ['votes_scale']
        time_feat: 时间特征列名，如 ['TiD','DiW','month','quarter']
        event_feat: 事件特征列名，如 ['Iswork_encoded','date_type_encoded']
        future_event_feat: 未来事件特征列名，如 ['top_area_l1_sum','top_area_l4_sum']
        batch_date:班次日期
        batch_code:班次编码
        seq_len: 历史窗口长度
        pred_len: 预测步长
        """
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 按仓库 ID 聚合数据
        self.data = []
        for batch_id, group in ts_data.groupby(batch_code):
            group = group.sort_values(batch_date)  # 确保按日期排序
            hist_data = group[hist_feat].values                   # (T, 1)
            time_data = group[time_feat].values                   # (T, len(time_feat))
            event_data = group[event_feat].values                 # (T, len(event_feat))
            future_event_data = group[future_event_feat].values   # (T, len(future_event_feat))
            dates_data = group[batch_date].values
            batch_codes_data = group[batch_code].values
            labels = group[hist_feat].values                      # (T, 1)
            
            for i in range(len(group) - seq_len - pred_len + 1):
                self.data.append({
                    # 'features': np.stack([hist_data[i:i + seq_len], time_data[i:i + seq_len], event_data[i:i + seq_len]], axis=1),
                    'hist_vol':hist_data[i:i + seq_len],
                    'times':time_data[i:i + seq_len],
                    'events':event_data[i:i + seq_len],
                    'future_events':future_event_data[i + seq_len:i + seq_len + pred_len],
                    'dates': [str(date)[2:12] for date in dates_data[i+seq_len:i+seq_len+pred_len]],
                    'batch_codes': [str(batch_code) for batch_code in batch_codes_data[i+seq_len:i+seq_len+pred_len]],
                    # 'dates': dates_data[i + seq_len:i + seq_len + pred_len],
                    # 'batch_codes': batch_codes_data[i + seq_len:i + seq_len + pred_len].tolist(),
                    'labels': labels[i + seq_len:i + seq_len + pred_len]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return (
            # torch.FloatTensor(data['features']),  # 特征
            torch.FloatTensor(data['hist_vol']),    # 历史件量
            torch.FloatTensor(data['times']),       # 时间特征
            torch.FloatTensor(data['events']),      # 事件特征
            torch.FloatTensor(data['future_events']),   # 未来事件
            data['dates'],                          # 预测日期
            data['id_codes'],                    #  ID
            torch.FloatTensor(data['labels']),      # 标签
        )
        
def split_dataset_by_date(dataset, split_date):
    train_indices = []
    val_indices = []
    for idx in range(len(dataset)):
        # 获取当前样本的日期
        dates = dataset[idx][4]
        # 假设日期是按升序排列的，取第一个日期作为判断依据
        first_date = dates[0]
        if first_date < split_date:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
            
    return train_indices, val_indices
        
def my_data_loader(train_data,vali_data, hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len,split_date):
    """_summary_

    Args:
        train_data (_type_): _description_
        vali_data (_type_): _description_
        hist_feat=['votes_scale']
        time_feat= ['TiD','DiW','month','quarter','Iswork_encoded']
        event_feat= ['date_type_encoded']
        future_event_feat= ['top_area_l1_sum','top_area_l4_sum']
        seq_len (_type_): _description_
        pred_len (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 创建数据集
    # train_dataset = TimeSeriesDataset(train_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
    # val_dataset = TimeSeriesDataset(vali_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
    # # train_dataset = LogisticsDataset(train_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
    # # val_dataset = LogisticsDataset(vali_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
    
    df = pd.concat([train_data, vali_data], axis=0, ignore_index=True)
    # 实例化 TimeSeriesDataset
    dataset = TimeSeriesDataset(df, hist_feat, time_feat, event_feat, future_event_feat, batch_date, batch_code, seq_len,pred_len)

    # 划分日期
    # split_date = '2024-12-01'
    train_dataset, val_dataset = split_dataset_by_date(dataset, split_date)
    
    # 查看数据集长度
    print("train 数据集长度:", len(train_dataset))
    print("val 数据集长度:", len(val_dataset))
    
    return train_dataset,val_dataset      

        
# def split_dataset_by_date(dataset, split_date):
#     train_indices = []
#     val_indices = []
#     for idx in range(len(dataset)):
#         # 获取当前样本的日期
#         dates = dataset[idx][4]
#         # 假设日期是按升序排列的，取第一个日期作为判断依据
#         first_date = dates[0]
#         if first_date < split_date:
#             train_indices.append(idx)
#         else:
#             val_indices.append(idx)

#     train_dataset = Subset(dataset, train_indices)
#     val_dataset = Subset(dataset, val_indices)
#     return train_dataset, val_dataset
        
# def my_data_loader(train_data,vali_data, hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len,batch_size):
#     """_summary_

#     Args:
#         train_data (_type_): _description_
#         vali_data (_type_): _description_
#         hist_feat=['votes_scale']
#         time_feat= ['TiD','DiW','month','quarter','Iswork_encoded']
#         event_feat= ['date_type_encoded']
#         future_event_feat= ['top_area_l1_sum','top_area_l4_sum']
#         seq_len (_type_): _description_
#         pred_len (_type_): _description_
#         batch_size (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     # 创建数据集
#     # train_dataset = TimeSeriesDataset(train_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
#     # val_dataset = TimeSeriesDataset(vali_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
#     # # train_dataset = LogisticsDataset(train_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
#     # # val_dataset = LogisticsDataset(vali_data,hist_feat, time_feat, event_feat, future_event_feat,batch_date,batch_code,seq_len,pred_len)
    
#     df = pd.concat([train_data, vali_data], axis=0, ignore_index=True)
#     # 实例化 TimeSeriesDataset
#     dataset = TimeSeriesDataset(df, hist_feat, time_feat, event_feat, future_event_feat, batch_date, batch_code, seq_len,pred_len)

#     # 划分日期
#     split_date = '2024-12-01'
#     train_dataset, val_dataset = split_dataset_by_date(dataset, split_date)
    
#     # 查看数据集长度
#     print("train 数据集长度:", len(train_dataset))
#     print("val 数据集长度:", len(val_dataset))
    
#     # 创建 DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader,val_loader
