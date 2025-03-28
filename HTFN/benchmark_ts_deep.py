import torch
import torch.nn
import numpy as np
import pandas as pd
import argparse
import sys
from io import StringIO
from torchinfo import summary

from utile import *
from loss_func import *
from htfn_arch import *
from abstract_model import *
from opts import *
from feature_process import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HTFN')
    # parser.add_argument('--gnn_struc', type=str, default='concat')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=180)
    parser.add_argument('--pred_len', type=int, default=7)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--penalty', type=float, default=1e-4)
    parser.add_argument('--droupout_rate', type=float, default=0.2)
    parser.add_argument('--epoch_train', type=int, default=30)
    parser.add_argument('--input_feat_dim', type=int, default=3)
    parser.add_argument('--node_emb_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=512)#>0
    parser.add_argument('--num_layer', type=int, default=2)#>0
    parser.add_argument('--num_heads', type=int, default=4)#>0
    parser.add_argument('--temp_dim_tid',type = int, default=10)
    parser.add_argument('--temp_dim_diw',type = int, default=10)
    parser.add_argument('--clip_value',type = int, default=1)
    parser.add_argument('--event_dim',type = int, default=3)
    parser.add_argument('--hidden_dim',type = int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()

    setup_seed(args.seed)
        
    print('load data')
    data_path = 'xxx/stid_dataset_new.csv'
    data = pd.read_csv(data_path)
    
    # get feature data
    train_data,vali_data = preprocess_data(data)
    
    # get dateset
    # 指定要选取的特征
    # selected_features = ['votes_scale','TiD','DiW','month','quarter','Iswork_encoded','date_type_encoded','top_area_l1_sum','top_area_l4_sum']
    hist_feat=['votes_scale']
    # time_feat= ['TiD','DiW','month','quarter','Iswork_encoded']
    time_feat= ['month','DiW','Iswork_encoded']
    event_feat= ['date_type_encoded','top_area_l1_sum','top_area_l4_sum']
    future_event_feat= ['date_type_encoded','top_area_l1_sum','top_area_l4_sum']
    date=['date']
    target_id = ['target_id']
    
    # 数据集划分日期
    split_date = '2024-12-01'
    
    
    reg= HTFN_TS()
    
    reg.seq_len = args.seq_len
    reg.pred_len = args.pred_len
    reg.batch_size = args.batch_size
    reg.device = args.device
    reg.lr = args.lr
    reg.penalty=args.penalty
    reg.input_feat_dim=args.input_feat_dim
    reg.node_emb_dim=args.node_emb_dim

    reg.embed_dim = args.embed_dim
    reg.num_layer=args.num_layer
    reg.temp_dim_tid=args.temp_dim_tid
    reg.temp_dim_diw=args.temp_dim_diw
    reg.clip_value=args.clip_value

    reg.dropout=args.droupout_rate
    reg.num_heads = args.num_heads
    
    reg.event_dim=args.event_dim
    reg.hidden_dim=args.hidden_dim
    
    reg.epoch_train = 100
    reg.epoch_polish = 0
    
    reg.shift = 0
    reg.scale = 15
    
    reg.metric = [['MAPE','WMAPE','MAE'],[MAPE,WMAPE,MAE]]
    
    train_data_list,vali_data_list = my_data_loader(train_data,vali_data, hist_feat, time_feat, event_feat, future_event_feat,date,target_id,reg.seq_len,reg.pred_len,split_date)
    
    # fit
    reg.fit(train_dataset=train_data_list,vali_dataset=vali_data_list)
    
    print("模型结构：")
    print(reg.model)
    
    try:
        # 生成 summary（必须指定 input_size 和 device）
        print(summary(reg.model,depth=10, col_width=30,row_settings=['var_names'],device=reg.device))
    except Exception as e:
        print(f"Error during summary generation: {e}", file=sys.stderr)
    
    # print("Begin Saving model")
    # torch.save(reg.model.state_dict(), 'model/'+'_'.join(['model',str(reg.num_layer),str(reg.embed_dim),str(reg.seq_len),str(reg.seed)])+'.pth')
    # print("Saving model completed.")
