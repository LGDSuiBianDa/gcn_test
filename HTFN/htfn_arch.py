import torch
import torch.nn as nn
import math

class TemporalEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.month_emb = nn.Embedding(12, hidden_dim)
        self.week_emb = nn.Embedding(7, hidden_dim//2)
        self.holiday_emb = nn.Embedding(2, hidden_dim//4)
        self.week_linear = nn.Linear(hidden_dim//2, hidden_dim)
        self.holiday_linear = nn.Linear(hidden_dim//4, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
    # def reset_parameters(self):
    #     for i in range(len(self.read_outs)):
    #         self.read_outs[i].reset_parameters()
        
    def forward(self, time_features):
        # time_features: [batch, seq_len, 3] (month, week_of_year, is_holiday)
        month_emb = self.month_emb(time_features[...,0].long()-1)
        week_emb = self.week_emb(time_features[...,1].long())
        holiday_emb = self.holiday_emb(time_features[...,2].long())
        week_emb = self.week_linear(week_emb)
        holiday_emb = self.holiday_linear(holiday_emb)
        combined_emb = month_emb + week_emb + holiday_emb
        return self.pos_encoder(combined_emb)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class HTFN(nn.Module):
    def __init__(self, seq_len, pred_len, num_layer, event_dim, hidden_dim, temp_dim_tid, temp_dim_diw, device='cuda', dropout=0.05, num_heads=4):
        super().__init__()
        # 时序特征处理
        self.temp_embed = TemporalEmbedding(hidden_dim)
        
        # 新增货量特征编码层
        self.vol_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 将单维货量映射到隐藏空间
            nn.LayerNorm(hidden_dim)
        )
        
        self.cnn = nn.Sequential(
            nn.Conv1d(2*hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2)
        )
        
        # self.bn_emb = nn.BatchNorm1d(hidden_dim)  # 添加 BatchNorm 到嵌入层之后
        # self.ln_emb = nn.LayerNorm(hidden_dim)    # 添加 LayerNorm 到嵌入层之后
        
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2)
        # )
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 事件特征处理
        self.event_emb = nn.Sequential(
            nn.Linear(event_dim, hidden_dim//2),  
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        
        self.event_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # self.event_encoder = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # 调整事件特征维度
        self.event_dim_adjust = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # 融合与预测
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # 展平操作，为全连接层做准备
            nn.Linear(hidden_dim, pred_len)
        )
    
    def reset_parameters(self):
        # for i in range(len(self.cnn)):
        #     self.conv[i].reset_parameters()
        #     self.ffn[i].reset_parameters()
        #     self.ffn_edge[i].reset_parameters()
        self.cnn.reset_parameters()
        self.event_emb.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, hist_vol, time_feat, event_feat, future_event):
        # 时序特征嵌入
        t_emb = self.temp_embed(time_feat)  # [B, T_hist, D]
        
        v_emb = self.vol_embed(hist_vol)  # [B, T_hist, 1] -> [B, T_hist, D]
        
        # 拼接时间和货量特征
        fused_emb = torch.cat([t_emb, v_emb], dim=-1)  # [B, T_hist, 2D]
        
        # cnn_feat = self.cnn(t_emb.transpose(1,2)).transpose(1,2)
        cnn_feat = self.cnn(fused_emb.transpose(1,2)).transpose(1,2)
        
        # hidden = cnn_feat.reshape(-1, hidden_dim)  # 调整形状以适应 BatchNorm 和 LayerNorm
        # hidden = self.bn_emb(hidden)  # 使用 BatchNorm
        # hidden = self.ln_emb(hidden)  # 使用 LayerNorm
        # cnn_feat = hidden.reshape(batch_size, seq_len, hidden_dim) # # [B, T, D]
        
        lstm_out, _ = self.lstm(cnn_feat)  # [B, T_hist, D]
        
        # 事件特征处理
        event_emb = self.event_emb(event_feat)  # [B, T_hist, D]
        event_enc, _ = self.event_encoder(event_emb)  # [B, T_hist, 2D]
        # event_enc = self.event_dim_adjust(event_enc)  # 调整维度 [B, T_hist, D]
        future_event_emb = self.event_emb(future_event)  # [B, T_pred, D]
        
        # 跨时段注意力
        cross_feat, _ = self.cross_attn(
            query=lstm_out[:, -1:, :],  # 取最后一个时间步 [B,hidden_dim]
            key=torch.cat([event_enc, future_event_emb], dim=1), # [B, T_hist + T_pred, D]
            value=torch.cat([event_enc, future_event_emb], dim=1) # [B, T_hist + T_pred, D]
        ) # [B, 1, D]
        
        cross_feat = cross_feat.transpose(1,2).contiguous() # [B, D, 1]
        
        # 多尺度预测
        pred = self.predictor(cross_feat) # [B, pred_len]
        # pred = self.predictor(cross_feat).unsqueeze(2) # # [B, pred_len, 1]
        return pred # [B, pred_len]
    
