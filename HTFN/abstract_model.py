import torch
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from utile import *
from loss_func import *
from htfn_arch import *

class HTFN_TS:
    
    seq_len = 90  # 使用过去7天的数据
    pred_len = 2  # 预测未来2天的中转货量
    batch_size=32
    device = 'cuda'
    lr = 1e-3
    penalty=1e-5
    input_feat_dim=3
    node_emb_dim=20
    embed_dim = 8
    num_layer=4
    temp_dim_tid=8
    temp_dim_diw=4
    clip_value=1.0
    epoch_train = 30
    epoch_polish = 00
    dropout=0.02
    num_heads = 4
    
    event_dim=3
    hidden_dim=8
    
    shift = 0
    scale = 15
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.training_parameters()
    
    def training_parameters(self):
        # self.crit = loss_L12(2)
        self.crit = nn.SmoothL1Loss()
        # self.crit = nn.MSELoss()
        # self.crit = nn.HuberLoss()
        
    def make_model(self):
        self.model = HTFN(self.seq_len, 
                          self.pred_len, 
                          self.num_layer, 
                          self.event_dim, 
                          self.hidden_dim, 
                          self.temp_dim_tid, 
                          self.temp_dim_diw, 
                          self.device, 
                          self.dropout, 
                          self.num_heads)
        self.model.reset_parameters()
        self.model.to(self.device)
    
    def fit(self,train_dataset,vali_dataset):
        self.train_dataset = train_dataset
        print('create data loader')
        self.train_loader =  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        if vali_dataset is not None:
            self.vali_dataset = vali_dataset
            self.vali_loader = DataLoader(self.vali_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.vali_dataset=None
            self.vali_loader=None

        print('start training')
        self.train_loss=[]
        self.test_loss=[]
        self.train_metric=[]
        self.test_metric=[]
        self.train(
                  sch_type = 'cyc', 
                   lr=self.lr,
                   epochs = self.epoch_train,
                   penalty=self.penalty
                #    s1=self.s1,
                #    s2 = self.s2,
                #    step_size= self.step_size,
                #    gamma = self.gamma
        )
        self.train(
                  sch_type = 'step', 
                   lr=self.lr,
                   epochs = self.epoch_polish,
                   penalty=self.penalty
                #    s1=self.s1,
                #    s2 = self.s2,
                #    step_size= self.step_size,
                #    gamma = self.gamma
        )
    
    
    def finetune(self,train_dataset,vali_dataset,metric=[],batch_size=32):
        self.train_dataset = train_dataset
        self.metric = metric
        print('create data loader')
        self.train_loader =  DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if vali_dataset is not None:
            self.vali_dataset = vali_dataset
            self.vali_loader = DataLoader(self.vali_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.vali_dataset=None
            self.vali_loader=None

        print('start training')
        self.train_loss=[]
        self.test_loss=[]
        self.train_metric=[]
        self.test_metric=[]
        self.train(
                  sch_type = 'cyc', 
                   lr=self.lr,
                   epochs = self.epoch_train,
                   penalty=self.penalty
                #    s1=self.s1,
                #    s2 = self.s2,
                #    step_size= self.step_size,
                #    gamma = self.gamma
        )
        self.train(
                  sch_type = 'step', 
                   lr=self.lr,
                   epochs = self.epoch_polish,
                   penalty=self.penalty
                #    s1=self.s1,
                #    s2 = self.s2,
                #    step_size= self.step_size,
                #    gamma = self.gamma
        )
    
    def train(self, sch_type='cyc',lr=0.001,epochs=50,penalty=0.001,factor=1,s1=10,s2=10,step_size=10,gamma = 0.9):
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=penalty)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr/100, last_epoch=-1, verbose=False)
        # if sch_type == 'cyc':
        #     scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, self.lr/10,self.lr, step_size_up=s1, step_size_down=s2)
        # elif sch_type == 'step':
        #     scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size,gamma=gamma,last_epoch=-1,verbose=False)
        
        for epoch in range(epochs):
            #print(self.model.node_encoder[1].initial_splitter.shared.shared_layers[0].weight)
            train_loss,train_metric=self.train_epoch()
            self.train_loss.append(train_loss)
            self.train_metric.append(train_metric)
            if self.vali_dataset is not None:
                test_loss,test_metric=self.vali_epoch()
                self.test_loss.append(test_loss)
                self.test_metric.append(test_metric)         
                
                print(f'epoch[{epoch}], / ,train_losse:{self.train_loss[-1]:.4f},train_metric:{self.train_metric}')
                print(f'/ , val_losse:{self.test_loss[-1]:.4f}, val_metric:{self.test_metric}')
            else:
                print(f'epoch[{epoch}], / ,train_losse:{self.train_loss[-1]:.4f},train_metric:{self.train_metric}')
            self.scheduler.step()    
        #return loss_list,test_loss,train_mape,test_mape,final_predict_t,final_predict_v
        return 
    
    def train_epoch(self):
        self.model.train()
        self.loss_track=[]
        
        train_losses = []
        predict_value_list = []
        real_value_list=[]
        predict_alpha_list=[]
        real_alpha_list=[]
        alpha_value_max={'node_cnt':0}
        
        train_loss_all = 0
        item_count=0
        
        train_metric={}
        
        for data in tqdm(self.train_loader):
            x_his, x_ts, x_event, x_future_event,batch_date,batch_code,y_true  = data
            item_count+=1
            x_his, x_ts, x_event, x_future_event,y_true = x_his.to(self.device), x_ts.to(self.device), x_event.to(self.device), x_future_event.to(self.device), y_true.squeeze(-1).to(self.device)
            
            # for param in self.model.parameters():
            #     param.grad = None
            
            output = self.model(x_his, x_ts, x_event, x_future_event) #前向传播 
            
            train_loss = self.crit(output*self.scale + self.shift, y_true*self.scale + self.shift) #计算损失
            
            self.loss_track.append(train_loss)     
            self.optimizer.zero_grad() #梯度清零
            train_loss.backward() #反向传播 计算梯度
            # 梯度裁剪
            # utils.clip_grad_norm_(model.parameters(), clip_value)
            # if self.clip_value:
            #     clip_grad_norm_(self.model.parameters(), self.clip_value)
            # # 打印梯度
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Gradient Norm: {total_norm}")
            
            self.optimizer.step() #更新参数
            
            train_loss_all += train_loss.item()
            
            # train_losses.append(train_loss.item())
            
            real_alpha_node = y_true.cpu().detach().numpy()  
            output_alpha_node = output.cpu().detach().numpy()
            
            predict_alpha_list.append(output_alpha_node)
            real_alpha_list.append(real_alpha_node)
            
            # scale1 = np.array([self.scale['edge_cnt'],self.scale['edge_wt']])
            # shift1 = np.array([self.shift['edge_cnt'],self.shift['edge_wt']])
            
            predict_value_list.append(np.exp(output_alpha_node*self.scale + self.shift)-1)
            real_value_list.append(np.exp(real_alpha_node*self.scale + self.shift)-1)
            
            alpha_value_max['node_cnt'] = max(alpha_value_max['node_cnt'],abs(output_alpha_node[:,0]).max())
            
        predict_value=np.concatenate(predict_value_list)
        real_value=np.concatenate(real_value_list)
        
        avg_train_loss=train_loss_all / item_count
        train_losses.append(avg_train_loss)
        
        for i,metric_key in enumerate(self.metric[0]):
            train_metric[metric_key+'_node_cnt']=self.metric[1][i](predict_value,real_value)
            
        print(f'alpha_value_max:{alpha_value_max}')
    
        return train_losses,train_metric
            
    def vali_epoch(self):
        self.model.eval()
        test_loss_all=0
        item_count = 0
        with torch.no_grad():
            vali_losses = []
            val_predict_value_list = []
            val_real_value_list=[]
            val_predict_alpha_list=[]
            val_real_alpha_list=[]
            vali_metric={}
            
            for data  in self.vali_loader:
                x_his, x_ts, x_event, x_future_event,batch_date,batch_code,y_true  = data
                item_count+=1
                x_his, x_ts, x_event, x_future_event,y_true = x_his.to(self.device), x_ts.to(self.device), x_event.to(self.device), x_future_event.to(self.device), y_true.squeeze(-1).to(self.device)
                
                output = self.model(x_his, x_ts, x_event, x_future_event)
                
                vali_loss = self.crit(output*self.scale + self.shift,y_true*self.scale + self.shift)
                
                test_loss_all += vali_loss.item()
                
                real_alpha_node = y_true.cpu().detach().numpy()  
                output_alpha_node=output.cpu().detach().numpy()
                
                val_predict_alpha_list.append(output_alpha_node)
                val_real_alpha_list.append(real_alpha_node)
                
                val_predict_value_list.append(np.exp(output_alpha_node*self.scale + self.shift)-1)
                val_real_value_list.append(np.exp(real_alpha_node*self.scale + self.shift)-1)
                
            val_predict_value=np.concatenate(val_predict_value_list)
            val_real_value=np.concatenate(val_real_value_list)
            
            avg_test_loss=test_loss_all / item_count
            vali_losses.append(avg_test_loss)
            
            for i,metric_key in enumerate(self.metric[0]):
                vali_metric[metric_key+'_node_cnt']=self.metric[1][i](val_predict_value,val_real_value)
            
            return vali_losses,vali_metric
        
        
