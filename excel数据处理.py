# 头文件
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
# 导入目标数据
import torch
data_lenth=400
filename = "D:\学习\雏雁计划\数据库.xlsx"
Dataset = pd.read_excel(filename)
# 列
data1= pd.read_excel(filename,usecols='A',header=0,nrows=data_lenth)
data2= pd.read_excel(filename,usecols='B',nrows=data_lenth)
# usecols ='A’时只取表格中A的一列的数据
# usecols = 'A:B’时取表格中A到B列的数据
# usecols = [‘ss’,‘yy’,‘dd’])时取列名为list中的数据
# print(data1)
# print(data2)
#数据预处理
train_data =np.array(data1)
train_data_X=torch.tensor(train_data)
Input_data=torch.empty_like(train_data_X)
# print(Input_data)
for i in range(data_lenth-1):
    if i == 0:
        Input_data[i]=0
    if i != 0:
        Input_data[i]=train_data_X[i]-train_data_X[i-1]
# print(Input_data)
result_data = np.array(data2)
train_data_Y=torch.tensor(result_data)
Output_data=torch.empty(train_data_Y.size(0),5)
# print(Output_data)
for i in range(data_lenth-1):
    if train_data_Y[i]==1:
        Output_data[i]=torch.tensor([1,0,0,0,0])
    elif train_data_Y[i]==2:
        Output_data[i]=torch.tensor([0,1,0,0,0])
    elif train_data_Y[i]==3:
        Output_data[i]=torch.tensor([0,0,1,0,0])
    elif train_data_Y[i]==4:
        Output_data[i]=torch.tensor([0,0,0,1,0])
    elif train_data_Y[i]==5:
        Output_data[i]=torch.tensor([0,0,0,0,1])
# print(Output_data)
# define LSTM neural networks
class LstmRNN(nn.Module):
    def __init__(self,input_size, hidden_size=1,output_size=5,num_layers=1):
        super().__init__()

        self.lstm= nn.LSTM(input_size,hidden_size,num_layers)
        self.forwardCalculation= nn.Linear(hidden_size,output_size)
    def forward(self,_x):
        x,_=self.lstm(_x)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
if __name__ == '__main__':

    train_tensor_in= Input_data.reshape(-1,5,1)
    train_tensor_out=Output_data.reshape(-1,5,5)

    INPUT_FEATURES_NUM=1
    OUTPUT_FEATURES_NUM=5
    lstm_model = LstmRNN(1,1,5,1)
    print('LSTM model:',lstm_model)
    print('model.parameters:',lstm_model.parameters())
    loss_function=nn.MSELoss()
    optimizer= torch.optim.Adam(lstm_model.parameters(),lr=1e-2)


    max_epochs=10000
    for epoch in range(max_epochs):
        output= lstm_model(train_tensor_in)
        loss= loss_function(output,train_tensor_out)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item()<1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
