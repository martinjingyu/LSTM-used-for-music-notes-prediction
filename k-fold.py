#在dataset1的基础上加入kfold
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader
import sys
from sklearn.model_selection import KFold
folder = KFold(n_splits=10, random_state=0, shuffle=True)
#设置最大循环次数
sys.setrecursionlimit(1000000)
#使用gpu
use_gpu=torch.cuda.is_available()
#从csv中导入数据
data_num=3324
sequence_len=15
batch_size=10
train_rate=0.6
input_size=88
hidden_size=16
layer_num=2

filepath='D:\学习\雏雁计划\数据库csv.csv'
dataset=pd.read_csv(filepath,usecols=[0,1],header=None,nrows=data_num)
#将数据转为向量
dataset=np.array(dataset)

print(dataset)
#读取数据第一列为输入
x = dataset[:,0]
#读取数据的数量
m,n=dataset.shape
#对数据进行变形，每十个为一条数据，第一个为0，后面为键位差
x_temp = np.zeros(m)

for i in range(m-1):
    if i==0 or i/10==0:
        continue
    else:
        x_temp[i]=x[i]-x[i-1]
zero=np.zeros((int((sequence_len-1)/2)))
temp_x= np.hstack([zero,x_temp,zero])
#记录时，从中提取sequence_len个，为一个sequence
data_x= np.zeros((m,sequence_len))
for i in range(m):
    data_x[i]=temp_x[i:i+sequence_len]
for i in range(m):
    min_num=data_x[i].min()
    for j in range(len(data_x[i])):
        data_x[i][j]-=min_num
#读取第二列为输出
data_y=dataset[:,1]
data_out = np.zeros((len(data_y), 5))
for i in range(len(data_y) ):
    if data_y[i] == 1:
        data_out[i] = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
    elif data_y[i] == 2:
        data_out[i] = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)
    elif data_y[i] == 3:
        data_out[i] = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
    elif data_y[i] == 4:
        data_out[i] = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)
    elif data_y[i] == 5:
        data_out[i] = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)

def modeol_train(model,loss_function,optimizer,max_epochs,train_iter,val_iter):
    loss_function = loss_function
    optimizer = optimizer


    # for i, data in enumerate(train_iter):
    #     inputdata,label= data
    #     print(inputdata)
    #     print(label)
    #开始模型训练
    for epoch in range(max_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss=0.0
        val_corrects = 0
        val_num = 0
        lstm_model.train()
        for i, data in enumerate(train_iter):
            inputdata, label = data
            # print(inputdata.shape)
            y = model(inputdata)
            pre_lab = torch.argmax(y, 1)
            loss = loss_function(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            # print(y)
            # print(pre_lab)
            # print(torch.argmax(label,1))
            train_corrects += torch.sum(pre_lab == torch.argmax(label, 1).data)
            train_num += len(label)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{}Train Loss:{:.4f}  Train Acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        lstm_model.eval()
        for i, data in enumerate(val_iter):
            inputdata, label = data
            # print(inputdata.shape)
            y = model(inputdata)
            pre_lab = torch.argmax(y, 1)
            loss = loss_function(y, label)
            val_loss += loss.item()*len(label)
            val_corrects += torch.sum(pre_lab ==torch.argmax(label,1).data)
            val_num +=len(label)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print('{}Val Loss:{:.4f}  Val Acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
#定义dataset类
class Mydata(Dataset):
    def __init__(self, datax, datay):
        datax = datax.astype('float32')
        datay = datay.astype('float32')
        datax = torch.from_numpy(datax)
        #将x重构为一个二维的张量
        one, two = datax.shape
        datax = datax.view(one, two)
        print(datax.shape)
        self.x = datax
        self.y = datay
        one, two = datay.shape
        self.len = one

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    #
    def __len__(self):
        return self.len
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(176,88)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fcl = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.int()
        x = self.embed(x)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fcl(r_out[:, -1, :])
        return out
train_num=int(train_rate*data_num)
data1=range(2964)#布格缪勒
data2=range(2965,3323)#tatto for lovers
data3=range(3324,3723)#大调琶音
data4=range(3724,4822)#指法
max_epochs = 15
train_loss_all = []
train_acc_all = []
val_loss_all = []
val_acc_all = []
acurency_for_kfold=[]
for train ,test in folder.split(data_x):
    # print(train,test)
    data_train_x,data_test_x=data_x[train],data_x[test]
    data_train_y,data_test_y=data_out[train],data_out[test]
    print("-----")
    print(data_train_x)
    # data_train_x=np.zeros(train_long)
    # for i in range(train_long):
    #     data_train_x[i]=data_x[train[i]]
    dataset_train_after = Mydata(data_train_x, data_train_y)
    dataset_test_after = Mydata(data_test_x, data_test_y)
    train_iter = DataLoader(dataset_train_after, batch_size=batch_size, drop_last=True, shuffle=True)
    test_iter = DataLoader(dataset_test_after, batch_size=batch_size, drop_last=True, shuffle=True)
    lstm_model = LSTM(input_size, hidden_size, 5, 2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    modeol_train(lstm_model,loss_function,optimizer,max_epochs,train_iter,test_iter)
    acurency_for_kfold.append(train_acc_all[-1])
print(acurency_for_kfold)
average_acc=acurency_for_kfold.sum()/len(acurency_for_kfold)
print("the average is :{:.4f}".format(average_acc))

# dataset_train_after = Mydata(data_train_x, data_train_y)
# dataset_val_after = Mydata(data_val_x,data_val_y)

# train_iter = DataLoader(dataset_train_after, batch_size=batch_size, drop_last=True,shuffle=True)
# val_iter = DataLoader(dataset_val_after,batch_size=batch_size,drop_last=True,shuffle=True)
# modeol_train(lstm_model,loss_function,optimizer,max_epochs,train_iter,val_iter)